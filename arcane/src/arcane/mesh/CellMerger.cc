// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CellMerger.cc                                               (C) 2000-2025 */
/*                                                                           */
/* Merges two meshes.                                                        */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/mesh/CellMerger.h"

#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/StringBuilder.h"
#include "arcane/utils/TraceInfo.h"
#include "arcane/utils/CheckedConvert.h"
#include "arcane/utils/TraceAccessor.h"

#include "arcane/core/Item.h"
#include "arcane/core/ItemEnumerator.h"
#include "arcane/core/IItemFamily.h"
#include "arcane/core/IItemFamilyTopologyModifier.h"
#include "arcane/core/IMesh.h"
#include "arcane/core/internal/IItemFamilyInternal.h"

#include "arcane/mesh/FaceReorienter.h"

#include <map>
#include <set>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Utility class for swapping entities between two entities
class ItemSwapperUtils
: public TraceAccessor
{
 public:

  explicit ItemSwapperUtils(IMesh* mesh)
  : TraceAccessor(mesh->traceMng())
  , m_face_reorienter(mesh)
  , m_cell_tm(mesh->cellFamily()->_internalApi()->topologyModifier())
  , m_face_tm(mesh->faceFamily()->_internalApi()->topologyModifier())
  , m_node_tm(mesh->nodeFamily()->_internalApi()->topologyModifier())
  {
  }

 public:

  /*!
   * \brief Swaps two nodes between two faces.
   *
   * Swaps the node at index \a face1_node_idx of face \a face1 with the
   * node at index \a face2_node_idx of face \a face2.
   */
  void swapFaceNodes(Face face_1, Face face_2,
                     Integer face1_node_idx, Integer face2_node_idx)
  {
    NodeLocalId face1_node = face_1.node(face1_node_idx);
    NodeLocalId face2_node = face_2.node(face2_node_idx);

    m_face_tm->replaceNode(face_1, face1_node_idx, face2_node);
    m_face_tm->replaceNode(face_2, face2_node_idx, face1_node);

    m_node_tm->findAndReplaceFace(face1_node, face_1, face_2);
    m_node_tm->findAndReplaceFace(face2_node, face_2, face_1);
  }

  /*!
   * \brief Swaps two nodes between two cells.
   *
   * Swaps the node at index \a cell1_node_idx of cell \a cell1 with the
   * node at index \a cell2_node_idx of cell \a cell2.
   */
  void swapCellNodes(Cell cell1, Cell cell2,
                     Integer cell1_node_idx, Integer cell2_node_idx)
  {
    NodeLocalId cell1_node = cell1.node(cell1_node_idx);
    NodeLocalId cell2_node = cell2.node(cell2_node_idx);

    m_cell_tm->replaceNode(cell1, cell1_node_idx, cell2_node);
    m_cell_tm->replaceNode(cell2, cell2_node_idx, cell1_node);

    m_node_tm->findAndReplaceCell(cell1_node, cell1, cell2);
    m_node_tm->findAndReplaceCell(cell2_node, cell2, cell1);
  }

  /*!
   * \brief Swaps two faces between two cells.
   *
   * Swaps the face at index \a cell1_face_idx of cell \a cell1 with the
   * face at index \a cell2_face_idx of cell \a cell2.
   */
  void swapCellFaces(Cell cell1, Cell cell2,
                     Integer cell1_face_idx, Integer cell2_face_idx)
  {
    FaceLocalId cell1_face = cell1.face(cell1_face_idx);
    FaceLocalId cell2_face = cell2.face(cell2_face_idx);

    m_cell_tm->replaceFace(cell1, cell1_face_idx, cell2_face);
    m_cell_tm->replaceFace(cell2, cell2_face_idx, cell1_face);

    m_face_tm->findAndReplaceCell(cell1_face, cell1, cell2);
    m_face_tm->findAndReplaceCell(cell2_face, cell2, cell1);
  }

  void checkAndChangeFaceOrientation(Cell cell)
  {
    // This could undoubtedly be optimized
    for (Integer i = 0, n = cell.nbFace(); i < n; ++i) {
      m_face_reorienter.checkAndChangeOrientation(cell.face(i));
    }
  }

 private:

  FaceReorienter m_face_reorienter;
  IItemFamilyTopologyModifier* m_cell_tm;
  IItemFamilyTopologyModifier* m_face_tm;
  IItemFamilyTopologyModifier* m_node_tm;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Finds the common face between two cells.
 *
 * This utility class aims to determine the common face between two
 * cells. If the two cells are not joined by a face, an error is
 * generated.
 */
class CommonFaceFinder
{
 public:

  typedef std::set<Integer> NodesLIDSet;

 private:

  Integer m_cell_1_local_number; //!< Number of the common face in cell 1
  Integer m_cell_2_local_number; //!< Number of the common face in cell 2

  NodesLIDSet m_nodes_lid_set; //! Set of localIds of common nodes

 public:

  /*!
   * Read access to the list of common node localIds
   *
   * @return m_nodes_lid_set
   */
  const NodesLIDSet& nodesLID() const
  {
    return m_nodes_lid_set;
  }

  /*!
   * Read access to the local number of the common face in
   * cell 1
   *
   * @return m_cell_1_local_number
   */
  Integer cell1LocalNumber() const
  {
    return m_cell_1_local_number;
  }

  /*!
   * Read access to the local number of the common face in
   * cell 2
   *
   * @return m_cell_2_local_number
   */
  Integer cell2LocalNumber() const
  {
    return m_cell_2_local_number;
  }

  /*!
   * Constructor. All data structures are generated upon calling the constructor
   *
   * @param i_cell_1 the 1st cell
   * @param i_cell_2 the 2nd cell
   */
  CommonFaceFinder(Cell i_cell_1, Cell i_cell_2);
  ~CommonFaceFinder() {}
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CommonFaceFinder::
CommonFaceFinder(Cell i_cell_1, Cell i_cell_2)
: m_cell_1_local_number(-1)
, m_cell_2_local_number(-1)
{
  typedef std::map<Integer, Integer> LIDCellMapping;
  LIDCellMapping faces1; // number of faces in cell 1
  LIDCellMapping faces2; // number of faces in cell 2

  { // creation of localId to number mappings in the cell for cell 1
    Integer n = 0;
    for (ItemEnumerator i_face(i_cell_1.faces()); i_face(); ++i_face) {
      faces1[i_face->localId()] = n;
      n++;
    }
  }
  { // creation of localId to number mappings in the cell for cell 2
    Integer n = 0;
    for (ItemEnumerator i_face(i_cell_2.faces()); i_face(); ++i_face) {
      faces2[i_face->localId()] = n;
      n++;
    }
  }

  // We now iterate through the two tables created previously. Since
  // they are sorted by increasing localId, we simply deduce
  // the common face.
  LIDCellMapping::const_iterator i_face1 = faces1.begin();
  LIDCellMapping::const_iterator i_face2 = faces2.begin();

  do {
    const Integer& lid1 = i_face1->first; // localId of the face in list 1
    const Integer& lid2 = i_face2->first; // localId of the face in list 2

    if (lid1 == lid2) { // we found the common face
      m_cell_1_local_number = i_face1->second;
      m_cell_2_local_number = i_face2->second;

      // Recording the localIds of the nodes of the common face
      Face common_face = i_cell_1.face(m_cell_1_local_number);

      for (NodeEnumerator i_node(common_face.nodes()); i_node(); ++i_node) {
        m_nodes_lid_set.insert(i_node->localId());
      }
      return;
    }
    else {
      if (lid1 < lid2) {
        ++i_face1;
      }
      else {
        ++i_face2;
      }
    }
  } while (i_face1 != faces1.end() && i_face2 != faces2.end());

  ARCANE_FATAL("Common face not found!");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Merges two faces in 2D (in fact, two edges).
 */
class Faces2DMerger
{
 private:

  Integer m_face_1_common_node_numbers; /**< Numbers in face 1 of nodes common with face 2 */
  Integer m_face_2_common_node_numbers; /**< Numbers in face 2 of nodes common with face 1 */
  Integer m_face_2_exchanged_node_numbers; /**< Numbers in face 2 of nodes that will define the merged face */

  /**
   * Initialization of m_face_1_common_node_numbers, m_face_2_common_node_numbers, and
   * m_face_2_exchanged_node_numbers
   *
   * @param i_face_1 face 1
   * @param i_face_2 face 2
   */
  void _setFacesNodeNumbers(Face i_face_1, Face i_face_2);

 public:

  /**
   * Constructor
   *
   * @param i_face_1 the face to keep
   * @param i_face_2 the abandoned face
   */
  Faces2DMerger(ItemSwapperUtils* swap_utils, Face i_face_1, Face i_face_2);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Faces2DMerger::
_setFacesNodeNumbers(Face i_face_1, Face i_face_2)
{
  typedef std::map<Integer, Integer> LocalIDToLocalNumber;
  LocalIDToLocalNumber face1_node_localId;
  LocalIDToLocalNumber face2_node_localId;

  {
    Integer n = 0;
    for (ItemEnumerator i_node(i_face_1.nodes()); i_node(); ++i_node) {
      face1_node_localId[i_node->localId()] = n++;
    }
  }
  {
    Integer n = 0;
    for (ItemEnumerator i_node(i_face_2.nodes()); i_node(); ++i_node) {
      face2_node_localId[i_node->localId()] = n++;
    }
  }

  //Integer face2_common_edge_node_number = std::numeric_limits<Integer>::max();

  for (LocalIDToLocalNumber::const_iterator
       i = face1_node_localId.begin(),
       j = face2_node_localId.begin();
       i != face1_node_localId.end() && j != face2_node_localId.end();) {
    Int32 node1_localId = i->first;
    Int32 node2_localId = j->first;
    if (node1_localId == node2_localId) {
      m_face_1_common_node_numbers = i->second;
      m_face_2_common_node_numbers = j->second;
      break;
    }
    else {
      if (node1_localId < node2_localId) {
        ++i;
      }
      else {
        ++j;
      }
    }
  }

  m_face_2_exchanged_node_numbers = (m_face_2_common_node_numbers + 1) % 2;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Faces2DMerger::
Faces2DMerger(ItemSwapperUtils* swap_utils, Face face1, Face face2)
: m_face_1_common_node_numbers(std::numeric_limits<Integer>::max())
, m_face_2_common_node_numbers(std::numeric_limits<Integer>::max())
, m_face_2_exchanged_node_numbers(std::numeric_limits<Integer>::max())
{
  ARCANE_ASSERT(face2.type() == IT_Line2, ("The cell is not a line"));

  _setFacesNodeNumbers(face1, face2);

  swap_utils->swapFaceNodes(face1, face2, m_face_1_common_node_numbers,
                            m_face_2_exchanged_node_numbers);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief In dimension 2, finds common faces between two cells
 * (The faces are actually edges).
 */
class Faces2DToMergeFinder
{
 private:

  //@{
  /**
   * These lists containing the numbers of the common faces of the two
   * cells are constructed so that the face
   * m_cell1_edge_face_list[i] is to be merged with the face
   * m_cell2_edge_face_list[i]
   */
  IntegerUniqueArray m_cell1_edge_face_list;
  IntegerUniqueArray m_cell2_edge_face_list;
  //@}

  /**
   * Determines the faces to be merged by considering the edges that
   * are supported by the common nodes between the cells. For this,
   * the faces are sorted according to the edges supported by common nodes.
   *
   * @param i_cell the cell studied
   * @param edge_face_list the list of face numbers
   * @param common_face_number the number of the common face in cell @a i_cell
   * @param common_face_nodes the localIds of the nodes of the common face
   */
  void _setEdgeFaceList(Cell i_cell,
                        IntegerArray& edge_face_list,
                        Integer common_face_number,
                        const CommonFaceFinder::NodesLIDSet& common_face_nodes)
  {
    typedef Integer _EdgeDescriptor;
    typedef std::map<_EdgeDescriptor, Integer> _EdgeFaceList;

    _EdgeFaceList temp_edge_face_list;
    Integer face_number = 0;
    // For each face in the cell
    for (FaceEnumerator i_face(i_cell.faces()); i_face(); ++i_face, ++face_number) {
      if (face_number == common_face_number) {
        continue; // if the face is the common face, do nothing
      }

      // Create the list of nodes of this face that are common
      std::multiset<Integer> node_list;
      for (NodeEnumerator i_node(i_face->nodes()); i_node(); ++i_node) {
        const Integer& node_lid = i_node->localId();
        if (common_face_nodes.find(node_lid) != common_face_nodes.end()) {
          node_list.insert(node_lid);
        }
      }

      switch (node_list.size()) {
      case 0: // the face is not to be retriangled [already processed]
      case 2:
        continue;
      case 1: { // If the list contains only one element, it means the face is to be merged
        std::multiset<Integer>::const_iterator i = node_list.begin();
        const Integer node_lid = *i;
        temp_edge_face_list[node_lid] = face_number;
        break;
      }
      default: {
        ARCANE_FATAL("Unexpected number of nodes on the common face !");
      }
      }
    }

    // We copy the data: we no longer need the edges
    edge_face_list.reserve((Integer)temp_edge_face_list.size());
    for (_EdgeFaceList::const_iterator i = temp_edge_face_list.begin();
         i != temp_edge_face_list.end(); ++i) {
      edge_face_list.add(i->second); // we store the numbers of the faces to be merged
    }
  }

 public:

  /**
   * Read-only access to the number of common edges
   *
   * @return m_cell1_edge_face_list.size()
   */
  Integer getNumber() const
  {
    return m_cell1_edge_face_list.size();
  }

  /**
   * Access the number in mesh 1 of the @a i-th face to be merged.
   *
   * @param i number in the list of faces to be merged
   *
   * @return the number of the @a i-th face to be merged
   */
  Integer cell1FaceNumber(Integer i) const
  {
    return m_cell1_edge_face_list[i];
  }

  /**
   * Access the number in mesh 2 of the @a i-th face to be merged.
   *
   * @param i number in the list of faces to be merged
   *
   * @return the number of the @a i-th face to be merged
   */
  Integer cell2FaceNumber(Integer i) const
  {
    return m_cell2_edge_face_list[i];
  }

  /**
   * Constructs the different data structures.
   *
   * @param i_cell_1 the first mesh
   * @param i_cell_2 the second mesh
   * @param common_face the information on the common face
   */
  Faces2DToMergeFinder(Cell cell1, Cell cell2,
                       const CommonFaceFinder& common_face)
  {
    this->_setEdgeFaceList(cell1,
                           m_cell1_edge_face_list,
                           common_face.cell1LocalNumber(),
                           common_face.nodesLID());
    this->_setEdgeFaceList(cell2,
                           m_cell2_edge_face_list,
                           common_face.cell2LocalNumber(),
                           common_face.nodesLID());

    ARCANE_ASSERT(m_cell1_edge_face_list.size() == m_cell2_edge_face_list.size(),
                  ("Incompatible number of 2D faces to merge !"));
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief This function-class searches for faces to merge when
 * merging two meshes
 */
class FacesToMergeFinder
{
 public:

  /**
   * Access the number of common faces between the two meshes
   *
   * @return m_cell1_edge_face_list.size()
   */
  Integer getNumber() const
  {
    return m_cell1_edge_face_list.size();
  }

  /**
   * Access the @a i-th face number in mesh 1
   *
   * @param i the number in the list of common faces
   *
   * @return the number in mesh 1 of the @a i-th common face
   */
  Integer cell1FaceNumber(Integer i) const
  {
    return m_cell1_edge_face_list[i];
  }

  /**
   * Access the @a i-th face number in mesh 2
   *
   * @param i the number in the list of common faces
   *
   * @return the number in mesh 2 of the @a i-th common face
   */
  Integer cell2FaceNumber(Integer i) const
  {
    return m_cell2_edge_face_list[i];
  }

  /**
   * Constructs the different data structures
   *
   * @param cell1 the first mesh
   * @param cell2 the second mesh
   * @param common_face the information of the common face
   */
  FacesToMergeFinder(Cell cell1, Cell cell2,
                     const CommonFaceFinder& common_face)
  {
    _setEdgeFaceList(cell1, m_cell1_edge_face_list,
                     common_face.cell1LocalNumber(), common_face.nodesLID());
    _setEdgeFaceList(cell2, m_cell2_edge_face_list,
                     common_face.cell2LocalNumber(), common_face.nodesLID());

    ARCANE_ASSERT(m_cell1_edge_face_list.size() == m_cell2_edge_face_list.size(),
                  ("Incompatible number of faces to merge !"));
  }

 private:

  //@{
  /**
   * These lists containing the numbers of the common faces of the two
   * meshes are constructed such that the face
   * m_cell1_edge_face_list[i] is to be merged with the face
   * m_cell2_edge_face_list[i]
   */
  IntegerUniqueArray m_cell1_edge_face_list;
  IntegerUniqueArray m_cell2_edge_face_list;
  //@}
  void _setEdgeFaceList(Cell i_cell,
                        IntegerArray& edge_face_list,
                        Integer common_face_number,
                        const CommonFaceFinder::NodesLIDSet& common_face_nodes);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/**
 * Determines the faces to merge by considering the edges that
 * are supported by the common vertices between the meshes. For this,
 * the faces are sorted according to the edges supported by common vertices.
 *
 * @param i_cell the mesh studied
 * @param edge_face_list the list of face numbers
 * @param common_face_number the number of the common face in mesh @a i_cell
 * @param common_face_nodes the localIds of the nodes of the common face
 */
void FacesToMergeFinder::
_setEdgeFaceList(Cell i_cell,
                 IntegerArray& edge_face_list,
                 Integer common_face_number,
                 const CommonFaceFinder::NodesLIDSet& common_face_nodes)
{
  typedef std::pair<Integer, Integer> _EdgeDescriptor;
  typedef std::map<_EdgeDescriptor, Integer> _EdgeFaceList;

  _EdgeFaceList temp_edge_face_list; // sorted list of face numbers by edges
  Integer face_number = 0;
  for (FaceEnumerator i_face(i_cell.faces()); i_face(); ++i_face, ++face_number) {
    if (face_number == common_face_number) { // if the face is the common face, it is not processed
      continue;
    }

    std::multiset<Integer> node_list; // list of localIds of the nodes of the face that are common
    for (NodeEnumerator i_node(i_face->nodes()); i_node(); ++i_node) {
      Int32 node_lid = i_node->localId();
      // if the node is common, add it to the list of common nodes
      if (common_face_nodes.find(node_lid) != common_face_nodes.end()) {
        node_list.insert(node_lid);
      }
    }

    switch (node_list.size()) {
    case 0: // the face is not to be retriangled [already processed]
    case 4:
      continue;
    case 2: {
      std::multiset<Integer>::const_iterator i = node_list.begin();
      const Integer first_node_lid = *i;
      ++i;
      const Integer second_node_lid = *i;
      // We create the edge/face correspondence (the nodes of the edge being sorted)
      temp_edge_face_list[std::make_pair(first_node_lid, second_node_lid)] = face_number;
      break;
    }
    default:
      ARCANE_FATAL("Unexpected number of nodes on the common face !");
    }
  }

  // We copy the data: we no longer need the edges. The
  // face numbers are oriented as desired.
  edge_face_list.reserve(CheckedConvert::toInteger(temp_edge_face_list.size()));
  for (_EdgeFaceList::const_iterator i = temp_edge_face_list.begin();
       i != temp_edge_face_list.end(); ++i) {
    edge_face_list.add(i->second);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief This function-class aims to merge two faces, where
 * the second is necessarily a quadrangle.
 */
class FaceToQuadrilateralMerger
{
 private:

  IntegerUniqueArray m_face_1_common_node_numbers; /**< Numbers in face 1 of the common vertices with face 2 */
  IntegerUniqueArray m_face_2_common_node_numbers; /**< Numbers in face 1 of the common vertices with face 2 */
  IntegerUniqueArray m_face_2_exchanged_node_numbers; /**< Numbers in face 2 of the vertices that will define the merged mesh */

  static const Integer m_quad_node_neighbors[4][2]; /**< List of neighboring nodes by edge in a quadrangle */

  /**
   * Initializes the quantities m_face_1_common_node_numbers m_face_2_common_node_numbers and
   * m_face_2_exchanged_node_numbers
   *
   * @param i_face_1 the first face
   * @param i_face_2 the second face
   */
  bool _setFacesNodeNumbers(Face i_face_1, Face i_face_2);

 public:

  /**
   * Constructor
   *
   * @param face1 the receiving face
   * @param face2 the abandoned face [MUST BE A QUADANGLE]
   */
  FaceToQuadrilateralMerger(ItemSwapperUtils* swap_utils, Face face1, Face face2);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool FaceToQuadrilateralMerger::
_setFacesNodeNumbers(Face i_face_1, Face i_face_2)
{
  typedef std::map<Integer, Integer> LocalIDToLocalNumber;
  LocalIDToLocalNumber face1_node_localId;
  LocalIDToLocalNumber face2_node_localId;

  {
    Integer n = 0;
    for (NodeEnumerator i_node(i_face_1.nodes()); i_node(); ++i_node) {
      face1_node_localId[i_node->localId()] = n++;
    }
  }
  {
    Integer n = 0;
    for (NodeEnumerator i_node(i_face_2.nodes()); i_node(); ++i_node) {
      face2_node_localId[i_node->localId()] = n++;
    }
  }

  m_face_1_common_node_numbers.reserve(2);
  m_face_2_common_node_numbers.reserve(2);

  std::set<Integer> face2_common_edge_node_number;

  for (LocalIDToLocalNumber::const_iterator
       i = face1_node_localId.begin(),
       j = face2_node_localId.begin();
       i != face1_node_localId.end() && j != face2_node_localId.end();) {
    const Integer& node1_localId = i->first;
    const Integer& node2_localId = j->first;
    if (node1_localId == node2_localId) {
      m_face_1_common_node_numbers.add(i->second);
      m_face_2_common_node_numbers.add(j->second);
      face2_common_edge_node_number.insert(j->second);
      ++i;
      ++j;
    }
    else {
      if (node1_localId < node2_localId) {
        ++i;
      }
      else {
        ++j;
      }
    }
  }

  if (m_face_1_common_node_numbers.size() == 0)
    return false;

  ARCANE_ASSERT((m_face_2_common_node_numbers.size() == 2) && (m_face_1_common_node_numbers.size() == 2),
                ("Incorrect number of shared vertices"));

  m_face_2_exchanged_node_numbers.reserve(2);
  for (Integer i = 0; i < m_face_2_common_node_numbers.size(); ++i) {
    const Integer& node_number = m_face_2_common_node_numbers[i];
    for (Integer j = 0; j < 2; ++j) {
      const Integer& edge_node = m_quad_node_neighbors[node_number][j];
      if (face2_common_edge_node_number.find(edge_node) == face2_common_edge_node_number.end()) {
        m_face_2_exchanged_node_numbers.add(edge_node);
        break;
      }
    }
  }
  return true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

FaceToQuadrilateralMerger::
FaceToQuadrilateralMerger(ItemSwapperUtils* swap_utils, Face face1, Face face2)
{
  ARCANE_ASSERT(face2.type() == IT_Quad4, ("The cell is not a quadrangle"));

  if (_setFacesNodeNumbers(face1, face2)) {
    ARCANE_ASSERT(m_face_2_exchanged_node_numbers.size() == 2,
                  ("Incorrect number of exchange vertices"));

    // Exchange of face nodes
    for (Integer i = 0; i < 2; ++i) {
      swap_utils->swapFaceNodes(face1, face2, m_face_1_common_node_numbers[i],
                                m_face_2_exchanged_node_numbers[i]);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const Integer
FaceToQuadrilateralMerger::
m_quad_node_neighbors[4][2] = { { 1, 3 }, { 0, 2 }, { 1, 3 }, { 0, 2 } };

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class CellToQuadrilateralMerger
{
 private:

  //! Numbers in mesh 1 of the common vertices with mesh 2
  IntegerUniqueArray m_cell_1_common_node_numbers;
  //! Numbers in mesh 2 of the common vertices with mesh 1
  IntegerUniqueArray m_cell_2_common_node_numbers;
  //! Numbers in mesh 2 of the vertices that will define the merged mesh
  IntegerUniqueArray m_cell_2_exchanged_node_numbers;

  //! List of neighboring nodes by edge in a quadrangle
  static const Integer m_quad_node_neighbors[4][2];

  /**
   * Initializes the quantities m_cell_1_common_node_numbers m_cell_2_common_node_numbers and
   * m_cell_2_exchanged_node_numbers
   *
   * @param i_cell_1 the first mesh
   * @param i_cell_2 the second mesh
   */
  void _setCellsNodeNumbers(Cell i_cell_1, Cell i_cell_2);

 public:

  /**
   * Constructor
   *
   * @param cell1 the retained mesh
   * @param cell2 the abandoned mesh [MUST BE A QUADANGLE]
   */
  CellToQuadrilateralMerger(ItemSwapperUtils* swap_utils, Cell cell1, Cell cell2);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CellToQuadrilateralMerger::
_setCellsNodeNumbers(Cell i_cell_1, Cell i_cell_2)
{
  typedef std::map<Integer, Integer> LocalIDToLocalNumber;
  LocalIDToLocalNumber cell1_node_localId;
  LocalIDToLocalNumber cell2_node_localId;

  {
    Integer n = 0;
    for (NodeEnumerator i_node(i_cell_1.nodes()); i_node(); ++i_node) {
      cell1_node_localId[i_node->localId()] = n++;
    }
  }
  {
    Integer n = 0;
    for (NodeEnumerator i_node(i_cell_2.nodes()); i_node(); ++i_node) {
      cell2_node_localId[i_node->localId()] = n++;
    }
  }

  std::set<Integer> cell2_common_edge_node_number;
  for (LocalIDToLocalNumber::const_iterator
       i = cell1_node_localId.begin(),
       j = cell2_node_localId.begin();
       i != cell1_node_localId.end() && j != cell2_node_localId.end();) {
    const Integer& node1_localId = i->first;
    const Integer& node2_localId = j->first;
    if (node1_localId == node2_localId) {
      m_cell_1_common_node_numbers.add(i->second);
      m_cell_2_common_node_numbers.add(j->second);
      cell2_common_edge_node_number.insert(j->second);
      ++i;
      ++j;
    }
    else {
      if (node1_localId < node2_localId) {
        ++i;
      }
      else {
        ++j;
      }
    }
  }

  ARCANE_ASSERT(m_cell_1_common_node_numbers.size() == 2,
                ("Bad number of shared vertices"));

  m_cell_2_exchanged_node_numbers.reserve(2);
  for (Integer i = 0; i < m_cell_2_common_node_numbers.size(); ++i) {
    const Integer& node_number = m_cell_2_common_node_numbers[i];
    for (Integer j = 0; j < 2; ++j) {
      const Integer& edge_node = m_quad_node_neighbors[node_number][j];
      if (cell2_common_edge_node_number.find(edge_node) == cell2_common_edge_node_number.end()) {
        m_cell_2_exchanged_node_numbers.add(edge_node);
        break;
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CellToQuadrilateralMerger::
CellToQuadrilateralMerger(ItemSwapperUtils* swap_utils, Cell cell1, Cell cell2)
{
  ARCANE_ASSERT(cell2.type() == IT_Quad4, ("Cell2 is not a IT_Quad4"));

  CommonFaceFinder common_face(cell1, cell2);

  this->_setCellsNodeNumbers(cell1, cell2);

  // Fusion of side meshes
  Faces2DToMergeFinder faces_to_merge(cell1, cell2, common_face);
  for (Integer i = 0; i < faces_to_merge.getNumber(); ++i) {
    Faces2DMerger(swap_utils,
                  cell1.face(faces_to_merge.cell1FaceNumber(i)),
                  cell2.face(faces_to_merge.cell2FaceNumber(i)));
  }

  // Face exchange.
  swap_utils->swapCellFaces(cell1, cell2,
                            common_face.cell1LocalNumber(),
                            (common_face.cell2LocalNumber() + 2) % 4); // opposite face

  // Mesh vertex exchange
  for (Integer i = 0, n = m_cell_1_common_node_numbers.size(); i < n; ++i) {
    swap_utils->swapCellNodes(cell1, cell2,
                              m_cell_1_common_node_numbers[i],
                              m_cell_2_exchanged_node_numbers[i]);
  }

  swap_utils->checkAndChangeFaceOrientation(cell1);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const Integer CellToQuadrilateralMerger::m_quad_node_neighbors[4][2] = { { 1, 3 }, { 0, 2 }, { 1, 3 }, { 0, 2 } };

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief This function-class aims to merge two meshes
 * where the second one is necessarily a hexahedron
 */
class CellToHexahedronMerger
{
 private:

  IntegerUniqueArray m_cell_1_common_node_numbers; /**< Numbers in mesh 1 of vertices common with mesh 2 */
  IntegerUniqueArray m_cell_2_common_node_numbers; /**< Numbers in mesh 2 of vertices common with mesh 1 */
  IntegerUniqueArray m_cell_2_exchanged_node_numbers; /**< Numbers in mesh 2 of vertices that will define the merged mesh */

  static const Integer m_hexa_node_neighbors[8][3]; /**< List of neighboring nodes by edge in a hexahedron */

  /**
   * Initialization of m_cell_1_common_node_numbers m_cell_2_common_node_numbers and
   * m_cell_2_exchanged_node_numbers
   *
   * @param cell1 mesh 1
   * @param cell2 mesh 2
   */
  void _setCellsNodeNumbers(Cell cell1, Cell cell2);

 public:

  /**
   * Constructor
   *
   * @param cell1 the mesh to keep
   * @param cell2 the discarded mesh [MUST BE A HEXAEDRON]
   */
  CellToHexahedronMerger(ItemSwapperUtils* swap_utils, Cell cell1, Cell cell2);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CellToHexahedronMerger::
_setCellsNodeNumbers(Cell cell1, Cell cell2)
{
  typedef std::map<Integer, Integer> LocalIDToLocalNumber;
  LocalIDToLocalNumber cell1_node_localId;
  LocalIDToLocalNumber cell2_node_localId;

  // We associate the numbers (in the mesh) of the nodes with their
  // localId. These lists are sorted by localId!
  {
    // first for mesh 1
    Integer n = 0;
    for (NodeEnumerator i_node(cell1.nodes()); i_node(); ++i_node) {
      cell1_node_localId[i_node->localId()] = n++;
    }
  }
  {
    // then for mesh 2
    Integer n = 0;
    for (NodeEnumerator i_node(cell2.nodes()); i_node(); ++i_node) {
      cell2_node_localId[i_node->localId()] = n++;
    }
  }

  // We then determine the set of common nodes between the two
  // meshes
  std::set<Integer> cell2_common_edge_node_number;
  for (LocalIDToLocalNumber::const_iterator
       i = cell1_node_localId.begin(),
       j = cell2_node_localId.begin();
       i != cell1_node_localId.end() && j != cell2_node_localId.end();) {
    Integer node1_localId = i->first;
    Integer node2_localId = j->first;
    if (node1_localId == node2_localId) { // if the nodes are the same
      // we store the numbers in the meshes of these vertices
      m_cell_1_common_node_numbers.add(i->second); // for mesh 1
      m_cell_2_common_node_numbers.add(j->second); // for mesh 2

      // and we create the ordered set of common nodes in the
      // second mesh
      cell2_common_edge_node_number.insert(j->second);
      ++i;
      ++j;
    }
    else {
      if (node1_localId < node2_localId) {
        ++i;
      }
      else {
        ++j;
      }
    }
  }

  ARCANE_ASSERT(m_cell_1_common_node_numbers.size() == 4,
                ("Bad number of shared vertices"));

  // We are now looking for the neighbors of the common nodes
  // belonging to the second mesh and which are not exchanged vertices.
  // These are the vertices that will form the new mesh
  // by substitution with the common vertices of the first mesh.
  m_cell_2_exchanged_node_numbers.reserve(4);
  for (Integer i = 0; i < m_cell_2_common_node_numbers.size(); ++i) {
    const Integer& node_number = m_cell_2_common_node_numbers[i];
    for (Integer j = 0; j < 3; ++j) {
      const Integer& edge_node = m_hexa_node_neighbors[node_number][j];
      if (cell2_common_edge_node_number.find(edge_node) == cell2_common_edge_node_number.end()) {
        m_cell_2_exchanged_node_numbers.add(edge_node);
        break;
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CellToHexahedronMerger::
CellToHexahedronMerger(ItemSwapperUtils* swap_utils, Cell cell1, Cell cell2)
{
  // TODO: merge this code with CellToQuadrilateralMerger.

  ARCANE_ASSERT(cell2.type() == IT_Hexaedron8, ("Cell2 is not a IT_Hexaedron8"));

  CommonFaceFinder common_face(cell1, cell2);

  this->_setCellsNodeNumbers(cell1, cell2);

  // Fusion of side meshes
  FacesToMergeFinder faces_to_merge(cell1, cell2, common_face);
  for (Integer i = 0; i < faces_to_merge.getNumber(); ++i) {
    FaceToQuadrilateralMerger(swap_utils,
                              cell1.face(faces_to_merge.cell1FaceNumber(i)),
                              cell2.face(faces_to_merge.cell2FaceNumber(i)));
  }

  // Face exchange.
  swap_utils->swapCellFaces(cell1, cell2,
                            common_face.cell1LocalNumber(),
                            (common_face.cell2LocalNumber() + 3) % 6); // opposite face

  // Mesh vertex exchange
  for (Integer i = 0, n = m_cell_1_common_node_numbers.size(); i < n; ++i) {
    swap_utils->swapCellNodes(cell1, cell2,
                              m_cell_1_common_node_numbers[i],
                              m_cell_2_exchanged_node_numbers[i]);
  }

  swap_utils->checkAndChangeFaceOrientation(cell1);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const Integer
CellToHexahedronMerger::
m_hexa_node_neighbors[8][3] = { { 1, 3, 4 }, { 0, 2, 5 }, { 1, 3, 6 }, { 0, 2, 7 }, { 0, 5, 7 }, { 1, 4, 6 }, { 2, 5, 7 }, { 3, 4, 6 } };

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String CellMerger::
_typeName(const CellMerger::_Type& t) const
{
  switch (t) {
  case Hexahedron:
    return "hexahedron";
  case Pyramid:
    return "pyramid";
  case Pentahedron:
    return "pentahedron";
  case Quadrilateral:
    return "quadrangle";
  case Triangle:
    return "triangle";
  default:
    return "unknown";
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CellMerger::_Type CellMerger::
_getCellType(const Integer& internal_cell_type) const
{
  switch (internal_cell_type) {
  case IT_Hexaedron8: {
    return Hexahedron;
  }
  case IT_Pyramid5: {
    return Pyramid;
  }
  case IT_Pentaedron6: {
    return Pentahedron;
  }
  case IT_Quad4: {
    return Quadrilateral;
  }
  case IT_Triangle3: {
    return Triangle;
  }
  default: {
    return NotMergeable;
  }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CellMerger::_Type CellMerger::
_promoteType(const _Type& t1, const _Type& t2) const
{
  switch (t1 * t2) {
  case 1:
    return Hexahedron;
  case 2:
    return Pyramid;
  case 3:
    return Pentahedron;
  case 100:
    return Quadrilateral;
  case 110:
    return Triangle;
  default:
    ARCANE_FATAL("Can not merge cells of type {0} and {1}", _typeName(t1), _typeName(t2));
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CellMerger::
merge(Cell i_cell_1, Cell i_cell_2)
{
  _Type cell_1_type = _getCellType(i_cell_1.type());
  IMesh* mesh = i_cell_1.itemFamily()->mesh();
  ItemSwapperUtils swap_utils(mesh);

  switch (cell_1_type) {
  case Hexahedron:
  case Pyramid:
  case Pentahedron: {
    CellToHexahedronMerger(&swap_utils, i_cell_1, i_cell_2);
    return;
  }
  case Quadrilateral:
  case Triangle: {
    {
      CellToQuadrilateralMerger(&swap_utils, i_cell_1, i_cell_2);
      return;
    }
  }
  case NotMergeable: {
    ARCANE_FATAL("Impossible to merge the entities !\n");
  }
  }
  ARCANE_FATAL("Merge for this kind of cell not implemented\n");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Cell CellMerger::
getCell(Cell i_cell_1, Cell i_cell_2)
{
  _Type cell_1_type = _getCellType(i_cell_1.type());
  _Type cell_2_type = _getCellType(i_cell_2.type());

  _Type merged_cell_type = _promoteType(cell_1_type, cell_2_type);

  switch (merged_cell_type) {
  case Hexahedron: {
    return i_cell_1;
  }
  case Pyramid:
  case Pentahedron: {
    if (cell_2_type == Hexahedron) {
      return i_cell_1;
    }
    else {
      return i_cell_2;
    }
  }
  case Quadrilateral: {
    return i_cell_1;
  }
  case Triangle: {
    if (cell_2_type == Quadrilateral) {
      return i_cell_1;
    }
    else {
      return i_cell_2;
    }
  }
  case NotMergeable: {
    ARCANE_FATAL("Impossible to merge the entities !\n");
  }
  default:
    ARCANE_FATAL("Merge for this kind of cell not implemented\n");
  }
  return 0;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemInternal* CellMerger::
getItemInternal(ItemInternal* i_cell_1, ItemInternal* i_cell_2)
{
  return ItemCompatibility::_itemInternal(getCell(i_cell_1, i_cell_2));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
