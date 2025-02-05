// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CellMerger.cc                                               (C) 2000-2025 */
/*                                                                           */
/* Fusionne deux mailles.                                                    */
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

//! Classe utilitaire pour échanger des entités entre deux entités
class ItemSwapperUtils
: public TraceAccessor
{
 public:

  explicit ItemSwapperUtils(IMesh* mesh)
  : TraceAccessor(mesh->traceMng()),
    m_face_reorienter(mesh),
    m_cell_tm(mesh->cellFamily()->_internalApi()->topologyModifier()),
    m_face_tm(mesh->faceFamily()->_internalApi()->topologyModifier()),
    m_node_tm(mesh->nodeFamily()->_internalApi()->topologyModifier())
  {
  }
 public:

  /*!
   * \brief Échange deux noeuds entre deux faces.
   *
   * Échange le noeud d'index \a face1_node_idx de la face \a face1 avec le
   * noeuds d'index \a face2_node_idx de la face \a face2.
   */
  void swapFaceNodes(Face face_1,Face face_2,
                     Integer face1_node_idx,Integer face2_node_idx)
  {
    NodeLocalId face1_node = face_1.node(face1_node_idx);
    NodeLocalId face2_node = face_2.node(face2_node_idx);

    m_face_tm->replaceNode(face_1,face1_node_idx,face2_node);
    m_face_tm->replaceNode(face_2,face2_node_idx,face1_node);

    m_node_tm->findAndReplaceFace(face1_node,face_1,face_2);
    m_node_tm->findAndReplaceFace(face2_node,face_2,face_1);
  }

  /*!
   * \brief Échange deux noeuds entre deux mailles.
   *
   * Échange le noeud d'index \a cell1_node_idx de la maille \a cell1 avec le
   * noeuds d'index \a cell2_node_idx de la maille \a cell2.
   */
  void swapCellNodes(Cell cell1,Cell cell2,
                     Integer cell1_node_idx,Integer cell2_node_idx)
  {
    NodeLocalId cell1_node = cell1.node(cell1_node_idx);
    NodeLocalId cell2_node = cell2.node(cell2_node_idx);

    m_cell_tm->replaceNode(cell1,cell1_node_idx,cell2_node);
    m_cell_tm->replaceNode(cell2,cell2_node_idx,cell1_node);

    m_node_tm->findAndReplaceCell(cell1_node,cell1,cell2);
    m_node_tm->findAndReplaceCell(cell2_node,cell2,cell1);
  }

  /*!
   * \brief Échange deux faces entre deux mailles.
   *
   * Échange la face d'index \a cell1_face_idx de la maille \a cell1 avec la
   * face d'index \a cell2_face_idx de la maille \a cell2.
   */
  void swapCellFaces(Cell cell1,Cell cell2,
                     Integer cell1_face_idx,Integer cell2_face_idx)
  {
    FaceLocalId cell1_face = cell1.face(cell1_face_idx);
    FaceLocalId cell2_face = cell2.face(cell2_face_idx);

    m_cell_tm->replaceFace(cell1,cell1_face_idx,cell2_face);
    m_cell_tm->replaceFace(cell2,cell2_face_idx,cell1_face);

    m_face_tm->findAndReplaceCell(cell1_face,cell1,cell2);
    m_face_tm->findAndReplaceCell(cell2_face,cell2,cell1);
  }

  void checkAndChangeFaceOrientation(Cell cell)
  {
    // Ceci pourrait sans doutes etre optimise
    for( Integer i=0, n=cell.nbFace(); i<n; ++i) {
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
 * \brief  Recherche la face commune à deux mailles.
 *
 * Cet fonction-classe a pour but de déterminer la face commune à deux
 * mailles. Si les deux mailles ne sont pas jointes par une face, on
 * génère une erreur.
 */
class CommonFaceFinder
{
 public:
  typedef std::set<Integer> NodesLIDSet;

 private:
  Integer m_cell_1_local_number; //!< Numéro de la face commune dans la maille 1
  Integer m_cell_2_local_number; //!< Numéro de la face commune dans la maille 2

  NodesLIDSet m_nodes_lid_set;	//! Ensemble des localId des sommets en communs

 public:
  /*! 
   * Access en lecture à la liste des localId des sommets en communs
   *
   * @return m_nodes_lid_set
   */
  const NodesLIDSet& nodesLID() const
  {
    return m_nodes_lid_set;
  }

  /*!
   * Access en lecture seule au numéro local de la face commune dans
   * la maille 1
   * 
   * @return m_cell_1_local_number
   */
  Integer cell1LocalNumber() const
  {
    return m_cell_1_local_number;
  }

  /*! 
   * Access en lecture seule au numéro local de la face commune dans
   * la maille 2
   * 
   * @return m_cell_1_local_number
   */
  Integer cell2LocalNumber() const
  {
    return m_cell_2_local_number;
  }

  /*! 
   * Constructeur. C'est à l'appel du constructeur que toutes les
   * structures de données sont générées
   * 
   * @param i_cell_1 la 1ère maille
   * @param i_cell_2 la 2ème maille
   */
  CommonFaceFinder(Cell i_cell_1,Cell i_cell_2);
  ~CommonFaceFinder(){}
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CommonFaceFinder::
CommonFaceFinder(Cell i_cell_1,Cell i_cell_2)
: m_cell_1_local_number(-1)
, m_cell_2_local_number(-1)
{
  typedef std::map<Integer, Integer> LIDCellMapping;
  LIDCellMapping faces1; // numero des faces de la maille 1
  LIDCellMapping faces2; // numero des faces de la maille 2

  { // création des correspondances localId numero dans la maille pour la maille 1
    Integer n = 0;
    for( ItemEnumerator i_face(i_cell_1.faces()); i_face(); ++i_face ) {
      faces1[i_face->localId()] = n;
      n++;
    }
  }
  { // création des correspondances localId numero dans la maille pour la maille 2
    Integer n = 0;
    for( ItemEnumerator i_face(i_cell_2.faces()); i_face(); ++i_face ) {
      faces2[i_face->localId()] = n;
      n++;
    }
  }

  // On parcours maintenant les deux tables créé précédemment, comme
  // elles sont triées par localId croissant, on en déduit
  // simplement la face commune.
  LIDCellMapping::const_iterator i_face1 = faces1.begin();
  LIDCellMapping::const_iterator i_face2 = faces2.begin();

  do {
    const Integer& lid1 = i_face1->first; // localId de la face dans la liste 1
    const Integer& lid2 = i_face2->first; // localId de la face dans la liste 2

    if (lid1 == lid2) { // on a trouvé la face commune
      m_cell_1_local_number = i_face1->second;
      m_cell_2_local_number = i_face2->second;

      // Enregistrement des localId des noeuds de la face commune
      Face common_face = i_cell_1.face(m_cell_1_local_number);

      for( NodeEnumerator i_node(common_face.nodes()); i_node(); ++i_node) {
        m_nodes_lid_set.insert(i_node->localId());
      }
      return;
    }
    else {
      if (lid1<lid2) {
        ++i_face1;
      } else {
        ++i_face2;
      }
    }
  } while (i_face1 != faces1.end() && i_face2 != faces2.end());

  ARCANE_FATAL("Face commune non trouvée !");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief  Fusionne deux faces en 2D (en fait deux arêtes).
 */
class Faces2DMerger
{
 private:
  Integer m_face_1_common_node_numbers; /**< Numéros dans la face 1 des sommets communs avec la face 2 */
  Integer m_face_2_common_node_numbers; /**< Numéros dans la face 2 des sommets communs avec la face 1 */
  Integer m_face_2_exchanged_node_numbers; /**< Numéros dans la face 2 des sommets qui définiront la face fusionnée */

  /** 
   * Initialisation des quantité m_face_1_common_node_numbers m_face_2_common_node_numbers et 
   * m_face_2_exchanged_node_numbers
   * 
   * @param i_face_1 la face 1
   * @param i_face_2 la face 2
   */
  void _setFacesNodeNumbers(Face i_face_1,Face i_face_2);

 public:
  /** 
   * Constructeur
   * 
   * @param i_face_1 la face conservé
   * @param i_face_2 la face abandonnée
   */
  Faces2DMerger(ItemSwapperUtils* swap_utils,Face i_face_1,Face i_face_2);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void  Faces2DMerger::
_setFacesNodeNumbers(Face i_face_1,Face i_face_2)
{
  typedef std::map<Integer, Integer> LocalIDToLocalNumber;
  LocalIDToLocalNumber face1_node_localId;
  LocalIDToLocalNumber face2_node_localId;

  {
    Integer n = 0;
    for (ItemEnumerator i_node(i_face_1.nodes()); i_node(); ++i_node) {
      face1_node_localId[i_node->localId()]=n++;
    }
  }
  {
    Integer n = 0;
    for (ItemEnumerator i_node(i_face_2.nodes()); i_node(); ++i_node) {
      face2_node_localId[i_node->localId()]=n++;
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
    } else {
      if (node1_localId < node2_localId) {
        ++i;
      } else {
        ++j;
      }
    }
  }

  m_face_2_exchanged_node_numbers = (m_face_2_common_node_numbers+1)%2;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Faces2DMerger::
Faces2DMerger(ItemSwapperUtils* swap_utils,Face face1,Face face2)
: m_face_1_common_node_numbers(std::numeric_limits<Integer>::max())
, m_face_2_common_node_numbers(std::numeric_limits<Integer>::max())
, m_face_2_exchanged_node_numbers(std::numeric_limits<Integer>::max())
{
  ARCANE_ASSERT(face2.type()==IT_Line2,("The cell is not a line"));

  _setFacesNodeNumbers(face1,face2);

  swap_utils->swapFaceNodes(face1,face2,m_face_1_common_node_numbers,
                            m_face_2_exchanged_node_numbers);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief En dimension 2, recherche des faces communes à deux mailles
 * (Les faces sont en fait des arêtes).
 */
class Faces2DToMergeFinder
{
 private:

  //@{
  /**
   * Ces listes contenant les numéros des faces communes des deux
   * mailles, sont construites de sorte à ce que la face
   * m_cell1_edge_face_list[i] soit a fusionner avec la face
   * m_cell2_edge_face_list[i]
   */
  IntegerUniqueArray m_cell1_edge_face_list;
  IntegerUniqueArray m_cell2_edge_face_list;
  //@}

  /** 
   * On détermine les faces à fusionner en considérant les arêtes qui
   * sont portées par les sommets communs entre les mailles. Pour cela
   * on trie les faces selon les arêtes portées par des sommets communs.
   * 
   * @param i_cell la maille étudiée
   * @param edge_face_list la liste des faces numéros des faces
   * @param common_face_number le numéro de la face commune dans la maille @a i_cell
   * @param common_face_nodes les localIds des noeuds de la face commune
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
    // Pour chaque face de la maille
    for ( FaceEnumerator i_face(i_cell.faces()); i_face(); ++i_face, ++face_number) {
      if (face_number == common_face_number) {
        continue; // si la face est la face commune, on ne fait rien
      }

      // On crée la liste des noeuds de cette face qui sont communs
      std::multiset<Integer> node_list;
      for( NodeEnumerator i_node(i_face->nodes()); i_node(); ++i_node) {
        const Integer& node_lid = i_node->localId();
        if (common_face_nodes.find(node_lid) != common_face_nodes.end()) {
          node_list.insert(node_lid);
        }
      }

      switch (node_list.size()) {
      case 0:  // la face n'est pas à retailler [déjà traitée]
      case 2:continue;
      case 1: { // Si la liste ne contient qu'un élément, c'est que la face est à fusionner
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

    // On recopie les donnée : on n'a plus besoin des arêtes
    edge_face_list.reserve((Integer)temp_edge_face_list.size());
    for (_EdgeFaceList::const_iterator i = temp_edge_face_list.begin();
         i != temp_edge_face_list.end(); ++i) {
      edge_face_list.add(i->second); // on stocke les numéros des faces à fusionner
    }
  }

 public:
  /**
   * Access en lecture seule au nombre d'arêtes communes
   *
   * @return m_cell1_edge_face_list.size()
   */
  Integer getNumber() const
  {
    return m_cell1_edge_face_list.size();
  }

  /**
   * Accède au numéro dans la maille 1 de la @a i ème face à fusionner.
   *
   * @param i numéro dans la liste des mailles à fusionner 
   *
   * @return le numéro de la @a i ème face à fusionner
   */
  Integer cell1FaceNumber(Integer i) const
  {
    return m_cell1_edge_face_list[i];
  }

  /** 
   * Accède au numéro dans la maille 2 de la @a i ème face à fusionner.
   * 
   * @param i numéro dans la liste des mailles à fusionner 
   * 
   * @return le numéro de la @a i ème face à fusionner
   */
  Integer cell2FaceNumber(Integer i) const
  {
    return m_cell2_edge_face_list[i];
  }

  /** 
   * Construit les différentes structures de données.
   * 
   * @param i_cell_1 la première maille
   * @param i_cell_2 la seconde maille
   * @param common_face les informations sur la face commune
   */
  Faces2DToMergeFinder(Cell cell1,Cell cell2,
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
 * \brief Cette fonction-classe recherche les faces à fusionner lors
 * de la fusion de deux mailles
 */
class FacesToMergeFinder
{
 public:

  /** 
   * Accès au nombre de faces communes entre les deux mailles
   * 
   * @return m_cell1_edge_face_list.size()
   */
  Integer getNumber() const
  {
    return m_cell1_edge_face_list.size();
  }

  /** 
   * Accès au @a i ème numéro des faces dans la maille 1
   * 
   * @param i le numéro dans la liste des faces communes
   * 
   * @return le numéro dans la maille 1 de @a i ème face commune
   */
  Integer cell1FaceNumber(Integer i) const
  {
    return m_cell1_edge_face_list[i];
  }

  /** 
   * Accès au @a i ème numéro des faces dans la maille 2
   * 
   * @param i le numéro dans la liste des faces communes
   * 
   * @return le numéro dans la maille 2 de @a i ème face commune
   */
  Integer cell2FaceNumber(Integer i) const
  {
    return m_cell2_edge_face_list[i];
  }

  /** 
   * Construit les différentes structures de données
   * 
   * @param cell1 la première maille
   * @param cell2 la seconde maille
   * @param common_face les informations de la face commune
   */
  FacesToMergeFinder(Cell cell1,Cell cell2,
                     const CommonFaceFinder& common_face)
  {
    _setEdgeFaceList(cell1,m_cell1_edge_face_list,
                     common_face.cell1LocalNumber(),common_face.nodesLID());
    _setEdgeFaceList(cell2,m_cell2_edge_face_list,
                     common_face.cell2LocalNumber(),common_face.nodesLID());

    ARCANE_ASSERT(m_cell1_edge_face_list.size() == m_cell2_edge_face_list.size(),
		  ("Incompatible number of faces to merge !"));
  }

 private:

  //@{
  /**
   * Ces listes contenant les numéros des faces communes des deux
   * mailles, sont construites de sorte à ce que la face
   * m_cell1_edge_face_list[i] soit a fusionner avec la face
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
 * On détermine les faces à fusionner en considérant les arêtes qui
 * sont portées par les sommets communs entre les mailles. Pour cela
 * on trie les faces selon les arêtes portées par des sommets communs.
 * 
 * @param i_cell la maille étudiée
 * @param edge_face_list la liste des faces numéros des faces
 * @param common_face_number le numéro de la face commune dans la maille @a i_cell
 * @param common_face_nodes les localIds des noeuds de la face commune
 */
void FacesToMergeFinder::
_setEdgeFaceList(Cell i_cell,
                 IntegerArray& edge_face_list,
                 Integer common_face_number,
                 const CommonFaceFinder::NodesLIDSet& common_face_nodes)
{
  typedef std::pair<Integer,Integer> _EdgeDescriptor;
  typedef std::map<_EdgeDescriptor, Integer> _EdgeFaceList;

  _EdgeFaceList temp_edge_face_list; // liste tries des numéros de faces par arêtes
  Integer face_number = 0;
  for (FaceEnumerator i_face(i_cell.faces()); i_face(); ++i_face, ++face_number) {
    if (face_number == common_face_number) { // si la face est la face commune elle n'est pas traitée
      continue;
    }

    std::multiset<Integer> node_list; // liste des localIds des noeuds de la face qui sont communs
    for (NodeEnumerator i_node(i_face->nodes()); i_node(); ++i_node) {
      Int32 node_lid = i_node->localId();
      // si le noeud est commun on l'ajoute à la liste des noeuds communs
      if (common_face_nodes.find(node_lid) != common_face_nodes.end()) {
        node_list.insert(node_lid);
      }
    }

    switch (node_list.size()) {
    case 0:  // la face n'est pas à retailler [déjà traitée]
    case 4:continue;
    case 2: {
      std::multiset<Integer>::const_iterator i = node_list.begin();
      const Integer first_node_lid = *i;
      ++i;
      const Integer second_node_lid = *i;
      // On crée la correspondance arête/face (les noeuds de l'arête étant triés)
      temp_edge_face_list[std::make_pair(first_node_lid, second_node_lid)] = face_number;
      break;
    }
    default:
      ARCANE_FATAL("Unexpected number of nodes on the common face !");
    }
  }

  // On recopie les donnée : on n'a plus besoin des arêtes. Les
  // faces numéros des faces sont orientés comme on le souhaite.
  edge_face_list.reserve(CheckedConvert::toInteger(temp_edge_face_list.size()));
  for (_EdgeFaceList::const_iterator i = temp_edge_face_list.begin();
       i != temp_edge_face_list.end(); ++i) {
    edge_face_list.add(i->second);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Cett fonction-classe a pour but de fusionner deux faces dont
 * la deuxième est forcément un quadrangle.
 */
class FaceToQuadrilateralMerger
{
 private:
  IntegerUniqueArray m_face_1_common_node_numbers; /**< Numéros dans la face 1 des sommets communs avec la face 2 */
  IntegerUniqueArray m_face_2_common_node_numbers; /**< Numéros dans la face 1 des sommets communs avec la face 2 */
  IntegerUniqueArray m_face_2_exchanged_node_numbers; /**< Numéros dans la face 2 des sommets qui définiront la maille fusionnée */

  static const Integer m_quad_node_neighbors[4][2]; /**< Liste des noeuds voisins par arête dans un quadrangle */

  /** 
   * Initialisation des quantité m_face_1_common_node_numbers m_face_2_common_node_numbers et 
   * m_face_2_exchanged_node_numbers
   * 
   * @param i_face_1 la face 1
   * @param i_face_2 la face 2
   */
  bool _setFacesNodeNumbers(Face i_face_1,Face i_face_2);

 public:
  /** 
   * Constructeur
   * 
   * @param face1 la face consevée
   * @param face2 la face abandonnée [OBLIGATOIREMENT UN QUADRANGLE]
   */
  FaceToQuadrilateralMerger(ItemSwapperUtils* swap_utils,Face face1,Face face2);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool FaceToQuadrilateralMerger::
_setFacesNodeNumbers(Face i_face_1,Face i_face_2)
{
  typedef std::map<Integer, Integer> LocalIDToLocalNumber;
  LocalIDToLocalNumber face1_node_localId;
  LocalIDToLocalNumber face2_node_localId;

  {
    Integer n = 0;
    for (NodeEnumerator i_node(i_face_1.nodes()); i_node(); ++i_node) {
      face1_node_localId[i_node->localId()]=n++;
    }
  }
  {
    Integer n = 0;
    for (NodeEnumerator i_node(i_face_2.nodes()); i_node(); ++i_node) {
      face2_node_localId[i_node->localId()]=n++;
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
      ++i;++j;
    } else {
      if (node1_localId < node2_localId) {
        ++i;
      } else {
        ++j;
      }
    }
  }

  if (m_face_1_common_node_numbers.size() == 0) return false;

  ARCANE_ASSERT((m_face_2_common_node_numbers.size() == 2)
		&& (m_face_1_common_node_numbers.size() == 2),
		("Incorrect number of shared vertices"));

  m_face_2_exchanged_node_numbers.reserve(2);
  for (Integer i = 0; i<m_face_2_common_node_numbers.size(); ++i) {
    const Integer& node_number = m_face_2_common_node_numbers[i];
    for (Integer j=0; j<2; ++j) {
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
FaceToQuadrilateralMerger(ItemSwapperUtils* swap_utils,Face face1,Face face2)
{
  ARCANE_ASSERT(face2.type()==IT_Quad4,("The cell is not a quadrangle"));

  if (_setFacesNodeNumbers(face1,face2)) {
    ARCANE_ASSERT(m_face_2_exchanged_node_numbers.size() == 2,
                  ("Incorrect number of exchange vertices"));

    // Echange des sommets des faces
    for (Integer i = 0; i<2; ++i) {
      swap_utils->swapFaceNodes(face1,face2,m_face_1_common_node_numbers[i],
                                m_face_2_exchanged_node_numbers[i]);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const Integer
FaceToQuadrilateralMerger::
m_quad_node_neighbors[4][2] = { {1,3},{0,2},{1,3},{0,2} };

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class CellToQuadrilateralMerger
{
 private:
  //! Numéros dans la maille 1 des sommets communs avec la maille 2
  IntegerUniqueArray m_cell_1_common_node_numbers;
  //! Numéros dans la maille 2 des sommets communs avec la maille 1
  IntegerUniqueArray m_cell_2_common_node_numbers;
  //! Numéros dans la maille 2 des sommets qui définiront la maille fusionnée
  IntegerUniqueArray m_cell_2_exchanged_node_numbers;

  //! Liste des noeuds voisins par arête dans un quadrangle
  static const Integer m_quad_node_neighbors[4][2];
  
  /** 
   * Initialisation des quantité m_cell_1_common_node_numbers m_cell_2_common_node_numbers et 
   * m_cell_2_exchanged_node_numbers
   * 
   * @param i_cell_1 la maille 1
   * @param i_cell_2 la maille 2
   */
  void _setCellsNodeNumbers(Cell i_cell_1,Cell i_cell_2);

 public:

  /** 
   * Constructeur
   * 
   * @param cell1 la maille conservé
   * @param cell2 la maille abandonnée [OBLIGATOIREMENT UN QUADRANGLE]
   */
  CellToQuadrilateralMerger(ItemSwapperUtils* swap_utils,Cell cell1,Cell cell2);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CellToQuadrilateralMerger::
_setCellsNodeNumbers(Cell i_cell_1,Cell i_cell_2)
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
      ++i;++j;
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
  for (Integer i = 0; i<m_cell_2_common_node_numbers.size(); ++i) {
    const Integer& node_number = m_cell_2_common_node_numbers[i];
    for (Integer j=0; j<2; ++j) {
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
CellToQuadrilateralMerger(ItemSwapperUtils* swap_utils,Cell cell1,Cell cell2)
{
  ARCANE_ASSERT(cell2.type()==IT_Quad4,("Cell2 is not a IT_Quad4"));

  CommonFaceFinder common_face(cell1,cell2);

  this->_setCellsNodeNumbers(cell1,cell2);

  // Fusion des mailles de côté
  Faces2DToMergeFinder faces_to_merge(cell1,cell2,common_face);
  for (Integer i = 0; i<faces_to_merge.getNumber(); ++i) {
    Faces2DMerger(swap_utils,
                  cell1.face(faces_to_merge.cell1FaceNumber(i)),
                  cell2.face(faces_to_merge.cell2FaceNumber(i)));
  }

  // Echange des faces.
  swap_utils->swapCellFaces(cell1,cell2,
                            common_face.cell1LocalNumber(),
                            (common_face.cell2LocalNumber()+2)%4); // face opposée

  // Echange des sommets des mailles
  for (Integer i=0, n=m_cell_1_common_node_numbers.size(); i<n; ++i) {
    swap_utils->swapCellNodes(cell1,cell2,
                              m_cell_1_common_node_numbers[i],
                              m_cell_2_exchanged_node_numbers[i]);
  }

  swap_utils->checkAndChangeFaceOrientation(cell1);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const Integer CellToQuadrilateralMerger::m_quad_node_neighbors[4][2]
=  { {1,3},{0,2},{1,3},{0,2} };

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Cette fonction-classe a pour but de fusionner deux mailles
 * dont la deuxième est forcément un hexahèdre
 */
class CellToHexahedronMerger
{
 private:
  IntegerUniqueArray m_cell_1_common_node_numbers; /**< Numéros dans la maille 1 des sommets communs avec la maille 2 */
  IntegerUniqueArray m_cell_2_common_node_numbers; /**< Numéros dans la maille 2 des sommets communs avec la maille 1 */
  IntegerUniqueArray m_cell_2_exchanged_node_numbers; /**< Numéros dans la maille 2 des sommets qui définiront la maille fusionnée */

  static const Integer m_hexa_node_neighbors[8][3]; /**< Liste des noeuds voisins par arête dans un hexahèdre */

  /** 
   * Initialisation des quantité m_cell_1_common_node_numbers m_cell_2_common_node_numbers et 
   * m_cell_2_exchanged_node_numbers
   * 
   * @param cell1 la maille 1
   * @param cell2 la maille 2
   */
  void _setCellsNodeNumbers(Cell cell1,Cell cell2);

 public:
  /** 
   * Constructeur
   * 
   * @param cell1 la maille conservé
   * @param cell2 la maille abandonnée [OBLIGATOIREMENT UN HEXAEDRE]
   */
  CellToHexahedronMerger(ItemSwapperUtils* swap_utils,Cell cell1,Cell cell2);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CellToHexahedronMerger::
_setCellsNodeNumbers(Cell cell1,Cell cell2)
{
  typedef std::map<Integer, Integer> LocalIDToLocalNumber;
  LocalIDToLocalNumber cell1_node_localId;
  LocalIDToLocalNumber cell2_node_localId;

  // On associe les numéros (dans la maille) des noeuds à leur
  // localId. Ces listes sont triées par localId !
  {
    // d'abord pour la maille 1
    Integer n = 0;
    for (NodeEnumerator i_node(cell1.nodes()); i_node(); ++i_node) {
      cell1_node_localId[i_node->localId()] = n++;
    }
  }
  {
    // puis pour la maille 2
    Integer n = 0;
    for (NodeEnumerator i_node(cell2.nodes()); i_node(); ++i_node) {
      cell2_node_localId[i_node->localId()] = n++;
    }
  }

  // On determine ensuite l'ensemble des noeuds communs aux deux
  // mailles
  std::set<Integer> cell2_common_edge_node_number;
  for (LocalIDToLocalNumber::const_iterator
	 i = cell1_node_localId.begin(),
	 j = cell2_node_localId.begin();
       i != cell1_node_localId.end() && j != cell2_node_localId.end();) {
    Integer node1_localId = i->first;
    Integer node2_localId = j->first;
    if (node1_localId == node2_localId) { // si les noeuds sont les mêmes
      // on stockes les numéros dans la mailles de ces sommets
      m_cell_1_common_node_numbers.add(i->second); // pour la maille 1
      m_cell_2_common_node_numbers.add(j->second); // pour la maille 2

      // et on crée l'ensemble ordonné des noeuds communs dans la
      // seconde maille
      cell2_common_edge_node_number.insert(j->second);
      ++i;++j;
    } else {
      if (node1_localId < node2_localId) {
        ++i;
      } else {
        ++j;
      }
    }
  }

  ARCANE_ASSERT(m_cell_1_common_node_numbers.size() == 4,
		("Bad number of shared vertices"));

  // On cherche maintenant les sommets voisins des noeuds commun
  // appartenant à la seconde maille et qui ne sont pas des sommets
  // échangés. Ce sont les sommets qui formeront la nouvelle maille
  // par substitution avec les sommets communs de la première maille.
  m_cell_2_exchanged_node_numbers.reserve(4);
  for (Integer i = 0; i<m_cell_2_common_node_numbers.size(); ++i) {
    const Integer& node_number = m_cell_2_common_node_numbers[i];
    for (Integer j=0; j<3; ++j) {
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
CellToHexahedronMerger(ItemSwapperUtils* swap_utils,Cell cell1,Cell cell2)
{
  // TODO: fusionner ce code avec CellToQuadrilateralMerger.

  ARCANE_ASSERT(cell2.type() == IT_Hexaedron8,("Cell2 is not a IT_Hexaedron8"));

  CommonFaceFinder common_face(cell1,cell2);

  this->_setCellsNodeNumbers(cell1,cell2);

  // Fusion des mailles de côté
  FacesToMergeFinder faces_to_merge(cell1,cell2,common_face);
  for (Integer i = 0; i<faces_to_merge.getNumber(); ++i) {
    FaceToQuadrilateralMerger(swap_utils,
                              cell1.face(faces_to_merge.cell1FaceNumber(i)),
                              cell2.face(faces_to_merge.cell2FaceNumber(i)));
  }

  // Echange des faces.
  swap_utils->swapCellFaces(cell1,cell2,
                            common_face.cell1LocalNumber(),
                            (common_face.cell2LocalNumber()+3)%6); // face opposée

  // Echange des sommets des mailles
  for (Integer i=0, n=m_cell_1_common_node_numbers.size(); i<n; ++i) {
    swap_utils->swapCellNodes(cell1,cell2,
                              m_cell_1_common_node_numbers[i],
                              m_cell_2_exchanged_node_numbers[i]);
  }

  swap_utils->checkAndChangeFaceOrientation(cell1);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const Integer
CellToHexahedronMerger::
m_hexa_node_neighbors[8][3] = { {1,3,4},{0,2,5},{1,3,6},{0,2,7},{0,5,7},{1,4,6},{2,5,7},{3,4,6} };

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String CellMerger::
_typeName(const CellMerger::_Type& t) const
{
  switch (t) {
  case Hexahedron:    return "hexahèdre";
  case Pyramid:       return "pyramide";
  case Pentahedron:   return "pentahèdre";
  case Quadrilateral: return "quadrangle";
  case Triangle:      return "triangle";
  default:            return "inconnu";
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CellMerger::_Type CellMerger::
_getCellType(const Integer& internal_cell_type) const
{
  switch(internal_cell_type) {
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
_promoteType(const _Type& t1,const _Type& t2) const
{
  switch(t1*t2) {
  case 1:   return Hexahedron;
  case 2:   return Pyramid;
  case 3:   return Pentahedron;
  case 100: return Quadrilateral;
  case 110: return Triangle;
  default:
    ARCANE_FATAL("Can not merge cells of type {0} and {1}",_typeName(t1),_typeName(t2));
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CellMerger::
merge(Cell i_cell_1,Cell i_cell_2)
{
  _Type cell_1_type = _getCellType(i_cell_1.type());
  IMesh* mesh = i_cell_1.itemFamily()->mesh();
  ItemSwapperUtils swap_utils(mesh);

  switch (cell_1_type) {
  case Hexahedron:
  case Pyramid:
  case Pentahedron:
    {
      CellToHexahedronMerger(&swap_utils,i_cell_1, i_cell_2);
      return;
    }
  case Quadrilateral:
  case Triangle:{
    {
      CellToQuadrilateralMerger(&swap_utils,i_cell_1,i_cell_2);
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
getCell(Cell i_cell_1,Cell i_cell_2)
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
    } else {
      return i_cell_2;
    }
  }
  case Quadrilateral: {
    return i_cell_1;
  }
  case Triangle: {
    if (cell_2_type == Quadrilateral) {
      return i_cell_1;
    } else {
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
  return ItemCompatibility::_itemInternal(getCell(i_cell_1,i_cell_2));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
