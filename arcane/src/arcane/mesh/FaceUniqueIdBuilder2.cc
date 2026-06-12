// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* FaceUniqueIdBuilder2.cc                                     (C) 2000-2024 */
/*                                                                           */
/* Construction of unique face identifiers.                                  */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/ArgumentException.h"
#include "arcane/utils/CheckedConvert.h"

#include "arcane/mesh/DynamicMesh.h"
#include "arcane/mesh/FaceUniqueIdBuilder.h"

#include "arcane/core/IParallelMng.h"
#include "arcane/core/Timer.h"

#include "arcane/parallel/BitonicSortT.H"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Construction of the uniqueId() for faces.
 *
 * This class allows calculating the uniqueId() for faces.
 * After calling computeFacesUniqueId(), the uniqueId() and owner()
 * fields of each face are set.
 *
 * This algorithm guarantees that the numbering is the same
 * regardless of the decomposition and the number of processors.
 * In sequential mode, the algorithm can be written as follows:
 \code
 * Int64 face_unique_id_counter = 0;
 * // Iterate over cells assuming increasing uniqueIds.
 * ENUMERATE_CELL(icell,allCells()){
 *   Cell cell = *icell;
 *   ENUMERATE_FACE(iface,cell.faces()){
 *    Face face = *iface;
 *    // If I don't already have a uniqueId(), assign one and increment the counter
 *    if (face.uniqueId()==NULL_ITEM_UNIQUE_ID){
 *      face.setUniqueId(face_unique_id_counter);
 *      ++face_unique_id_counter;
 *    }
 *   }
 * }
 \endcode
 * The sequential algorithm assumes that cells are traversed in increasing order
 * of uniqueIds. For a given face, therefore, the cell with the smallest uniqueId()
 * will provide the face's uniqueId() and thus the face's owner.
 *
 * This version uses a parallel sort to ensure that
 * the number of messages increases by log2(N), where N is the number of processors.
 * This avoids potentially having a large number of messages, which
 * is not supported by certain MPI implementations (for example MPC).
 */
class FaceUniqueIdBuilder2
: public TraceAccessor
{
 public:

  class NarrowCellFaceInfo;
  class WideCellFaceInfo;
  class AnyFaceInfo;
  class BoundaryFaceInfo;
  class ResendCellInfo;
  class AnyFaceBitonicSortTraits;
  class BoundaryFaceBitonicSortTraits;
  class UniqueIdSorter;

  // Choose the correct typedef depending on the exchange structure choice.
  typedef NarrowCellFaceInfo CellFaceInfo;

 public:

  //! Constructs an instance for the mesh \a mesh
  explicit FaceUniqueIdBuilder2(DynamicMesh* mesh);

 public:

  void computeFacesUniqueIdAndOwnerVersion3();
  void computeFacesUniqueIdAndOwnerVersion5();

 private:

  DynamicMesh* m_mesh = nullptr;
  IParallelMng* m_parallel_mng = nullptr;
  bool m_is_verbose = false;

 private:

  void _resendCellsAndComputeFacesUniqueId(ConstArrayView<AnyFaceInfo> all_csi);
  void _checkFacesUniqueId();
  void _unsetFacesUniqueId();
  void _computeAndSortBoundaryFaces(Array<BoundaryFaceInfo>& boundary_faces_info);
  void _computeParallel();
  void _computeSequential();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Stores info about a face of a cell.

 * This structure is used during face sorting. Since the sort
 * is parallel and to limit the size of the messages sent,
 * the size of this structure must be as small as possible.
 * For this, we assume that max values are not reached.
 * Normally, we have:
 * - cell_uid -> Int64
 * - rank -> Int32
 * - local_face_index -> Int32
 * We assume the following practical limits:
 * - cell_uid         -> 39 bits, or 250 billion cells
 * - rank             -> 20 bits, or 1,048,576 PEs
 * - local_face_index -> 5 bits, or 32 faces per cell
 *
 * Logically, these limits will not be reached for a while (it is
 * 2012). And when that happens, it will be enough to use the structure
 * WideCellFaceInfo by changing the typedef accordingly.
 *
 * We therefore use a single Int64, with the first 39 bits for the uid,
 * the next 20 for the rank, and the last 5 for the local_index.
 * Note that to avoid sign problems, we store the given value
 * plus 1.
 */
class FaceUniqueIdBuilder2::NarrowCellFaceInfo
{
 public:

  static const Int64 BITS_CELL_UID = 39;
  static const Int64 BITS_RANK = 20;
  static const Int64 BITS_INDEX = 5;
  static const Int64 ONE_INT64 = 1;
  static const Int64 MASK_CELL_UID = (ONE_INT64 << BITS_CELL_UID) - 1;
  static const Int64 MASK_RANK = ((ONE_INT64 << BITS_RANK) - 1) << BITS_CELL_UID;
  static const Int64 MASK_INDEX = ((ONE_INT64 << BITS_INDEX) - 1) << (BITS_CELL_UID + BITS_RANK);

 public:

  NarrowCellFaceInfo()
  {
    setValue(NULL_ITEM_UNIQUE_ID, -1, -1);
  }

 public:

  bool isMaxValue() const
  {
    Int64 max_id = (MASK_CELL_UID - 1);
    return cellUid() == max_id;
  }

  void setMaxValue()
  {
    Int64 max_id = (MASK_CELL_UID - 1);
    setValue(max_id, -1, -1);
  }

  void setValue(Int64 cell_uid, Int32 _rank, Int32 face_local_index)
  {
    Int64 v_fli = face_local_index + 1;
    Int64 v_rank = _rank + 1;
    Int64 v_uid = cell_uid + 1;
    m_value = v_fli << (BITS_CELL_UID + BITS_RANK);
    m_value += v_rank << (BITS_CELL_UID);
    m_value += v_uid;
    if (cellUid() != cell_uid)
      ARCANE_FATAL("Bad uid expected='{0}' computed='{1}' v={2}", cell_uid, cellUid(), m_value);
    if (rank() != _rank)
      ARCANE_FATAL("Bad rank expected='{0}' computed='{1}'", _rank, rank());
    if (faceLocalIndex() != face_local_index)
      ARCANE_FATAL("Bad local_index expected='{0}' computed='{1}'", face_local_index, faceLocalIndex());
  }
  Int64 cellUid() const { return (m_value & MASK_CELL_UID) - 1; }
  Int32 rank() const { return CheckedConvert::toInt32(((m_value & MASK_RANK) >> BITS_CELL_UID) - 1); }
  Int32 faceLocalIndex() const { return CheckedConvert::toInt32(((m_value & MASK_INDEX) >> (BITS_CELL_UID + BITS_RANK)) - 1); }

  bool isValid() const { return cellUid() != NULL_ITEM_UNIQUE_ID; }

 private:

  Int64 m_value = -1;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Stores info about a face of a cell.
 */
class FaceUniqueIdBuilder2::WideCellFaceInfo
{
 public:

  WideCellFaceInfo()
  : m_cell_uid(NULL_ITEM_UNIQUE_ID)
  , m_rank(-1)
  , m_face_local_index(-1)
  {}

 public:

  bool isMaxValue() const
  {
    Int64 max_id = INT64_MAX;
    return cellUid() == max_id;
  }

  void setMaxValue()
  {
    Int64 max_id = INT64_MAX;
    setValue(max_id, -1, -1);
  }
  void setValue(Int64 cell_uid, Int32 rank, Int32 face_local_index)
  {
    m_cell_uid = cell_uid;
    m_rank = rank;
    m_face_local_index = face_local_index;
  }
  Int64 cellUid() const { return m_cell_uid; }
  Int32 rank() const { return m_rank; }
  Int32 faceLocalIndex() const { return m_face_local_index; }
  bool isValid() const { return m_cell_uid != NULL_ITEM_UNIQUE_ID; }

 private:

  Int64 m_cell_uid;
  Int32 m_rank;
  Int32 m_face_local_index;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Info for managing boundary faces of sub-domains.
 *
 * Regardless of the face type, a face can be uniquely determined
 * by its first three nodes.
 */
class FaceUniqueIdBuilder2::BoundaryFaceInfo
{
 public:

  BoundaryFaceInfo()
  : m_node0_uid(NULL_ITEM_UNIQUE_ID)
  , m_node1_uid(NULL_ITEM_UNIQUE_ID)
  , m_node2_uid(NULL_ITEM_UNIQUE_ID)
  , m_cell_uid(NULL_ITEM_UNIQUE_ID)
  , m_rank(-1)
  , m_face_local_index(-1)
  {}
  bool hasSameNodes(const BoundaryFaceInfo& fsi) const
  {
    return fsi.m_node0_uid == m_node0_uid && fsi.m_node1_uid == m_node1_uid && fsi.m_node2_uid == m_node2_uid;
  }
  void setNodes(Face face)
  {
    Integer nb_node = face.nbNode();
    if (nb_node >= 1)
      m_node0_uid = face.node(0).uniqueId();
    if (nb_node >= 2)
      m_node1_uid = face.node(1).uniqueId();
    if (nb_node >= 3)
      m_node2_uid = face.node(2).uniqueId();
  }

 public:

  Int64 m_node0_uid;
  Int64 m_node1_uid;
  Int64 m_node2_uid;
  Int64 m_cell_uid;
  Int32 m_rank;
  Int32 m_face_local_index;

 public:
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Info for managing faces of sub-domains.
 *
 * An instance of this class contains for a mesh face
 * the info on the two cells attached to it. For each cell, we
 * store the uniqueId(), the owner, and the local index of the face
 */
class FaceUniqueIdBuilder2::AnyFaceInfo
{
 public:

  AnyFaceInfo() = default;

 public:

  void setCell0(Int64 uid, Int32 rank, Int32 face_local_index)
  {
    m_cell0.setValue(uid, rank, face_local_index);
  }
  void setCell1(Int64 uid, Int32 rank, Int32 face_local_index)
  {
    m_cell1.setValue(uid, rank, face_local_index);
  }

 public:

  CellFaceInfo m_cell0;
  CellFaceInfo m_cell1;

 public:
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Attention, this class must have a size multiple of Int64
class FaceUniqueIdBuilder2::ResendCellInfo
{
 public:

  Int64 m_cell_uid;
  // This field contains both the local index of the face in the cell
  // and the rank of the cell owner.
  // m_face_local_index_and_owner_rank / nb_rank -> face_index
  // m_face_local_index_and_owner_rank % nb_rank -> owner_rank
  Int32 m_face_local_index_and_owner_rank;
  Int32 m_index_in_rank_list;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Functor for sorting BoundaryFaceInfo via bitonic sort.
 */
class FaceUniqueIdBuilder2::BoundaryFaceBitonicSortTraits
{
 public:

  static bool compareLess(const BoundaryFaceInfo& k1, const BoundaryFaceInfo& k2)
  {
    if (k1.m_node0_uid < k2.m_node0_uid)
      return true;
    if (k1.m_node0_uid > k2.m_node0_uid)
      return false;

    // k1.node0_uid == k2.node0_uid
    if (k1.m_node1_uid < k2.m_node1_uid)
      return true;
    if (k1.m_node1_uid > k2.m_node1_uid)
      return false;

    // k1.node1_uid == k2.node1_uid
    if (k1.m_node2_uid < k2.m_node2_uid)
      return true;
    if (k1.m_node2_uid > k2.m_node2_uid)
      return false;

    // k1.node2_uid == k2.node2_uid
    return (k1.m_cell_uid < k2.m_cell_uid);
  }

  static Parallel::Request send(IParallelMng* pm, Int32 rank, ConstArrayView<BoundaryFaceInfo> values)
  {
    const BoundaryFaceInfo* fsi_base = values.data();
    return pm->send(ByteConstArrayView(messageSize(values), (const Byte*)fsi_base), rank, false);
  }
  static Parallel::Request recv(IParallelMng* pm, Int32 rank, ArrayView<BoundaryFaceInfo> values)
  {
    BoundaryFaceInfo* fsi_base = values.data();
    return pm->recv(ByteArrayView(messageSize(values), (Byte*)fsi_base), rank, false);
  }
  static Integer messageSize(ConstArrayView<BoundaryFaceInfo> values)
  {
    return CheckedConvert::toInteger(values.size() * sizeof(BoundaryFaceInfo));
  }
  static BoundaryFaceInfo maxValue()
  {
    BoundaryFaceInfo fsi;
    fsi.m_cell_uid = INT64_MAX;
    fsi.m_rank = INT32_MAX;
    fsi.m_node0_uid = INT64_MAX;
    fsi.m_node1_uid = INT64_MAX;
    fsi.m_node2_uid = INT64_MAX;
    return fsi;
  }
  static bool isValid(const BoundaryFaceInfo& fsi)
  {
    return fsi.m_cell_uid != INT64_MAX;
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Functor for sorting AnyFaceInfo using bitonic sort.
 *
 * The goal is to sort the list so that the cells with the smallest
 * uniqueId() come first, and for the same cell, the smallest
 * \a face_local_index comes first.
 *
 * Only the information from the first cell of AnyFaceInfo is used
 * for sorting (the second cell only serves to send the information
 * to the concerned processors).
 */
class FaceUniqueIdBuilder2::AnyFaceBitonicSortTraits
{
 public:

  static bool compareLess(const AnyFaceInfo& k1, const AnyFaceInfo& k2)
  {
    Int64 k1_cell0_uid = k1.m_cell0.cellUid();
    Int64 k2_cell0_uid = k2.m_cell0.cellUid();
    if (k1_cell0_uid < k2_cell0_uid)
      return true;
    if (k1_cell0_uid > k2_cell0_uid)
      return false;

    Int64 k1_face0_local_index = k1.m_cell0.faceLocalIndex();
    Int64 k2_face0_local_index = k2.m_cell0.faceLocalIndex();
    if (k1_face0_local_index < k2_face0_local_index)
      return true;
    if (k1_face0_local_index > k2_face0_local_index)
      return false;

    return (k1.m_cell1.cellUid() < k2.m_cell1.cellUid());
  }

  static Parallel::Request send(IParallelMng* pm, Int32 rank, ConstArrayView<AnyFaceInfo> values)
  {
    const AnyFaceInfo* fsi_base = values.data();
    Integer message_size = CheckedConvert::toInteger(values.size() * sizeof(AnyFaceInfo));
    return pm->send(ByteConstArrayView(message_size, (const Byte*)fsi_base), rank, false);
  }

  static Parallel::Request recv(IParallelMng* pm, Int32 rank, ArrayView<AnyFaceInfo> values)
  {
    AnyFaceInfo* fsi_base = values.data();
    Integer message_size = CheckedConvert::toInteger(values.size() * sizeof(AnyFaceInfo));
    return pm->recv(ByteArrayView(message_size, (Byte*)fsi_base), rank, false);
  }

  static AnyFaceInfo maxValue()
  {
    AnyFaceInfo csi;
    csi.m_cell0.setMaxValue();
    csi.m_cell1.setMaxValue();
    return csi;
  }

  static bool isValid(const AnyFaceInfo& csi)
  {
    return !csi.m_cell0.isMaxValue();
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class FaceUniqueIdBuilder2::UniqueIdSorter
{
 public:

  bool operator()(const Item& i1, const Item& i2) const
  {
    return i1.uniqueId() < i2.uniqueId();
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

FaceUniqueIdBuilder2::
FaceUniqueIdBuilder2(DynamicMesh* mesh)
: TraceAccessor(mesh->traceMng())
, m_mesh(mesh)
, m_parallel_mng(mesh->parallelMng())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Calculates the unique IDs of each face in parallel.
 */
void FaceUniqueIdBuilder2::
computeFacesUniqueIdAndOwnerVersion3()
{
  if (m_parallel_mng->isParallel())
    _computeParallel();
  else
    _computeSequential();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Calculates the unique IDs of each face sequentially.
 */
void FaceUniqueIdBuilder2::
_computeSequential()
{
  info() << "Compute FacesUniqueId() Sequential V3";

  //TODO: allow not to start at zero.
  Int64 face_unique_id_counter = 0;

  ItemInternalMap& cells_map = m_mesh->cellsMap();
  Integer nb_cell = cells_map.count();
  UniqueArray<Cell> cells;
  cells.reserve(nb_cell);
  // First, the cells must be sorted by their uniqueId()
  // in ascending order
  cells_map.eachItem([&](Cell item) {
    cells.add(item);
  });
  std::sort(std::begin(cells), std::end(cells), UniqueIdSorter());

  // Invalidate uids to ensure they are all positioned.
  _unsetFacesUniqueId();

  for (Integer i = 0; i < nb_cell; ++i) {
    Cell cell = cells[i];
    for (Face face : cell.faces()) {
      if (face.uniqueId() == NULL_ITEM_UNIQUE_ID) {
        face.mutableItemBase().setUniqueId(face_unique_id_counter);
        ++face_unique_id_counter;
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Calculates the unique IDs of each face in parallel.
 */
void FaceUniqueIdBuilder2::
_computeParallel()
{
  IParallelMng* pm = m_parallel_mng;
  Int32 my_rank = pm->commRank();

  bool is_verbose = m_is_verbose;

  ItemInternalMap& cells_map = m_mesh->cellsMap();

  info() << "Compute FacesUniqueId() V3 using parallel sort";

  // Calculate and sort for boundary faces
  UniqueArray<BoundaryFaceInfo> boundary_faces_info;
  _computeAndSortBoundaryFaces(boundary_faces_info);

  // Here, the boundary faces are sorted based on their nodes.
  // Normally, in this list, 2 consecutive BoundaryFaceInfo elements that
  // have the same nodes represent the same face. In this case, we generate
  // an AnyFaceInfo with the information of the two cells from these two elements of
  // the list, being careful to put the cell with the smaller uniqueId() first.
  // If two consecutive elements of the list do not have the same nodes, it
  // means that the face is on the edge of the global domain.
  // We must still handle the case of two consecutive elements of the list
  // that are located on different processors. To manage this case, each
  // processor sends the last element of its list to the next one if it cannot
  // be merged with the second-to-last, in the hope that it can be merged with the
  // first one of the next processor's list.
  UniqueArray<AnyFaceInfo> all_face_list;
  {
    ConstArrayView<BoundaryFaceInfo> all_fsi = boundary_faces_info;
    Integer n = all_fsi.size();
    bool is_last_already_done = false;
    for (Integer i = 0; i < n; ++i) {
      const BoundaryFaceInfo& fsi = all_fsi[i];
      Int64 cell_uid0 = fsi.m_cell_uid;
      bool is_inside = false;
      //TODO: handle the case if the previous cell is on another proc
      // For this, it is necessary to retrieve the last value of the previous proc
      // and check if it corresponds to our first value
      is_inside = ((i + 1) != n && fsi.hasSameNodes(all_fsi[i + 1]));
      if (is_last_already_done) {
        is_last_already_done = false;
      }
      else {
        AnyFaceInfo csi;
        if (is_inside) {
          const BoundaryFaceInfo& next_fsi = all_fsi[i + 1];
          Int64 cell_uid1 = next_fsi.m_cell_uid;
          if (cell_uid0 < cell_uid1) {
            csi.setCell0(cell_uid0, fsi.m_rank, fsi.m_face_local_index);
            csi.setCell1(cell_uid1, next_fsi.m_rank, next_fsi.m_face_local_index);
          }
          else {
            csi.setCell0(cell_uid1, next_fsi.m_rank, next_fsi.m_face_local_index);
            csi.setCell1(cell_uid0, fsi.m_rank, fsi.m_face_local_index);
          }
          is_last_already_done = true;
        }
        else {
          csi.setCell0(cell_uid0, fsi.m_rank, fsi.m_face_local_index);
        }
        all_face_list.add(csi);
      }
      if (is_verbose)
        info() << "FACES_KEY i=" << i
               << " n0=" << fsi.m_node0_uid
               << " n1=" << fsi.m_node1_uid
               << " n2=" << fsi.m_node2_uid
               << " cell=" << fsi.m_cell_uid
               << " rank=" << fsi.m_rank
               << " li=" << fsi.m_face_local_index
               << " in=" << is_inside;
    }
  }

  // Add the faces belonging to our sub-domain.
  // These are all faces that have 2 connected cells.
  cells_map.eachItem([&](Cell cell) {
    Integer cell_nb_face = cell.nbFace();
    Int64 cell_uid = cell.uniqueId();
    for (Integer z = 0; z < cell_nb_face; ++z) {
      Face face = cell.face(z);
      if (face.nbCell() != 2)
        continue;
      Cell cell0 = face.cell(0);
      Cell cell1 = face.cell(1);
      Cell next_cell = (cell0 == cell) ? cell1 : cell0;
      Int64 next_cell_uid = next_cell.uniqueId();
      // Only record if I am the cell with the smaller uid
      if (cell_uid < next_cell_uid) {
        AnyFaceInfo csi;
        csi.m_cell0.setValue(cell_uid, my_rank, z);
        // The face_local_index of cell 1 will not be used
        csi.m_cell1.setValue(next_cell_uid, my_rank, -1);
        all_face_list.add(csi);
      }
    }
  });

  if (is_verbose) {
    Integer n = all_face_list.size();
    for (Integer i = 0; i < n; ++i) {
      const AnyFaceInfo& csi = all_face_list[i];
      info() << "CELL_TO_SORT i=" << i
             << " cell0=" << csi.m_cell0.cellUid()
             << " lidx0=" << csi.m_cell0.faceLocalIndex()
             << " cell1=" << csi.m_cell1.cellUid();
    }
  }

  info() << "ALL_FACE_LIST memorysize=" << sizeof(AnyFaceInfo) * all_face_list.size();
  Parallel::BitonicSort<AnyFaceInfo, AnyFaceBitonicSortTraits> all_face_sorter(pm);
  all_face_sorter.setNeedIndexAndRank(false);
  Real sort_begin_time = platform::getRealTime();
  all_face_sorter.sort(all_face_list);
  Real sort_end_time = platform::getRealTime();
  info() << "END_ALL_FACE_SORTER time=" << (Real)(sort_end_time - sort_begin_time);

  _resendCellsAndComputeFacesUniqueId(all_face_sorter.keys());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Determines the list of boundary faces for each subdomain
 * and sorts them across all procs.
 */
void FaceUniqueIdBuilder2::
_computeAndSortBoundaryFaces(Array<BoundaryFaceInfo>& boundary_faces_info)
{
  IParallelMng* pm = m_parallel_mng;
  Int32 my_rank = pm->commRank();
  bool is_verbose = m_is_verbose;
  ItemInternalMap& faces_map = m_mesh->facesMap();

  Parallel::BitonicSort<BoundaryFaceInfo, BoundaryFaceBitonicSortTraits> boundary_face_sorter(pm);
  boundary_face_sorter.setNeedIndexAndRank(false);

  //UniqueArray<BoundaryFaceInfo> boundary_face_list;
  boundary_faces_info.clear();
  faces_map.eachItem([&](Face face) {
    BoundaryFaceInfo fsi;
    Integer nb_cell = face.nbCell();
    if (nb_cell == 2)
      return;

    fsi.m_rank = my_rank;
    fsi.setNodes(face);
    Cell cell = face.cell(0);
    fsi.m_cell_uid = cell.uniqueId();
    Integer face_local_index = 0;
    for (Integer z = 0, zs = cell.nbFace(); z < zs; ++z)
      if (cell.face(z) == face) {
        face_local_index = z;
        break;
      }
    fsi.m_face_local_index = face_local_index;
    boundary_faces_info.add(fsi);
  });

  if (is_verbose) {
    ConstArrayView<BoundaryFaceInfo> all_fsi = boundary_faces_info;
    Integer n = all_fsi.size();
    for (Integer i = 0; i < n; ++i) {
      const BoundaryFaceInfo& fsi = all_fsi[i];
      info() << "KEY i=" << i
             << " n0=" << fsi.m_node0_uid
             << " n1=" << fsi.m_node1_uid
             << " n2=" << fsi.m_node2_uid
             << " cell=" << fsi.m_cell_uid
             << " rank=" << fsi.m_rank
             << " li=" << fsi.m_face_local_index;
    }
  }

  Real sort_begin_time = platform::getRealTime();
  boundary_face_sorter.sort(boundary_faces_info);
  Real sort_end_time = platform::getRealTime();
  info() << "END_BOUNDARY_FACE_SORT time=" << (Real)(sort_end_time - sort_begin_time);

  {
    ConstArrayView<BoundaryFaceInfo> all_bfi = boundary_face_sorter.keys();
    Integer n = all_bfi.size();
    if (is_verbose) {
      for (Integer i = 0; i < n; ++i) {
        const BoundaryFaceInfo& bfi = all_bfi[i];
        info() << " AFTER KEY i=" << i
               << " n0=" << bfi.m_node0_uid
               << " n1=" << bfi.m_node1_uid
               << " n2=" << bfi.m_node2_uid
               << " cell=" << bfi.m_cell_uid
               << " rank=" << bfi.m_rank
               << " li=" << bfi.m_face_local_index;
      }
    }

    // As a single node may be present in the previous proc's list, each PE
    // (except 0) sends to the previous process the start of its list which contains the same nodes.

    // TODO: merge this code with that of GhostLayerBuilder2
    UniqueArray<BoundaryFaceInfo> end_face_list;
    Integer begin_own_list_index = 0;
    if (n != 0 && my_rank != 0) {
      if (BoundaryFaceBitonicSortTraits::isValid(all_bfi[0])) {
        Int64 node0_uid = all_bfi[0].m_node0_uid;
        for (Integer i = 0; i < n; ++i) {
          if (all_bfi[i].m_node0_uid != node0_uid) {
            begin_own_list_index = i;
            break;
          }
          else
            end_face_list.add(all_bfi[i]);
        }
      }
    }
    info() << "BEGIN_OWN_LIST_INDEX=" << begin_own_list_index;
    if (is_verbose) {
      for (Integer k = 0, kn = end_face_list.size(); k < kn; ++k)
        info() << " SEND n0=" << end_face_list[k].m_node0_uid
               << " n1=" << end_face_list[k].m_node1_uid
               << " n2=" << end_face_list[k].m_node2_uid;
    }

    UniqueArray<BoundaryFaceInfo> end_face_list_recv;

    UniqueArray<Parallel::Request> requests;
    Integer recv_message_size = 0;
    Integer send_message_size = BoundaryFaceBitonicSortTraits::messageSize(end_face_list);

    Int32 nb_rank = pm->commSize();

    // First sends and receives the sizes.
    if (my_rank != (nb_rank - 1)) {
      requests.add(pm->recv(IntegerArrayView(1, &recv_message_size), my_rank + 1, false));
    }
    if (my_rank != 0) {
      requests.add(pm->send(IntegerConstArrayView(1, &send_message_size), my_rank - 1, false));
    }

    pm->waitAllRequests(requests);
    requests.clear();

    if (recv_message_size != 0) {
      Integer message_size = CheckedConvert::toInteger(recv_message_size / sizeof(BoundaryFaceInfo));
      end_face_list_recv.resize(message_size);
      requests.add(BoundaryFaceBitonicSortTraits::recv(pm, my_rank + 1, end_face_list_recv));
    }
    if (send_message_size != 0)
      requests.add(BoundaryFaceBitonicSortTraits::send(pm, my_rank - 1, end_face_list));

    pm->waitAllRequests(requests);

    boundary_faces_info.clear();
    boundary_faces_info.addRange(all_bfi.subConstView(begin_own_list_index, n - begin_own_list_index));
    boundary_faces_info.addRange(end_face_list_recv);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void FaceUniqueIdBuilder2::
_resendCellsAndComputeFacesUniqueId(ConstArrayView<AnyFaceInfo> all_csi)
{
  IParallelMng* pm = m_parallel_mng;
  Int32 nb_rank = pm->commSize();
  Int32 my_rank = pm->commRank();
  bool is_verbose = m_is_verbose;

  ItemInternalMap& cells_map = m_mesh->cellsMap();

  Int64 nb_computed_face = all_csi.size();

  if (is_verbose) {
    for (Integer i = 0; i < nb_computed_face; ++i) {
      const AnyFaceInfo& csi = all_csi[i];
      info() << "CELLS_KEY i=" << i
             << " cell0=" << csi.m_cell0.cellUid()
             << " lidx0=" << csi.m_cell0.faceLocalIndex()
             << " cell1=" << csi.m_cell1.cellUid()
             << " lidx1=" << csi.m_cell1.faceLocalIndex()
             << " rank0=" << csi.m_cell0.rank()
             << " rank1=" << csi.m_cell1.rank();
    }
  }

  // Calculation for each rank of the number of values to send
  // and stores it in nb_info_to_send;
  IntegerUniqueArray nb_info_to_send(nb_rank, 0);
  {
    for (Integer i = 0; i < nb_computed_face; ++i) {
      const AnyFaceInfo& csi = all_csi[i];
      Int32 rank0 = csi.m_cell0.rank();
      Int32 rank1 = csi.m_cell1.rank();

      ++nb_info_to_send[rank0];

      // It must only be sent if the rank is different from m_rank0
      if (csi.m_cell1.isValid() && rank1 != rank0)
        ++nb_info_to_send[rank1];
    }
  }

  // Array for each process indicating the uniqueId() of the first
  // face of this process.
  Int64UniqueArray all_first_face_uid(nb_rank);
  {
    // Each process retrieves the number of cells in the list.
    // Since this list will be sorted, this corresponds to the uid of the first
    // face of this process.
    Int64 nb_cell_to_sort = all_csi.size();
    pm->allGather(Int64ConstArrayView(1, &nb_cell_to_sort), all_first_face_uid);

    Int64 to_add = 0;
    for (Integer i = 0; i < nb_rank; ++i) {
      Int64 next = all_first_face_uid[i];
      all_first_face_uid[i] = to_add;
      to_add += next;
    }
  }

  Integer total_nb_to_send = 0;
  IntegerUniqueArray nb_info_to_send_indexes(nb_rank, 0);
  for (Integer i = 0; i < nb_rank; ++i) {
    nb_info_to_send_indexes[i] = total_nb_to_send;
    total_nb_to_send += nb_info_to_send[i];
  }
  info() << "TOTAL_NB_TO_SEND=" << total_nb_to_send;

  UniqueArray<ResendCellInfo> resend_infos(total_nb_to_send);
  {
    for (Integer i = 0; i < nb_computed_face; ++i) {
      const AnyFaceInfo& csi = all_csi[i];
      Int32 rank0 = csi.m_cell0.rank();
      Int32 rank1 = csi.m_cell1.rank();

      ResendCellInfo& rci0 = resend_infos[nb_info_to_send_indexes[rank0]];
      rci0.m_cell_uid = csi.m_cell0.cellUid();
      rci0.m_face_local_index_and_owner_rank = (csi.m_cell0.faceLocalIndex() * nb_rank) + rank0;
      rci0.m_index_in_rank_list = i;
      ++nb_info_to_send_indexes[rank0];

      if (csi.m_cell1.isValid() && rank1 != rank0) {
        ResendCellInfo& rci1 = resend_infos[nb_info_to_send_indexes[rank1]];
        rci1.m_cell_uid = csi.m_cell1.cellUid();
        // Even if I am cell 1, the owner of the face will be cell 0.
        rci1.m_face_local_index_and_owner_rank = (csi.m_cell1.faceLocalIndex() * nb_rank) + rank0;
        rci1.m_index_in_rank_list = i;
        ++nb_info_to_send_indexes[rank1];
      }
    }
  }

  // Perform a single reduce
  Int64 total_nb_computed_face = pm->reduce(Parallel::ReduceSum, nb_computed_face);
  info() << "TOTAL_NB_COMPUTED_FACE=" << total_nb_computed_face;

  // Indicates to each PE how many infos I will send to it
  if (is_verbose)
    for (Integer i = 0; i < nb_rank; ++i)
      info() << "NB_TO_SEND: I=" << i << " n=" << nb_info_to_send[i];

  IntegerUniqueArray nb_info_to_recv(nb_rank, 0);
  {
    Timer::SimplePrinter sp(traceMng(), "Sending size with AllToAll");
    pm->allToAll(nb_info_to_send, nb_info_to_recv, 1);
  }

  if (is_verbose)
    for (Integer i = 0; i < nb_rank; ++i)
      info() << "NB_TO_RECV: I=" << i << " n=" << nb_info_to_recv[i];

  Integer total_nb_to_recv = 0;
  for (Integer i = 0; i < nb_rank; ++i)
    total_nb_to_recv += nb_info_to_recv[i];

  // It is highly likely that this will not work if the array is too large,
  // it must proceed with arrays that do not exceed 2Go because of the
  // Int32 of MPI.
  // TODO: Perform the AllToAll multiple times if necessary.
  UniqueArray<ResendCellInfo> recv_infos;
  {
    Int32 vsize = sizeof(ResendCellInfo) / sizeof(Int64);
    Int32UniqueArray send_counts(nb_rank);
    Int32UniqueArray send_indexes(nb_rank);
    Int32UniqueArray recv_counts(nb_rank);
    Int32UniqueArray recv_indexes(nb_rank);
    Int32 total_send = 0;
    Int32 total_recv = 0;
    for (Integer i = 0; i < nb_rank; ++i) {
      send_counts[i] = (Int32)(nb_info_to_send[i] * vsize);
      recv_counts[i] = (Int32)(nb_info_to_recv[i] * vsize);
      send_indexes[i] = total_send;
      recv_indexes[i] = total_recv;
      total_send += send_counts[i];
      total_recv += recv_counts[i];
    }
    recv_infos.resize(total_nb_to_recv);

    Int64ConstArrayView send_buf(total_nb_to_send * vsize, (Int64*)resend_infos.data());
    Int64ArrayView recv_buf(total_nb_to_recv * vsize, (Int64*)recv_infos.data());

    info() << "BUF_SIZES: send=" << send_buf.size() << " recv=" << recv_buf.size();
    {
      Timer::SimplePrinter sp(traceMng(), "Send values with AllToAll");
      pm->allToAllVariable(send_buf, send_counts, send_indexes, recv_buf, recv_counts, recv_indexes);
    }
  }

  // Invalidates the uids to ensure they will all be positioned.
  _unsetFacesUniqueId();

  if (is_verbose) {
    Integer index = 0;
    for (Int32 rank = 0; rank < nb_rank; ++rank) {
      for (Integer z = 0, zs = nb_info_to_recv[rank]; z < zs; ++z) {
        const ResendCellInfo& rci = recv_infos[index];
        ++index;

        Int64 cell_uid = rci.m_cell_uid;
        Int32 full_local_index = rci.m_face_local_index_and_owner_rank;
        Int32 face_local_index = full_local_index / nb_rank;
        Int32 owner_rank = full_local_index % nb_rank;
        Int64 face_uid = all_first_face_uid[rank] + rci.m_index_in_rank_list;
        info() << "RECV index=" << index << " uid=" << cell_uid
               << " local_idx=" << full_local_index
               << " face_local_idx=" << face_local_index
               << " owner_rank=" << owner_rank
               << " rank_idx=" << rci.m_index_in_rank_list
               << " rank=" << rank
               << " first_face_uid=" << all_first_face_uid[rank]
               << " computed_uid=" << face_uid;
      }
    }
  }

  // Positions the uniqueId() and the owner() of the faces.
  {
    Integer index = 0;
    for (Int32 i = 0; i < nb_rank; ++i) {
      Int32 rank = i;
      for (Integer z = 0, zs = nb_info_to_recv[i]; z < zs; ++z) {
        const ResendCellInfo& rci = recv_infos[index];
        ++index;

        Int64 cell_uid = rci.m_cell_uid;
        Int32 full_local_index = rci.m_face_local_index_and_owner_rank;
        Int32 face_local_index = full_local_index / nb_rank;
        Int32 owner_rank = full_local_index % nb_rank;

        Cell cell = cells_map.tryFind(cell_uid);
        if (cell.null())
          ARCANE_FATAL("Can not find cell data for '{0}'", cell_uid);
        Face face = cell.face(face_local_index);
        Int64 face_uid = all_first_face_uid[rank] + rci.m_index_in_rank_list;
        face.mutableItemBase().setUniqueId(face_uid);
        face.mutableItemBase().setOwner(owner_rank, my_rank);
      }
    }
  }

  // Checks that all faces have a valid uid
  _checkFacesUniqueId();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Calculates uniqueId() via a hash generated by the uniqueId() of the nodes.
 */
void FaceUniqueIdBuilder2::
computeFacesUniqueIdAndOwnerVersion5()
{
  info() << "Compute FacesUniqueId() V5 (experimental)";

  IParallelMng* pm = m_parallel_mng;
  Int32 my_rank = pm->commRank();
  bool is_parallel = pm->isParallel();

  ItemInternalMap& faces_map = m_mesh->facesMap();
  UniqueArray<Int64> nodes_uid;
  faces_map.eachItem([&](Face face) {
    Int32 nb_node = face.nbNode();
    nodes_uid.resize(nb_node);
    {
      Int32 index = 0;
      for (Node node : face.nodes()) {
        nodes_uid[index] = node.uniqueId();
        ++index;
      }
    }
    Int64 new_face_uid = MeshUtils::generateHashUniqueId(nodes_uid);
    face.mutableItemBase().setUniqueId(new_face_uid);
    // In parallel, indicates that the owner of this face must be positioned
    // if it is a boundary face.
    Int32 new_rank = my_rank;
    if (is_parallel && face.nbCell() == 1)
      new_rank = A_NULL_RANK;
    face.mutableItemBase().setOwner(new_rank, my_rank);
  });
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Invalidates the uids to ensure they are all positioned.
 */
void FaceUniqueIdBuilder2::
_unsetFacesUniqueId()
{
  ItemInternalMap& faces_map = m_mesh->facesMap();
  faces_map.eachItem([&](Item face) {
    face.mutableItemBase().unsetUniqueId();
  });
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Checks that all faces have a valid uid.
 */
void FaceUniqueIdBuilder2::
_checkFacesUniqueId()
{
  ItemInternalMap& faces_map = m_mesh->facesMap();
  Integer nb_error = 0;
  faces_map.eachItem([&](Face face) {
    Int64 face_uid = face.uniqueId();
    if (face_uid == NULL_ITEM_UNIQUE_ID) {
      info() << "Bad face uid cell0=" << face.cell(0).uniqueId();
      ++nb_error;
    }
  });
  if (nb_error != 0)
    ARCANE_FATAL("Internal error in face uniqueId computation: nb_invalid={0}", nb_error);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" void
_computeFaceUniqueIdVersion3(DynamicMesh* mesh)
{
  FaceUniqueIdBuilder2 f(mesh);
  f.computeFacesUniqueIdAndOwnerVersion3();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" void
_computeFaceUniqueIdVersion5(DynamicMesh* mesh)
{
  FaceUniqueIdBuilder2 f(mesh);
  f.computeFacesUniqueIdAndOwnerVersion5();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
