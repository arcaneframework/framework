// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* FaceUniqueIdBuilder.h                                       (C) 2000-2026 */
/*                                                                           */
/* Construction of unique face IDs.                                          */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MESH_FACEUNIQUEIDBUILDER_H
#define ARCANE_MESH_FACEUNIQUEIDBUILDER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"

#include "arcane/mesh/DynamicMeshIncrementalBuilder.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class DynamicMesh;
class T_CellFaceInfo;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Construction of ghost layers.
 */
class FaceUniqueIdBuilder
: public TraceAccessor
{
 public:

  using BoundaryInfosMap = HashTableMapT<Int32, SharedArray<Int64>>;
  using BoundaryInfosMapEnumerator = HashTableMapEnumeratorT<Int32, SharedArray<Int64>>;

 public:

  //! Constructs an instance for the \a mesh
  explicit FaceUniqueIdBuilder(DynamicMeshIncrementalBuilder* mesh_builder);

 public:

  void computeFacesUniqueIds();

 private:

  DynamicMesh* m_mesh = nullptr;
  DynamicMeshIncrementalBuilder* m_mesh_builder = nullptr;
  //! Distribute the face numbering of version 1 (ARCANE_DISTRIBUTED_FACE_NUMBERING=1)
  bool m_is_v1_distributed = false;

 private:

  void _computeFacesUniqueIdsSequential();
  void _computeFacesUniqueIdsParallelV1();
  void _v1Distributed();
  void _v1CheckAndDump(Int64ConstArrayView faces_new_uid, Int64ConstArrayView faces_opposite_cell_uid);
  void _v1CellBlockSizes(Int64ConstArrayView faces_opposite_cell_uid,
                         IntegerArray& my_cells_nb_back_face,
                         Array<T_CellFaceInfo>& my_cells_faces_info);
  void _v1ByAllGather(Int64ArrayView faces_opposite_cell_uid,
                      IntegerArrayView faces_opposite_cell_index,
                      IntegerArrayView faces_opposite_cell_owner,
                      HashTableMapT<Int64, Int64>& cells_first_face_uid,
                      IntegerArray& my_cells_nb_back_face,
                      Array<T_CellFaceInfo>& my_cells_faces_info);
  void _v1AssignFaceUniqueIds(Int64ConstArrayView faces_opposite_cell_uid,
                              IntegerConstArrayView faces_opposite_cell_index,
                              IntegerConstArrayView faces_opposite_cell_owner,
                              HashTableMapT<Int64, Int64>& cells_first_face_uid,
                              IntegerConstArrayView my_cells_nb_back_face);
  void _computeFacesUniqueIdsParallelV2();
  void _exchangeData(IParallelExchanger* exchanger, BoundaryInfosMap& boundary_infos_to_send);
  void _checkNoDuplicate();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
