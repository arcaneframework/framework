// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* FaceUniqueIdBuilder.h                                       (C) 2000-2024 */
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

 private:

  void _computeFacesUniqueIdsSequential();
  void _computeFacesUniqueIdsParallelV1();
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
