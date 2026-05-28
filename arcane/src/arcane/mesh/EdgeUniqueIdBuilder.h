// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* EdgeUniqueIdBuilder.h                                       (C) 2000-2024 */
/*                                                                           */
/* Construction of unique edge IDs.                                          */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MESH_EDGEUNIQUEIDBUILDER_H
#define ARCANE_MESH_EDGEUNIQUEIDBUILDER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"

#include "arcane/mesh/DynamicMeshIncrementalBuilder.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class SerializeBuffer;
class IParallelExchanger;
class IParallelMng;
} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{
class DynamicMesh;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Construction of ghost layers.
 */
class EdgeUniqueIdBuilder
: public TraceAccessor
{
 public:

  typedef HashTableMapT<Int32, SharedArray<Int64>> BoundaryInfosMap;
  typedef HashTableMapEnumeratorT<Int32, SharedArray<Int64>> BoundaryInfosMapEnumerator;

 public:

  //! Constructs an instance for the mesh \a mesh
  explicit EdgeUniqueIdBuilder(DynamicMeshIncrementalBuilder* mesh_builder);

 public:

  void computeEdgesUniqueIds();

 private:

  DynamicMesh* m_mesh;
  DynamicMeshIncrementalBuilder* m_mesh_builder;

 private:

  void _computeEdgesUniqueIdsSequential();
  void _computeEdgesUniqueIdsParallel3();
  void _computeEdgesUniqueIdsParallelV2();
  void _computeEdgesUniqueIdsParallel64bit();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
