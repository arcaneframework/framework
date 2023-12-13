// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CartesianMeshAllocateBuildInfo.cc                           (C) 2000-2023 */
/*                                                                           */
/* Informations pour allouer les entités d'un maillage cartésien.            */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/CartesianMeshAllocateBuildInfo.h"
#include "arcane/core/internal/CartesianMeshAllocateBuildInfoInternal.h"

#include "arcane/utils/CheckedConvert.h"
#include "arcane/utils/Vector3.h"
#include "arcane/utils/Vector2.h"
#include "arcane/utils/Array.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/FixedArray.h"

#include "arcane/core/IPrimaryMesh.h"
#include "arcane/core/ItemTypeId.h"
#include "arcane/core/IMeshInitialAllocator.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class CartesianMeshAllocateBuildInfo::Impl
{
 public:

  explicit Impl(IPrimaryMesh* mesh)
  : m_mesh(mesh)
  {
    m_internal.m_p = this;
  }

 public:

  IPrimaryMesh* m_mesh = nullptr;
  Int32 m_mesh_dimension = -1;
  Int32 m_face_builder_version = -1;
  Int32 m_edge_builder_version = -1;
  CartesianMeshAllocateBuildInfoInternal m_internal;

  Int64x3 m_global_nb_cells = {};
  Int32x3 m_own_nb_cells = {};
  Int64x3 m_first_own_cell_offset;
  Int64 m_cell_unique_id_offset = -1;
  Int64 m_node_unique_id_offset = -1;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 CartesianMeshAllocateBuildInfoInternal::
meshDimension() const
{
  return m_p->m_mesh_dimension;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshAllocateBuildInfoInternal::
setFaceBuilderVersion(Int32 version)
{
  m_p->m_face_builder_version = version;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 CartesianMeshAllocateBuildInfoInternal::
faceBuilderVersion() const
{
  return m_p->m_face_builder_version;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshAllocateBuildInfoInternal::
setEdgeBuilderVersion(Int32 version)
{
  m_p->m_edge_builder_version = version;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 CartesianMeshAllocateBuildInfoInternal::
edgeBuilderVersion() const
{
  return m_p->m_edge_builder_version;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const Int64x3& CartesianMeshAllocateBuildInfoInternal::
globalNbCells() const
{
  return m_p->m_global_nb_cells;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const Int32x3& CartesianMeshAllocateBuildInfoInternal::
ownNbCells() const
{
  return m_p->m_own_nb_cells;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const Int64x3& CartesianMeshAllocateBuildInfoInternal::
firstOwnCellOffset() const
{
  return m_p->m_first_own_cell_offset;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 CartesianMeshAllocateBuildInfoInternal::
cellUniqueIdOffset() const
{
  return m_p->m_cell_unique_id_offset;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 CartesianMeshAllocateBuildInfoInternal::
nodeUniqueIdOffset() const
{
  return m_p->m_node_unique_id_offset;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartesianMeshAllocateBuildInfo::
CartesianMeshAllocateBuildInfo(IPrimaryMesh* mesh)
: m_p(new Impl(mesh))
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartesianMeshAllocateBuildInfo::
~CartesianMeshAllocateBuildInfo()
{
  delete m_p;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshAllocateBuildInfo::
setInfos2D(std::array<Int64, 2> global_nb_cells,
           std::array<Int32, 2> own_nb_cells,
           Int64 cell_unique_id_offset,
           Int64 node_unique_id_offset)
{
  setInfos2D(global_nb_cells, own_nb_cells, { 0, 0 }, cell_unique_id_offset);
  m_p->m_node_unique_id_offset = node_unique_id_offset;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshAllocateBuildInfo::
setInfos2D(const Int64x2& global_nb_cells,
           const Int32x2& own_nb_cells,
           const Int64x2& first_own_cell_offset,
           Int64 cell_unique_id_offset)
{
  m_p->m_mesh_dimension = 2;
  m_p->m_global_nb_cells = { global_nb_cells.x, global_nb_cells.y, 0 };
  m_p->m_own_nb_cells = { own_nb_cells.x, own_nb_cells.y, 0 };
  m_p->m_first_own_cell_offset = { first_own_cell_offset.x, first_own_cell_offset.y, 0 };
  m_p->m_cell_unique_id_offset = cell_unique_id_offset;
  m_p->m_node_unique_id_offset = 0;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshAllocateBuildInfo::
setInfos3D(std::array<Int64, 3> global_nb_cells,
           std::array<Int32, 3> own_nb_cells,
           Int64 cell_unique_id_offset,
           Int64 node_unique_id_offset)
{
  setInfos3D(Int64x3(global_nb_cells), Int32x3(own_nb_cells), { 0, 0, 0 }, cell_unique_id_offset);
  m_p->m_node_unique_id_offset = node_unique_id_offset;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshAllocateBuildInfo::
setInfos3D(const Int64x3& global_nb_cells,
           const Int32x3& own_nb_cells,
           const Int64x3& first_own_cell_offset,
           Int64 cell_unique_id_offset)
{
  m_p->m_mesh_dimension = 3;
  m_p->m_global_nb_cells = global_nb_cells;
  m_p->m_own_nb_cells = own_nb_cells;
  m_p->m_first_own_cell_offset = first_own_cell_offset;
  m_p->m_cell_unique_id_offset = cell_unique_id_offset;
  m_p->m_node_unique_id_offset = 0;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshAllocateBuildInfo::
allocateMesh()
{
  IPrimaryMesh* pm = m_p->m_mesh;

  IMeshInitialAllocator* a = pm->initialAllocator();
  if (!a)
    ARCANE_FATAL("Mesh implementation has no IMeshInitialAllocator");

  ICartesianMeshInitialAllocator* specific_allocator = a->cartesianMeshAllocator();
  if (!specific_allocator)
    ARCANE_FATAL("Mesh does not support 'ICartesianMeshInitialAllocator'");

  pm->traceMng()->info() << "Allocate mesh from CartesianMeshAllocateBuildInfo";
  specific_allocator->allocate(*this);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartesianMeshAllocateBuildInfoInternal* CartesianMeshAllocateBuildInfo::
_internal()
{
  return &m_p->m_internal;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
