// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* UnstructuredMeshAllocateBuildInfo.cc                        (C) 2000-2023 */
/*                                                                           */
/* Informations pour allouer les entités d'un maillage non structuré.        */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/UnstructuredMeshAllocateBuildInfo.h"
#include "arcane/core/internal/UnstructuredMeshAllocateBuildInfoInternal.h"

#include "arcane/utils/Array.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/FatalErrorException.h"

#include "arcane/core/IPrimaryMesh.h"
#include "arcane/core/ItemTypeId.h"
#include "arcane/core/IMeshInitialAllocator.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class UnstructuredMeshAllocateBuildInfo::Impl
{
 public:

  explicit Impl(IPrimaryMesh* mesh)
  : m_mesh(mesh)
  {
    m_internal.m_p = this;
  }

  void addCell(ItemTypeId type_id, Int64 cell_uid, SmallSpan<const Int64> nodes_uid)
  {
    m_cells_infos.add(type_id);
    m_cells_infos.add(cell_uid);
    m_cells_infos.addRange(nodes_uid);
    ++m_nb_cell;
  }

 public:

  IPrimaryMesh* m_mesh = nullptr;
  Int32 m_mesh_dimension = -1;
  UniqueArray<Int64> m_cells_infos;
  Int32 m_nb_cell = 0;
  UnstructuredMeshAllocateBuildInfoInternal m_internal;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ConstArrayView<Int64> UnstructuredMeshAllocateBuildInfoInternal::
cellsInfos() const
{
  return m_p->m_cells_infos;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 UnstructuredMeshAllocateBuildInfoInternal::
meshDimension() const
{
  return m_p->m_mesh_dimension;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 UnstructuredMeshAllocateBuildInfoInternal::
nbCell() const
{
  return m_p->m_nb_cell;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

UnstructuredMeshAllocateBuildInfo::
UnstructuredMeshAllocateBuildInfo(IPrimaryMesh* mesh)
: m_p(new Impl(mesh))
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

UnstructuredMeshAllocateBuildInfo::
~UnstructuredMeshAllocateBuildInfo()
{
  delete m_p;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void UnstructuredMeshAllocateBuildInfo::
setMeshDimension(Int32 v)
{
  m_p->m_mesh_dimension = v;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void UnstructuredMeshAllocateBuildInfo::
addCell(ItemTypeId type_id, Int64 cell_uid, SmallSpan<const Int64> nodes_uid)
{
  m_p->addCell(type_id, cell_uid, nodes_uid);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void UnstructuredMeshAllocateBuildInfo::
allocateMesh()
{
  IPrimaryMesh* pm = m_p->m_mesh;

  IMeshInitialAllocator* a = pm->initialAllocator();
  if (!a)
    ARCANE_FATAL("Mesh implementation has no IMeshInitialAllocator");

  IUnstructuredMeshInitialAllocator* specific_allocator = a->unstructuredMeshAllocator();
  if (!specific_allocator)
    ARCANE_FATAL("Mesh does not support 'IUnstructuredMeshInitialAllocator'");

  pm->traceMng()->info() << "Allocate mesh from UnstructuredMeshAllocateBuildInfo";
  specific_allocator->allocate(*this);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void UnstructuredMeshAllocateBuildInfo::
preAllocate(Int32 nb_cell, Int64 nb_connectivity_node)
{
  m_p->m_cells_infos.reserve((nb_cell * 2) + nb_connectivity_node);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

UnstructuredMeshAllocateBuildInfoInternal* UnstructuredMeshAllocateBuildInfo::
_internal()
{
  return &m_p->m_internal;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
