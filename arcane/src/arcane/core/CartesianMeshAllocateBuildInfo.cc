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
  UniqueArray<Int64> m_cells_infos;
  Int32 m_nb_cell = 0;
  CartesianMeshAllocateBuildInfoInternal m_internal;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ConstArrayView<Int64> CartesianMeshAllocateBuildInfoInternal::
cellsInfos() const
{
  return m_p->m_cells_infos;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 CartesianMeshAllocateBuildInfoInternal::
meshDimension() const
{
  return m_p->m_mesh_dimension;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 CartesianMeshAllocateBuildInfoInternal::
nbCell() const
{
  return m_p->m_nb_cell;
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
setInfos(std::array<Int64, 3> global_nb_cells,
         std::array<Int32, 3> own_nb_cells,
         Int64 cell_unique_id_offset,
         Int64 node_unique_id_offset)
{
  m_p->m_mesh_dimension = 3;

  UniqueArray<Int64>& cells_infos = m_p->m_cells_infos;

  auto [own_nb_cell_x, own_nb_cell_y, own_nb_cell_z] = own_nb_cells;
  auto [all_nb_cell_x, all_nb_cell_y, all_nb_cell_z] = global_nb_cells;

  Int32 own_nb_node_x = own_nb_cell_x + 1;
  Int32 own_nb_node_y = own_nb_cell_y + 1;
  Int32 own_nb_node_z = own_nb_cell_z + 1;
  Int32 own_nb_node_xy = CheckedConvert::multiply(own_nb_node_x, own_nb_node_y);
  Int32 own_nb_node_xyz = CheckedConvert::multiply(own_nb_node_xy, own_nb_node_z);

  Int32 own_nb_cell_xy = CheckedConvert::multiply(own_nb_cell_x, own_nb_cell_y);
  Int32 own_nb_cell_xyz = CheckedConvert::multiply(own_nb_cell_xy, own_nb_cell_z);

  Int64 all_nb_cell_xy = ((Int64)all_nb_cell_x) * ((Int64)all_nb_cell_y);

  Int64 all_nb_node_x = all_nb_cell_x + 1;
  Int64 all_nb_node_y = all_nb_cell_y + 1;
  Int64 all_nb_node_xy = all_nb_node_x * all_nb_node_y;

  m_p->m_nb_cell = own_nb_cell_xyz;

  cells_infos.resize(own_nb_cell_xyz * (1 + 1 + 8));

  // TODO: ne pas utiliser ce tableau et calculer directement en fonction
  // de la position (i,j,k) de la maille.
  UniqueArray<Int64> nodes_unique_id(own_nb_node_xyz);
  {
    Integer node_local_id = 0;
    for (Int64 z = 0; z < own_nb_node_z; ++z) {
      for (Int64 y = 0; y < own_nb_node_y; ++y) {
        for (Int64 x = 0; x < own_nb_node_x; ++x) {
          Int64 node_unique_id = node_unique_id_offset + x + y * all_nb_node_x + z * all_nb_node_xy;
          nodes_unique_id[node_local_id] = node_unique_id;
          ++node_local_id;
        }
      }
    }
  }

  Integer cells_infos_index = 0;
  for (Integer z = 0; z < own_nb_cell_z; ++z) {
    for (Integer y = 0; y < own_nb_cell_y; ++y) {
      for (Integer x = 0; x < own_nb_cell_x; ++x) {
        Int64 cell_unique_id = cell_unique_id_offset + x + y * all_nb_cell_x + z * all_nb_cell_xy;
        cells_infos[cells_infos_index] = IT_Hexaedron8;
        ++cells_infos_index;
        cells_infos[cells_infos_index] = cell_unique_id;
        ++cells_infos_index;
        Integer node_lid = x + y * own_nb_node_x + z * own_nb_node_xy;
        cells_infos[cells_infos_index + 0] = nodes_unique_id[node_lid];
        cells_infos[cells_infos_index + 1] = nodes_unique_id[node_lid + 1];
        cells_infos[cells_infos_index + 2] = nodes_unique_id[node_lid + own_nb_node_x + 1];
        cells_infos[cells_infos_index + 3] = nodes_unique_id[node_lid + own_nb_node_x + 0];
        cells_infos[cells_infos_index + 4] = nodes_unique_id[node_lid + own_nb_node_xy];
        cells_infos[cells_infos_index + 5] = nodes_unique_id[node_lid + own_nb_node_xy + 1];
        cells_infos[cells_infos_index + 6] = nodes_unique_id[node_lid + own_nb_node_xy + own_nb_node_x + 1];
        cells_infos[cells_infos_index + 7] = nodes_unique_id[node_lid + own_nb_node_xy + own_nb_node_x + 0];
        cells_infos_index += 8;
      }
    }
  }
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
