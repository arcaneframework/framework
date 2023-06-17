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
  Int32 m_face_builder_version = -1;
  Int32 m_edge_builder_version = -1;
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
  m_p->m_mesh_dimension = 2;

  UniqueArray<Int64>& cells_infos = m_p->m_cells_infos;

  auto [own_nb_cell_x, own_nb_cell_y] = own_nb_cells;
  auto [all_nb_cell_x, all_nb_cell_y] = global_nb_cells;

  Int32 own_nb_cell_xy = CheckedConvert::multiply(own_nb_cell_x, own_nb_cell_y);

  Int64 all_nb_node_x = all_nb_cell_x + 1;

  m_p->m_nb_cell = own_nb_cell_xy;

  cells_infos.resize(own_nb_cell_xy * (1 + 1 + 4));

  //! Classe pour calculer le uniqueId() d'un noeud en fonction de sa position dans la grille.
  class NodeUniqueIdComputer
  {
   public:

    NodeUniqueIdComputer(Int64 base_offset, Int64 all_nb_node_x)
    : m_base_offset(base_offset)
    , m_all_nb_node_x(all_nb_node_x)
    {}

   public:

    Int64 compute(Int32 x, Int32 y)
    {
      return m_base_offset + x + y * m_all_nb_node_x;
    }

   private:

    Int64 m_base_offset;
    Int64 m_all_nb_node_x;
  };

  Integer cells_infos_index = 0;
  NodeUniqueIdComputer node_uid_computer(node_unique_id_offset, all_nb_node_x);

  for (Integer y = 0; y < own_nb_cell_y; ++y) {
    for (Integer x = 0; x < own_nb_cell_x; ++x) {
      Int64 cell_unique_id = cell_unique_id_offset + x + y * all_nb_cell_x;
      cells_infos[cells_infos_index] = IT_Quad4;
      ++cells_infos_index;
      cells_infos[cells_infos_index] = cell_unique_id;
      ++cells_infos_index;
      cells_infos[cells_infos_index + 0] = node_uid_computer.compute(x + 0, y + 0);
      cells_infos[cells_infos_index + 1] = node_uid_computer.compute(x + 1, y + 0);
      cells_infos[cells_infos_index + 2] = node_uid_computer.compute(x + 1, y + 1);
      cells_infos[cells_infos_index + 3] = node_uid_computer.compute(x + 0, y + 1);
      cells_infos_index += 4;
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshAllocateBuildInfo::
setInfos3D(std::array<Int64, 3> global_nb_cells,
           std::array<Int32, 3> own_nb_cells,
           Int64 cell_unique_id_offset,
           Int64 node_unique_id_offset)
{
  m_p->m_mesh_dimension = 3;

  UniqueArray<Int64>& cells_infos = m_p->m_cells_infos;

  auto [own_nb_cell_x, own_nb_cell_y, own_nb_cell_z] = own_nb_cells;
  auto [all_nb_cell_x, all_nb_cell_y, all_nb_cell_z] = global_nb_cells;

  Int32 own_nb_cell_xy = CheckedConvert::multiply(own_nb_cell_x, own_nb_cell_y);
  Int32 own_nb_cell_xyz = CheckedConvert::multiply(own_nb_cell_xy, own_nb_cell_z);

  Int64 all_nb_cell_xy = all_nb_cell_x * all_nb_cell_y;

  Int64 all_nb_node_x = all_nb_cell_x + 1;
  Int64 all_nb_node_y = all_nb_cell_y + 1;
  Int64 all_nb_node_xy = all_nb_node_x * all_nb_node_y;

  m_p->m_nb_cell = own_nb_cell_xyz;

  cells_infos.resize(own_nb_cell_xyz * (1 + 1 + 8));

  //! Classe pour calculer le uniqueId() d'un noeud en fonction de sa position dans la grille.
  class NodeUniqueIdComputer
  {
   public:

    NodeUniqueIdComputer(Int64 base_offset, Int64 all_nb_node_x, Int64 all_nb_node_xy)
    : m_base_offset(base_offset)
    , m_all_nb_node_x(all_nb_node_x)
    , m_all_nb_node_xy(all_nb_node_xy)
    {}

   public:

    Int64 compute(Int32 x, Int32 y, Int32 z)
    {
      return m_base_offset + x + y * m_all_nb_node_x + z * m_all_nb_node_xy;
    }

   private:

    Int64 m_base_offset;
    Int64 m_all_nb_node_x;
    Int64 m_all_nb_node_xy;
  };

  Integer cells_infos_index = 0;
  NodeUniqueIdComputer node_uid_computer(node_unique_id_offset, all_nb_node_x, all_nb_node_xy);

  for (Integer z = 0; z < own_nb_cell_z; ++z) {
    for (Integer y = 0; y < own_nb_cell_y; ++y) {
      for (Integer x = 0; x < own_nb_cell_x; ++x) {
        Int64 cell_unique_id = cell_unique_id_offset + x + y * all_nb_cell_x + z * all_nb_cell_xy;
        cells_infos[cells_infos_index] = IT_Hexaedron8;
        ++cells_infos_index;
        cells_infos[cells_infos_index] = cell_unique_id;
        ++cells_infos_index;
        cells_infos[cells_infos_index + 0] = node_uid_computer.compute(x + 0, y + 0, z + 0);
        cells_infos[cells_infos_index + 1] = node_uid_computer.compute(x + 1, y + 0, z + 0);
        cells_infos[cells_infos_index + 2] = node_uid_computer.compute(x + 1, y + 1, z + 0);
        cells_infos[cells_infos_index + 3] = node_uid_computer.compute(x + 0, y + 1, z + 0);
        cells_infos[cells_infos_index + 4] = node_uid_computer.compute(x + 0, y + 0, z + 1);
        cells_infos[cells_infos_index + 5] = node_uid_computer.compute(x + 1, y + 0, z + 1);
        cells_infos[cells_infos_index + 6] = node_uid_computer.compute(x + 1, y + 1, z + 1);
        cells_infos[cells_infos_index + 7] = node_uid_computer.compute(x + 0, y + 1, z + 1);
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
