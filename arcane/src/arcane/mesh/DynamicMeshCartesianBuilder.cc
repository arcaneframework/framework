// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DynamicMeshCartesianBuilder.cc                              (C) 2000-2023 */
/*                                                                           */
/* Génération des maillages cartésiens pour DynamicMesh.                     */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/CheckedConvert.h"

#include "arcane/core/IMeshUniqueIdMng.h"
#include "arcane/core/internal/CartesianMeshAllocateBuildInfoInternal.h"

#include "arcane/mesh/DynamicMesh.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Informations pour un échange de maillage entre sous-domaines.
 *
 * L'utilisation de cette classe est réservée à DynamicMesh.
 */
class ARCANE_MESH_EXPORT DynamicMeshCartesianBuilder
: public TraceAccessor
{
 public:

  DynamicMeshCartesianBuilder(DynamicMesh* mesh,
                              CartesianMeshAllocateBuildInfoInternal* build_info);

 protected:

  UniqueArray<Int64> m_cells_infos;
  Int32 m_nb_cell = 0;
  DynamicMesh* m_mesh = nullptr;
  CartesianMeshAllocateBuildInfoInternal* m_build_info = nullptr;

 public:

  void allocate();

 protected:

  virtual void _buildCellList() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class DynamicMeshCartesian2DBuilder
: public DynamicMeshCartesianBuilder
{
 public:

  DynamicMeshCartesian2DBuilder(DynamicMesh* mesh,
                                CartesianMeshAllocateBuildInfoInternal* build_info)
  : DynamicMeshCartesianBuilder(mesh, build_info)
  {
  }

 protected:

  void _buildCellList() override;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMeshCartesian2DBuilder::
_buildCellList()
{
  const Int64 cell_unique_id_offset = m_build_info->cellUniqueIdOffset();
  const Int64 node_unique_id_offset = m_build_info->nodeUniqueIdOffset();

  auto own_nb_cells = m_build_info->ownNbCells();
  auto global_nb_cells = m_build_info->globalNbCells();

  UniqueArray<Int64>& cells_infos = m_cells_infos;

  const Int32 own_nb_cell_x = own_nb_cells[0];
  const Int32 own_nb_cell_y = own_nb_cells[1];
  const Int64 all_nb_cell_x = global_nb_cells[0];

  Int32 own_nb_cell_xy = CheckedConvert::multiply(own_nb_cell_x, own_nb_cell_y);

  Int64 all_nb_node_x = all_nb_cell_x + 1;

  m_nb_cell = own_nb_cell_xy;

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

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class DynamicMeshCartesian3DBuilder
: public DynamicMeshCartesianBuilder
{
 public:

  DynamicMeshCartesian3DBuilder(DynamicMesh* mesh,
                                CartesianMeshAllocateBuildInfoInternal* build_info)
  : DynamicMeshCartesianBuilder(mesh, build_info)
  {
  }

 public:

  void _buildCellList() override;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMeshCartesian3DBuilder::
_buildCellList()
{
  UniqueArray<Int64>& cells_infos = m_cells_infos;

  const Int64 cell_unique_id_offset = m_build_info->cellUniqueIdOffset();
  const Int64 node_unique_id_offset = m_build_info->nodeUniqueIdOffset();

  auto [own_nb_cell_x, own_nb_cell_y, own_nb_cell_z] = m_build_info->ownNbCells();
  auto [all_nb_cell_x, all_nb_cell_y, all_nb_cell_z] = m_build_info->globalNbCells();

  Int32 own_nb_cell_xy = CheckedConvert::multiply(own_nb_cell_x, own_nb_cell_y);
  Int32 own_nb_cell_xyz = CheckedConvert::multiply(own_nb_cell_xy, own_nb_cell_z);

  Int64 all_nb_cell_xy = all_nb_cell_x * all_nb_cell_y;

  Int64 all_nb_node_x = all_nb_cell_x + 1;
  Int64 all_nb_node_y = all_nb_cell_y + 1;
  Int64 all_nb_node_xy = all_nb_node_x * all_nb_node_y;

  m_nb_cell = own_nb_cell_xyz;

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

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

DynamicMeshCartesianBuilder::
DynamicMeshCartesianBuilder(DynamicMesh* mesh, CartesianMeshAllocateBuildInfoInternal* build_info)
: TraceAccessor(mesh->traceMng())
, m_mesh(mesh)
, m_build_info(build_info)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicMeshCartesianBuilder::
allocate()
{
  auto* x = m_build_info;
  const Int32 dimension = m_build_info->meshDimension();

  m_mesh->setDimension(dimension);

  Int32 face_builder_version = x->faceBuilderVersion();
  if (face_builder_version >= 0)
    m_mesh->meshUniqueIdMng()->setFaceBuilderVersion(face_builder_version);
  Int32 edge_builder_version = x->edgeBuilderVersion();
  if (edge_builder_version >= 0)
    m_mesh->meshUniqueIdMng()->setEdgeBuilderVersion(edge_builder_version);

  _buildCellList();

  m_mesh->allocateCells(m_nb_cell, m_cells_infos, true);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_MESH_EXPORT void
allocateCartesianMesh(DynamicMesh* mesh, CartesianMeshAllocateBuildInfo& build_info)
{
  auto* x = build_info._internal();
  Int32 dimension = x->meshDimension();

  if (dimension == 3) {
    DynamicMeshCartesian3DBuilder builder(mesh, x);
    builder.allocate();
  }
  else if (dimension == 2) {
    DynamicMeshCartesian2DBuilder builder(mesh, x);
    builder.allocate();
  }
  else
    ARCANE_FATAL("Not supported dimension '{0}'. Only 2 or 3 is supported", dimension);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
