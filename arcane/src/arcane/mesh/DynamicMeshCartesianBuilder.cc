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
#include "arcane/core/CartesianGridDimension.h"
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
  UniqueArray<Int64>& cells_infos = m_cells_infos;

  const Int64 cell_unique_id_offset = m_build_info->cellUniqueIdOffset();
  const Int64 node_unique_id_offset = m_build_info->nodeUniqueIdOffset();

  CartesianGridDimension all_grid_dimension(m_build_info->globalNbCells());
  CartesianGridDimension own_grid_dimension(m_build_info->ownNbCells());
  Int64x3 first_own_cell_offset(m_build_info->firstOwnCellOffset());

  Int64x3 own_nb_cell = own_grid_dimension.nbCell();

  const Int64 own_nb_cell_xy = own_grid_dimension.totalNbCell();
  m_nb_cell = CheckedConvert::toInt32(own_nb_cell_xy);

  cells_infos.resize(own_nb_cell_xy * (1 + 1 + 4));

  Integer cells_infos_index = 0;
  CartesianGridDimension::NodeUniqueIdComputer2D node_uid_computer(all_grid_dimension.getNodeComputer2D(node_unique_id_offset));
  CartesianGridDimension::CellUniqueIdComputer2D cell_uid_computer(all_grid_dimension.getCellComputer2D(cell_unique_id_offset));

  for (Integer y = 0; y < own_nb_cell.y; ++y) {
    for (Integer x = 0; x < own_nb_cell.x; ++x) {
      Int64 gx = x + first_own_cell_offset.x;
      Int64 gy = y + first_own_cell_offset.y;
      Int64 cell_unique_id = cell_uid_computer.compute(gx, gy);
      cells_infos[cells_infos_index] = IT_Quad4;
      ++cells_infos_index;
      cells_infos[cells_infos_index] = cell_unique_id;
      ++cells_infos_index;
      std::array<Int64, 4> node_uids = node_uid_computer.computeForCell(gx, gy);
      for (Int32 i = 0; i < 4; ++i)
        cells_infos[cells_infos_index + i] = node_uids[i];
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

  CartesianGridDimension all_grid_dimension(m_build_info->globalNbCells());
  CartesianGridDimension own_grid_dimension(m_build_info->ownNbCells());
  Int64x3 first_own_cell_offset(m_build_info->firstOwnCellOffset());

  Int64x3 own_nb_cell = own_grid_dimension.nbCell();

  const Int64 own_nb_cell_xyz = own_grid_dimension.totalNbCell();
  m_nb_cell = CheckedConvert::toInt32(own_nb_cell_xyz);

  cells_infos.resize(own_nb_cell_xyz * (1 + 1 + 8));

  Integer cells_infos_index = 0;
  CartesianGridDimension::NodeUniqueIdComputer3D node_uid_computer(all_grid_dimension.getNodeComputer3D(node_unique_id_offset));
  CartesianGridDimension::CellUniqueIdComputer3D cell_uid_computer(all_grid_dimension.getCellComputer3D(cell_unique_id_offset));

  for (Integer z = 0; z < own_nb_cell.z; ++z) {
    for (Integer y = 0; y < own_nb_cell.y; ++y) {
      for (Integer x = 0; x < own_nb_cell.x; ++x) {
        Int64 gx = x + first_own_cell_offset.x;
        Int64 gy = y + first_own_cell_offset.y;
        Int64 gz = z + first_own_cell_offset.z;
        Int64 cell_unique_id = cell_uid_computer.compute(gx, gy, gz);
        cells_infos[cells_infos_index] = IT_Hexaedron8;
        ++cells_infos_index;
        cells_infos[cells_infos_index] = cell_unique_id;
        ++cells_infos_index;
        std::array<Int64, 8> node_uids = node_uid_computer.computeForCell(gx, gy, gz);
        for (Int32 i = 0; i < 8; ++i)
          cells_infos[cells_infos_index + i] = node_uids[i];
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
