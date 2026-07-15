// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshSectionService.cc                                       (C) 2000-2026 */
/*                                                                           */
/* TODO.                                        */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/IMesh.h"
#include "arcane/core/IMeshFactoryMng.h"
#include "arcane/core/IMeshMng.h"
#include "arcane/core/IMeshModifier.h"
#include "arcane/core/IMeshSection.h"
#include "arcane/core/IPrimaryMesh.h"
#include "arcane/core/MeshBuildInfo.h"

#include "arcane/std/MeshSection_axl.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class MeshSectionService
: public ArcaneMeshSectionObject
{
 public:

  explicit MeshSectionService(const ServiceBuildInfo& sbi)
  : ArcaneMeshSectionObject(sbi)
  {}

 public:

  void addPlan(const Real3& p0, const Real3& normal) override;

 public:

  void setVariables(VariableCollection variables) override;
  void updateSection() override;

  MeshHandle meshSection() override
  {
    return m_cloned_mesh->handle();
  }
  VariableCollection variables() override { return {}; }

 private:

  void _createMesh();
  void _createCells(Int32& nb_cell, UniqueArray<Int64>& cells_infos, std::unordered_map<Int64, Real3>& pos_node);
  void _compute();

 private:

  VariableCollection m_variables_ori;
  IPrimaryMesh* m_cloned_mesh = nullptr;
  UniqueArray<std::pair<Real3, Real3>> m_plans;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE_MESHSECTION(MeshSectionService, MeshSectionService);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshSectionService::
addPlan(const Real3& p0, const Real3& normal)
{
  m_plans.add({ p0, math::normalizeReal3(normal) });
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshSectionService::
setVariables(VariableCollection variables)
{
  m_variables_ori = variables;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshSectionService::
updateSection()
{
  if (m_cloned_mesh == nullptr) {
    _createMesh();
  }
  else {
    m_cloned_mesh->modifier()->clearItems();
  }
  _compute();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshSectionService::
_createMesh()
{
  IMeshMng* mm = subDomain()->meshMng();
  IParallelMng* pm = subDomain()->parallelMng();
  // TODO gérer cas où il y a plusieurs services pour même maillage.
  MeshBuildInfo mbi(mesh()->name() + "_MeshSection");
  mbi.addParallelMng(makeRef(pm));
  m_cloned_mesh = mm->meshFactoryMng()->createMesh(mbi);
  m_cloned_mesh->modifier()->setDynamic(true);
  m_cloned_mesh->setDimension(mesh()->dimension());
  m_cloned_mesh->endAllocate();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshSectionService::
_createCells(Int32& sd_nb_cell, UniqueArray<Int64>& cells_infos, std::unordered_map<Int64, Real3>& pos_node)
{
  VariableNodeReal3& node_coord = mesh()->nodesCoordinates();

  ENUMERATE_ (Cell, icell, allCells()) {
    Real3 b{ 0 };
    for (Node node : icell->nodes()) {
      b += node_coord[node];
    }
    b /= icell->nbNode();

    bool in_cut = true;
    for (auto& [p0, normal] : m_plans) {
      const Real dist = math::dot({ b - p0 }, normal);
      if (dist < 0) {
        in_cut = false;
        break;
      }
    }

    if (!in_cut)
      continue;

    Int16 cell_type = icell->itemTypeId();
    cells_infos.add(cell_type);

    Int64 cell_uid = icell->uniqueId().asInt64();
    cells_infos.add(cell_uid);

    for (Node node : icell->nodes()) {
      Int64 node_uid = node.uniqueId().asInt64();
      cells_infos.add(node_uid);
      pos_node[node.uniqueId()] = node_coord[node];
    }
    ++sd_nb_cell;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshSectionService::
_compute()
{
  UniqueArray<Int64> cells_infos;
  cells_infos.reserve(10000);

  std::unordered_map<Int64, Real3> coord_map;

  Int32 nb_cell = 0;

  _createCells(nb_cell, cells_infos, coord_map);

  m_cloned_mesh->modifier()->addCells(nb_cell, cells_infos);
  m_cloned_mesh->modifier()->endUpdate();

  {
    VariableNodeReal3& node_coords(m_cloned_mesh->nodesCoordinates());
    ENUMERATE_ (Node, inode, m_cloned_mesh->allNodes()) {
      node_coords[inode] = coord_map[inode->uniqueId()];
    }
  }

  info() << "New mesh -- NbNode : " << m_cloned_mesh->nbNode() << " -- NbCells : " << m_cloned_mesh->nbCell();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
