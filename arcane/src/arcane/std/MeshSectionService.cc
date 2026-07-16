// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshSectionService.cc                                       (C) 2000-2026 */
/*                                                                           */
/* Service allowing the creation of a mesh with a section of another mesh.   */
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

/*!
 * \brief Service allowing the creation of a mesh with a section of another
 * mesh.
 *
 * You can define a section of mesh with the method \a addPlan(). All cells
 * (their barycenter) between these planes will be copied in a second mesh
 * created by this service. You can get this mesh with the method
 * \a meshSection().
 */
class MeshSectionService
: public ArcaneMeshSectionObject
{
 public:

  explicit MeshSectionService(const ServiceBuildInfo& sbi)
  : ArcaneMeshSectionObject(sbi)
  {}

 public:

  void addPlane(const Real3& p0, const Real3& normal) override;

  void setVariables(VariableCollection variables) override;
  void updateSection() override;

  MeshHandle meshSection() override
  {
    return m_cloned_mesh->handle();
  }
  VariableCollection variables() override { return {}; }

 private:

  void _createMesh();
  void _createCells(Int32& nb_cell, UniqueArray<Int64>& cells_infos, Int32& nb_face, UniqueArray<Int64>& faces_infos, std::unordered_map<Int64, Real3>& pos_node);
  void _compute();

 private:

  // VariableCollection m_variables_ori;
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
addPlane(const Real3& p0, const Real3& normal)
{
  m_plans.add({ p0, math::normalizeReal3(normal) });
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshSectionService::
setVariables(VariableCollection variables)
{
  ARCANE_UNUSED(variables);
  ARCANE_NOT_YET_IMPLEMENTED("Not supported yet");
  // m_variables_ori = variables;
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
_createCells(Int32& sd_nb_cell, UniqueArray<Int64>& cells_infos, Int32& sd_nb_face, UniqueArray<Int64>& faces_infos, std::unordered_map<Int64, Real3>& pos_node)
{
  VariableNodeReal3& node_coord = mesh()->nodesCoordinates();

  VariableFaceBool is_added(VariableBuildInfo(mesh(), "IsAdded"));
  is_added.fill(false);

  ENUMERATE_ (Cell, icell, ownCells()) {
    {
      Real3 b{ 0 };
      for (Node node : icell->nodes()) {
        b += node_coord[node];
      }
      b /= icell->nbNode();

      bool in_plan = true;
      for (auto& [p0, normal] : m_plans) {
        const Real dist = math::dot({ b - p0 }, normal);
        if (dist < 0) {
          in_plan = false;
          break;
        }
      }

      if (!in_plan)
        continue;
    }

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

    for (Face face : icell->faces()) {
      if (is_added[face]) continue;
      is_added[face] = true;

      Int16 face_type = face.itemTypeId();
      faces_infos.add(face_type);

      Int64 face_uid = face.uniqueId().asInt64();
      faces_infos.add(face_uid);

      for (Node node : face.nodes()) {
        Int64 node_uid = node.uniqueId().asInt64();
        faces_infos.add(node_uid);
        pos_node[node.uniqueId()] = node_coord[node];
      }
      ++sd_nb_face;
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshSectionService::
_compute()
{
  UniqueArray<Int64> cells_infos;
  cells_infos.reserve(10000);

  UniqueArray<Int64> faces_infos;
  faces_infos.reserve(10000);

  std::unordered_map<Int64, Real3> coord_map;

  Int32 nb_cell = 0;
  Int32 nb_face = 0;

  _createCells(nb_cell, cells_infos, nb_face, faces_infos, coord_map);

  m_cloned_mesh->modifier()->addFaces(nb_face, faces_infos);
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
