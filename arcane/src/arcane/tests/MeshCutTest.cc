// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshCutTest.cc                                              (C) 2000-2026 */
/*                                                                           */
/* MeshCut test service.                                                     */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/List.h"
#include "arcane/utils/Real3.h"

#include "arcane/core/Directory.h"
#include "arcane/core/IItemFamily.h"
#include "arcane/core/IMeshSection.h"
#include "arcane/core/IPostProcessorWriter.h"
#include "arcane/core/ITimeLoop.h"
#include "arcane/core/ITimeLoopMng.h"
#include "arcane/core/IVariableMng.h"
#include "arcane/core/ServiceBuilder.h"
#include "arcane/core/TimeLoopEntryPointInfo.h"

#include "arcane/tests/MeshCutTest_axl.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace ArcaneTest
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arcane;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class MeshCutTest
: public ArcaneMeshCutTestObject
{
 public:

  explicit MeshCutTest(const ModuleBuildInfo& mbi);
  ~MeshCutTest() override = default;

public:

  static void staticInitialize(ISubDomain* sd);

 public:

  void compute() override;

  void _initVars();
  void _checkVars(IMesh* new_mesh);
  Real3 _faceNormal(const Face& face, const Real3& center, VariableNodeReal3& node_coord) const;

 private:

  UniqueArray<Real> times;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_MODULE_MESHCUTTEST(MeshCutTest);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshCutTest::
MeshCutTest(const ModuleBuildInfo& mbi)
: ArcaneMeshCutTestObject(mbi)
{}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshCutTest::
staticInitialize(ISubDomain* sd)
{
  String time_loop_name("MeshCutTestLoop");

  ITimeLoopMng* tlm = sd->timeLoopMng();
  ITimeLoop* time_loop = tlm->createTimeLoop(time_loop_name);

  {
    List<TimeLoopEntryPointInfo> clist;
    clist.add(TimeLoopEntryPointInfo("MeshCutTest.Compute"));
    time_loop->setEntryPoints(ITimeLoop::WComputeLoop, clist);
  }

  {
    StringList clist;
    clist.add("MeshCutTest");
    time_loop->setRequiredModulesName(clist);
    clist.clear();
    clist.add("ArcanePostProcessing");
    clist.add("ArcaneCheckpoint");
    time_loop->setOptionalModulesName(clist);
  }

  tlm->registerTimeLoop(time_loop);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshCutTest::
compute()
{
  {
    m_global_time = globalIteration();
    times.add(m_global_time());
  }

  _initVars();

  //
  // MeshHandle meshhsection;
  // {
  //   ServiceBuilder<IMeshSection> spp0(mesh()->handle());
  //   Ref<IMeshSection> pp0 = spp0.createReference("MeshSection");
  //
  //   pp0->addPlane({0.95, 0, 0}, {1, 0, 0});
  //   pp0->addPlane({0.97, 0, 0}, {-1, 0, 0});
  //
  //   pp0->updateSection();
  //   meshhsection = pp0->meshSection();
  // }
  // IMesh* meshsection = meshhsection.mesh();
  //
  // if (options()->enablePostProcessing())
  // {
  //   ServiceBuilder<IPostProcessorWriter> spp(meshhsection);
  //   Ref<IPostProcessorWriter> pp = spp.createReference("VtkHdfV2PostProcessor");
  //   Directory output_directory = Directory(subDomain()->exportDirectory(), "amrtestpost1");
  //   pp->setBaseDirectoryName(output_directory.path());
  //   IPostProcessorWriter* post_processor = pp.get();
  //   post_processor->setTimes(times);
  //
  //   VariableList variables;
  //   variables.add(meshsection->nodesCoordinates().variable());
  //   post_processor->setVariables(variables);
  //
  //   ItemGroupList groups;
  //   groups.add(meshsection->allNodes());
  //   post_processor->setGroups(groups);
  //
  //   IVariableMng* vm = meshsection->variableMng();
  //   vm->writePostProcessing(post_processor);
  // }


  MeshHandle meshhcut;
  VariableCollection cloned_var;
  {
    ServiceBuilder<IMeshSection> spp0(mesh()->handle());
    // ServiceBuilder<IMeshSection> spp0(meshhsection);
    Ref<IMeshSection> pp0 = spp0.createReference("MeshCut");

    for (auto plane : options()->plane()) {
      pp0->addPlane(plane->p0() + (plane->p0Velocity() * globalIteration()), plane->normal());
    }

    VariableCollection vc;
    vc.add(m_on_cells0);
    vc.add(m_on_cells1);

    pp0->setVariables(vc);

    pp0->updateSection();
    meshhcut = pp0->meshSection();
    cloned_var = pp0->variables();
  }
  IMesh* meshcut = meshhcut.mesh();
  _checkVars(meshcut);

  if (options()->enablePostProcessing())
  {
    ServiceBuilder<IPostProcessorWriter> spp(meshhcut);
    Ref<IPostProcessorWriter> pp = spp.createReference("Ensight7PostProcessor");
    Directory output_directory = Directory(subDomain()->exportDirectory(), "amrtestpost1");
    output_directory.createDirectory();
    pp->setBaseDirectoryName(output_directory.path());
    IPostProcessorWriter* post_processor = pp.get();
    post_processor->setTimes(times);

    VariableList variables;
    variables.add(meshcut->nodesCoordinates().variable());
    for (VariableCollection::Enumerator i(cloned_var); ++i;) {
      variables.add(*i);
    }
    post_processor->setVariables(variables);

    ItemGroupList groups;
    groups.add(meshcut->allNodes());
    groups.add(meshcut->allCells());
    post_processor->setGroups(groups);

    IVariableMng* vm = meshcut->variableMng();
    vm->writePostProcessing(post_processor);
  }

  if (options()->enablePostProcessing())
  {
    ServiceBuilder<IPostProcessorWriter> spp(mesh()->handle());
    Ref<IPostProcessorWriter> pp = spp.createReference("Ensight7PostProcessor");
    Directory output_directory = Directory(subDomain()->exportDirectory(), "amrtestpost2");
    output_directory.createDirectory();
    pp->setBaseDirectoryName(output_directory.path());
    IPostProcessorWriter* post_processor = pp.get();
    post_processor->setTimes(times);

    VariableList variables;
    variables.add(mesh()->nodesCoordinates().variable());
    variables.add(m_on_cells0);
    variables.add(m_on_cells1);
    post_processor->setVariables(variables);

    ItemGroupList groups;
    groups.add(mesh()->allNodes());
    groups.add(mesh()->allCells());
    post_processor->setGroups(groups);

    IVariableMng* vm = mesh()->variableMng();
    vm->writePostProcessing(post_processor);
  }
}


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshCutTest::
_initVars()
{
  m_on_cells1.resize(3);

  ENUMERATE_(Cell, icell, mesh()->allCells()){
    m_on_cells0[icell] = icell->uniqueId().asInt32();
    for (Integer i = 0; i < 3; ++i) {
      m_on_cells1[icell][i] = icell->uniqueId().asInt32() * i;
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshCutTest::
_checkVars(IMesh* new_mesh)
{
  VariableCellReal on_cells0(VariableBuildInfo(new_mesh, m_on_cells0.name()));
  VariableCellReal3 bary_new_mesh(VariableBuildInfo(new_mesh, "Bary"));
  bary_new_mesh.fill(Real3{ 0 });
  VariableNodeReal3& node_coord_new_mesh = new_mesh->nodesCoordinates();
  VariableNodeReal3& node_coord_ori_mesh = mesh()->nodesCoordinates();

  UniqueArray<Int64> uid_to_check;
  uid_to_check.reserve(new_mesh->allCells().size());
  ENUMERATE_ (Cell, icell, new_mesh->allCells()) {
    uid_to_check.add(on_cells0[icell]);
    for (Node node : icell->nodes()) {
      bary_new_mesh[icell] += node_coord_new_mesh[node];
    }
    bary_new_mesh[icell] /= icell->nbNode();
  }

  UniqueArray<Int32> lid_to_check(new_mesh->allCells().size());

  IItemFamily* ori_mesh_cell_family = mesh()->cellFamily();
  ori_mesh_cell_family->itemsUniqueIdToLocalId(lid_to_check, uid_to_check, true);

  CellVectorView view_3d_cells = mesh()->cellFamily()->view(lid_to_check);

  Int32 iter = 0;
  ENUMERATE_ (Cell, inewcell, new_mesh->allCells()) {
    Cell cell_3d = view_3d_cells[iter++];

    Real3 bary_cell_3d{ 0 };
    for (Node node : cell_3d.nodes()) {
      bary_cell_3d += node_coord_ori_mesh[node];
    }
    bary_cell_3d /= cell_3d.nbNode();


    for (Face face : cell_3d.faces()) {
      Real3 bary_face{ 0 };
      for (Node node : face.nodes()) {
        bary_face += node_coord_ori_mesh[node];
      }
      bary_face /= face.nbNode();

      //info() << "FaceUID : " << face.uniqueId() << " -- BaryFace : " << bary_face << " -- Bary2D : " << bary_new_mesh[inewcell];

      Real3 normal = _faceNormal(face, bary_face, node_coord_ori_mesh);
      {
        Real d = math::dot({ bary_cell_3d - bary_face }, normal);
        Real dd = math::isNearlyZeroWithEpsilon(d, 1e-10) ? 0 : d;
        if (dd < 0) {
          normal *= -1;
        }
      }

      Real d = math::dot({ bary_new_mesh[inewcell] - bary_face }, normal);
      Real dd = math::isNearlyZeroWithEpsilon(d, 1e-10) ? 0 : d;
      if (dd < 0) {
        ARCANE_FATAL("Bad 2D");
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Real3 MeshCutTest::
_faceNormal(const Face& face, const Real3& center, VariableNodeReal3& node_coord) const
{
  NodeLocalIdView node_ids = face.nodeIds();
  Int32 nb_nodes = node_ids.size();

  Real3 normal = Real3::zero();

  for (Int32 i = 0; i < nb_nodes; ++i) {
    Int32 j = (i + 1) % nb_nodes;
    Real3 pi = node_coord[node_ids[i]];
    Real3 pj = node_coord[node_ids[j]];
    normal += Real(0.5) * math::cross(pi - center, pj - center);
  }

  return normal;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace ArcaneTest

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
