// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshCriteriaLoadBalanceMngTestModule.cc                     (C) 2000-2024 */
/*                                                                           */
/* Module de test pour les implementations de MeshCriteriaLoadBalanceMng.    */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ILoadBalanceMng.h"
#include "arcane/core/IMesh.h"
#include "arcane/core/IMeshModifier.h"
#include "arcane/core/ITimeLoopMng.h"
#include "arcane/impl/MeshCriteriaLoadBalanceMng.h"
#include "arcane/tests/MeshCriteriaLoadBalanceMngTest_axl.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace ArcaneTest
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arcane;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class MeshCriteriaLoadBalanceMngTestModule
: public ArcaneMeshCriteriaLoadBalanceMngTestObject
{
 public:

  explicit MeshCriteriaLoadBalanceMngTestModule(const ModuleBuildInfo& mbi);

 public:

  VersionInfo versionInfo() const override { return { 1, 0, 0 }; }
  void init() override;
  void loop() override;
  void exit() override;

 private:
  UniqueArray<Ref<VariableCellReal>> m_density_meshes_ref;
  Ref<VariableCellReal> m_density_mesh0_ref;
  Ref<VariableCellReal> m_density_mesh1_ref;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshCriteriaLoadBalanceMngTestModule::
MeshCriteriaLoadBalanceMngTestModule(const ModuleBuildInfo& mbi)
: ArcaneMeshCriteriaLoadBalanceMngTestObject(mbi)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshCriteriaLoadBalanceMngTestModule::
init()
{
  m_density_meshes_ref.resize(subDomain()->meshes().size());

  for(Integer imesh = 0; imesh < subDomain()->meshes().size(); ++imesh){
    IMesh* mesh = subDomain()->meshes()[imesh];
    m_density_meshes_ref[imesh] = makeRef(new VariableCellReal(VariableBuildInfo(mesh, "Density")));

    VariableCellReal& density = *(m_density_meshes_ref[imesh].get());

    mesh->modifier()->setDynamic(true);
    ENUMERATE_ (Cell, icell, mesh->ownCells()){
      density[icell] = icell->uniqueId().asInt32();
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshCriteriaLoadBalanceMngTestModule::
loop()
{
  //for(Integer imesh = 0; imesh < subDomain()->meshes().size(); ++imesh){
  for (Integer imesh = 0; imesh < 2; ++imesh) {
    VariableCellReal& density = *(m_density_meshes_ref[imesh].get());
    IMesh* mesh = subDomain()->meshes()[imesh];

    MeshCriteriaLoadBalanceMng mesh_criteria = MeshCriteriaLoadBalanceMng(subDomain(), mesh->handle());
    mesh_criteria.addCriterion(density);
  }

  for (Integer imesh = 0; imesh < subDomain()->meshes().size(); ++imesh) {
    IMesh* mesh = subDomain()->meshes()[imesh];
    VariableCellReal& density = *(m_density_meshes_ref[imesh].get());

    Real sum = {};
    ENUMERATE_ (Cell, icell, mesh->ownCells()) {
      sum += density[icell];
    }
    debug() << "Actual Balancing"
            << " -- Mesh : " << mesh->name()
            << " -- NbCells : " << mesh->ownCells().size()
            << " -- Sum density : " << sum;

    ENUMERATE_ (Cell, icell, mesh->ownCells()) {
      debug() << "CellUniqueId : " << icell->uniqueId();
    }
  }
  subDomain()->timeLoopMng()->registerActionMeshPartition((IMeshPartitionerBase*)options()->partitioner0());
  subDomain()->timeLoopMng()->registerActionMeshPartition((IMeshPartitionerBase*)options()->partitioner1());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshCriteriaLoadBalanceMngTestModule::
exit()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_MODULE_MESHCRITERIALOADBALANCEMNGTEST(MeshCriteriaLoadBalanceMngTestModule);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace ArcaneTest

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
