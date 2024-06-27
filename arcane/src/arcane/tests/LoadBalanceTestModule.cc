// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* LoadBalanceTestModule.cc                               (C) 2000-2024 */
/*                                                                           */
/* Module de test pour les implementations de LoadBalance.             */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ILoadBalanceMng.h"
#include "arcane/core/IMesh.h"
#include "arcane/core/IMeshModifier.h"
#include "arcane/core/ITimeLoopMng.h"
#include "arcane/tests/LoadBalanceTest_axl.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace ArcaneTest
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arcane;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class LoadBalanceTestModule
: public ArcaneLoadBalanceTestObject
{
 public:

  explicit LoadBalanceTestModule(const ModuleBuildInfo& mbi);

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

LoadBalanceTestModule::
LoadBalanceTestModule(const ModuleBuildInfo& mbi)
: ArcaneLoadBalanceTestObject(mbi)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void LoadBalanceTestModule::
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

void LoadBalanceTestModule::
loop()
{
  for(Integer imesh = 0; imesh < subDomain()->meshes().size(); ++imesh){
    IMesh* mesh = subDomain()->meshes()[imesh];
    VariableCellReal& density = *(m_density_meshes_ref[imesh].get());

    ILoadBalanceMng* lb_mng = subDomain()->loadBalanceMng();
    lb_mng->addCriterion(density);

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
  subDomain()->timeLoopMng()->registerActionMeshPartition((IMeshPartitionerBase*)options()->partitioner());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void LoadBalanceTestModule::
exit()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_MODULE_LOADBALANCETEST(LoadBalanceTestModule);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace ArcaneTest

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
