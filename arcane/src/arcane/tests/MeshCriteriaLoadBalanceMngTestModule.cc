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
#include "arcane/core/IParallelMng.h"
#include "arcane/core/ITimeLoopMng.h"
#include "arcane/core/ServiceBuilder.h"
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

  UniqueArray<Ref<VariableCellInt32>> m_density_meshes_ref;
  UniqueArray<Int32> m_sum;
  UniqueArray<Ref<IMeshPartitioner>> m_partitioners;
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
  m_sum.resize(subDomain()->meshes().size());
  m_partitioners.resize(subDomain()->meshes().size());

  for(Integer imesh = 0; imesh < subDomain()->meshes().size(); ++imesh){
    IMesh* mesh = subDomain()->meshes()[imesh];
    m_density_meshes_ref[imesh] = makeRef(new VariableCellInt32(VariableBuildInfo(mesh, "Density")));

    Int32 sum{};

    VariableCellInt32& density = *(m_density_meshes_ref[imesh].get());

    mesh->modifier()->setDynamic(true);
    ENUMERATE_ (Cell, icell, mesh->ownCells()){
      density[icell] = icell->uniqueId().asInt32();
      sum += icell->uniqueId().asInt32();
    }
    m_sum[imesh] = parallelMng()->reduce(IParallelMng::eReduceType::ReduceSum, sum);

    Int32 max = parallelMng()->reduce(IParallelMng::eReduceType::ReduceMax, sum);

    Real target = Real(m_sum[imesh]) / parallelMng()->commSize();

    Real lb_quality = (max - target) * 100 / (m_sum[imesh] - target); // TODO div 0

    debug() << "Initial Balancing"
            << " -- Mesh : " << mesh->name()
            << " -- NbCells : " << mesh->ownCells().size()
            << " -- Local sum density : " << sum
            << " -- All domains sum density : " << m_sum[imesh]
            << " -- Quality of LoadBalance (between 0 and 100, lower is better) : " << lb_quality;

    m_partitioners[imesh] = ServiceBuilder<IMeshPartitioner>::createReference(subDomain(), "DefaultPartitioner", mesh);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshCriteriaLoadBalanceMngTestModule::
loop()
{
  //for(Integer imesh = 0; imesh < subDomain()->meshes().size(); ++imesh){
  for (Integer imesh = 0; imesh < 2; ++imesh) {
    VariableCellInt32& density = *(m_density_meshes_ref[imesh].get());
    IMesh* mesh = subDomain()->meshes()[imesh];

    MeshCriteriaLoadBalanceMng mesh_criteria = MeshCriteriaLoadBalanceMng(subDomain(), mesh->handle());
    mesh_criteria.addCriterion(density);
  }

  for (Integer imesh = 0; imesh < subDomain()->meshes().size(); ++imesh) {
    IMesh* mesh = subDomain()->meshes()[imesh];
    VariableCellInt32& density = *(m_density_meshes_ref[imesh].get());

    Int32 sum{};
    ENUMERATE_ (Cell, icell, mesh->ownCells()) {
      sum += density[icell];
    }
    Int32 sum_sum = parallelMng()->reduce(IParallelMng::eReduceType::ReduceSum, sum);

    Int32 max = parallelMng()->reduce(IParallelMng::eReduceType::ReduceMax, sum);

    Real target = Real(sum_sum) / parallelMng()->commSize();

    Real lb_quality = (Real(max) - target) * 100 / (sum_sum - target);

    debug() << "Actual Balancing"
            << " -- Mesh : " << mesh->name()
            << " -- NbCells : " << mesh->ownCells().size()
            << " -- Local sum density : " << sum
            << " -- All domains sum density : " << sum_sum
            << " -- Original density : " << m_sum[imesh]
            << " -- Quality of LoadBalance (between 0 and 100, lower is better) : " << lb_quality;

    //    ENUMERATE_ (Cell, icell, mesh->ownCells()) {
    //      debug() << "CellUniqueId : " << icell->uniqueId();
    //    }
    ARCANE_ASSERT((sum_sum == m_sum[imesh]), ("Different sum"));
    subDomain()->timeLoopMng()->registerActionMeshPartition((IMeshPartitionerBase*)m_partitioners[imesh].get());
  }
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
