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
#include "arcane/core/MeshCriteriaLoadBalanceMng.h"
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
  UniqueArray<Real> m_quality;
  UniqueArray<Ref<IMeshPartitionerBase>> m_partitioners;
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
  m_quality.resize(subDomain()->meshes().size());

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

    m_quality[imesh] = (max - target) * 100 / (m_sum[imesh] - target);

    debug() << "Initial Balancing"
            << " -- Mesh : " << mesh->name()
            << " -- NbCells : " << mesh->ownCells().size()
            << " -- Quality of LoadBalance (between 0 and 100, lower is better) : " << m_quality[imesh];
    debug() << " -- Local density : " << sum
            << " -- Target density per SubDomain : " << target
            << " -- ReduceMax density : " << max
            << " -- ReduceSum density : " << m_sum[imesh];

    m_partitioners[imesh] = ServiceBuilder<IMeshPartitionerBase>::createReference(subDomain(), "DefaultPartitioner", mesh);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshCriteriaLoadBalanceMngTestModule::
loop()
{
  for (Integer imesh = 0; imesh < subDomain()->meshes().size(); ++imesh) {
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
            << " -- Quality of LoadBalance (between 0 and 100, lower is better) : " << lb_quality
            << " -- Initial Quality of LoadBalance : " << m_quality[imesh];
    debug() << " -- Local density : " << sum
            << " -- Target density per SubDomain : " << target
            << " -- ReduceMax density : " << max
            << " -- ReduceSum density : " << sum_sum
            << " -- Initial ReduceSum density : " << m_sum[imesh];

    //    ENUMERATE_ (Cell, icell, mesh->ownCells()) {
    //      debug() << "CellUniqueId : " << icell->uniqueId();
    //    }
    ARCANE_ASSERT((sum_sum == m_sum[imesh]), ("Different sum"));
    if (lb_quality > m_quality[imesh]) {
      warning() << "Bad quality, need checking test";
    }

    subDomain()->timeLoopMng()->registerActionMeshPartition(static_cast<IMeshPartitionerBase*>(m_partitioners[imesh].get()));
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
