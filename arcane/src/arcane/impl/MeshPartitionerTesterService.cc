// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshPartitionerTesterService.cc                             (C) 2000-2022 */
/*                                                                           */
/* Mesh partitioner tester.                                                  */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/NotImplementedException.h"
#include "arcane/utils/CheckedConvert.h"

#include "arcane/core/IParallelMng.h"
#include "arcane/core/ISubDomain.h"
#include "arcane/core/CommonVariables.h"
#include "arcane/core/IMesh.h"
#include "arcane/core/IMeshSubMeshTransition.h"
#include "arcane/core/IMeshUtilities.h"
#include "arcane/core/FactoryService.h"
#include "arcane/core/ServiceFactory.h"
#include "arcane/core/Service.h"
#include "arcane/core/ItemPrinter.h"
#include "arcane/core/IMeshPartitioner.h"
#include "arcane/core/BasicService.h"

#include "arcane/core/IMeshPartitionConstraintMng.h"
#include "arcane/impl/MeshPartitionerTesterService_axl.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

// TODO: eventually (mid 2022), remove the implementation of 'IMeshPartitioner' and only
// keep that of 'IMeshPartitionerBase'.
// This will allow removing all methods with NotImplementedException.

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Mesh repartitioning test class.
 *
 * This class is used only to test mesh repartitioning.
 * It simply changes the owner of the meshes based
 * on the numbering and without taking into account potential imbalances
 * during computation time. 
 */
class MeshPartitionerTester
: public ArcaneMeshPartitionerTesterServiceObject
, public IMeshPartitioner
{
 public:
 public:

  MeshPartitionerTester(const ServiceBuildInfo& sbi);

 public:

  IMesh* mesh() const override { return BasicService::mesh(); }

  void build() override {}
  void partitionMesh(bool initial_partition) override;
  void partitionMesh(bool initial_partition, Int32 nb_part) override
  {
    ARCANE_UNUSED(initial_partition);
    ARCANE_UNUSED(nb_part);
    throw NotImplementedException(A_FUNCINFO);
  }

  void notifyEndPartition() override {}

 public:

  void setMaximumComputationTime(Real v) override { m_max_computation_time = v; }
  Real maximumComputationTime() const override { return m_max_computation_time; }
  void setComputationTimes(RealConstArrayView v) override { m_computation_times.copy(v); }
  RealConstArrayView computationTimes() const override { return m_computation_times; }
  void setImbalance(Real v) override { m_imbalance = v; }
  Real imbalance() const override { return m_imbalance; }
  void setMaxImbalance(Real v) override { m_max_imbalance = v; }
  Real maxImbalance() const override { return m_max_imbalance; }

  ArrayView<float> cellsWeight() const override { return m_cells_weight; }

  void setCellsWeight(ArrayView<float> weights, Integer nb_weight) override
  {
    m_cells_weight = weights;
    m_nb_weight = nb_weight;
  }

  void setILoadBalanceMng(ILoadBalanceMng*) override
  {
    throw NotImplementedException(A_FUNCINFO);
  }

  ILoadBalanceMng* loadBalanceMng() const override
  {
    throw NotImplementedException(A_FUNCINFO);
  }

 private:

  Real m_imbalance;
  Real m_max_imbalance;
  Real m_max_computation_time;
  Integer m_nb_weight;
  ArrayView<float> m_cells_weight;
  RealUniqueArray m_computation_times;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshPartitionerTester::
MeshPartitionerTester(const ServiceBuildInfo& sbi)
: ArcaneMeshPartitionerTesterServiceObject(sbi)
, m_imbalance(0.0)
, m_max_imbalance(0.0)
, m_max_computation_time(0.0)
, m_nb_weight(0)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshPartitionerTester::
partitionMesh(bool initial_partition)
{
  IPrimaryMesh* mesh = this->mesh()->toPrimaryMesh();

  Int32 sub_rank_divider = 0;
  if (options()) {
    sub_rank_divider = options()->subRankDivider();
  }
  info() << "Using MeshPartitionerTester sub_rank_divider=" << sub_rank_divider;

  ISubDomain* sd = subDomain();
  IParallelMng* pm = sd->parallelMng();
  Int32 nb_rank = pm->commSize();
  Int32 sid = pm->commRank();

  if (nb_rank == 1) {
    warning() << "Can't test the mesh repartitioning with"
              << "only one subdomain...";
    return;
  }

  VariableItemInt32& cells_new_owner = mesh->itemsNewOwner(IK_Cell);

  if (initial_partition) {
    // The goal is to have a correct partition but not
    // perfect, especially in the case of cuboids, because after
    // a real partitioner has nothing to do.
    // For the initial partitioning, assume that the mesh
    // initially generates cells with localId() that are topologically similar
    // to each other.
    // If we take a consecutive sequence of cells, we thus have a
    // not too badly formed cell block. To test the partitioner,
    // we generate 3x more blocks than subdomains and distribute them
    // among the subdomains.
    Int64 nb_cell = mesh->ownCells().size();
    Int64 nb_bloc = nb_rank * 3;
    Int64 cell_index = 0;
    ENUMERATE_CELL (icell, mesh->ownCells()) {
      Cell cell = *icell;
      // Use Int64 instead of Int32 to ensure no overflow.
      Int64 new_owner = ((cell_index * nb_bloc) / nb_cell) % nb_rank;
      cells_new_owner[cell] = CheckedConvert::toInt32(new_owner);
      ++cell_index;
    }
  }
  else {
    Integer current_iteration = sd->commonVariables().globalIteration();
    Integer nb_cell = mesh->ownCells().size();

    //Integer max_cell_index = (nb_rank+1)*5;
    Integer max_cell_index = nb_cell / 2;

    {
      Integer cell_index = 0;
      ENUMERATE_CELL (i_cell, mesh->ownCells()) {
        Cell cell = *i_cell;
        Int32 new_owner = cell.owner();
        if (cell_index < (max_cell_index + (sid * 10))) {
          // Force the first cell to remain in this subdomain
          // to ensure there is at least one left.
          if (cell_index != 0) {
            Int32 xx = (new_owner * 2 + current_iteration + cell_index / 10 + 17) % nb_rank;
            if (sub_rank_divider > 0) {
              xx = xx / sub_rank_divider;
              xx = (xx * sub_rank_divider + new_owner) % nb_rank;
            }
            new_owner = xx;
          }
          ++cell_index;
        }
        cells_new_owner[cell] = new_owner;
      }
    }
  }

  cells_new_owner.synchronize();
  if (mesh->partitionConstraintMng()) {
    // Deal with Tied Cells
    mesh->partitionConstraintMng()->computeAndApplyConstraints();
  }
  mesh->utilities()->changeOwnersFromCells();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE(MeshPartitionerTester,
                        ServiceProperty("MeshPartitionerTester", ST_SubDomain | ST_CaseOption),
                        ARCANE_SERVICE_INTERFACE(IMeshPartitioner),
                        ARCANE_SERVICE_INTERFACE(IMeshPartitionerBase));

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
