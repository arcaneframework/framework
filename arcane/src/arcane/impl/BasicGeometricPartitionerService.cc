// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* BasicGeometricPartitionerService.cc                         (C) 2000-2025 */
/*                                                                           */
/* Service de partitionnement géométrique de maillage.                       */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/NotImplementedException.h"

#include "arcane/core/IParallelMng.h"
#include "arcane/core/IPrimaryMesh.h"
#include "arcane/core/IMeshUtilities.h"
#include "arcane/core/IMeshPartitioner.h"
#include "arcane/core/IItemFamily.h"

#include "arcane/core/IMeshPartitionConstraintMng.h"
#include "arcane/impl/BasicGeometricPartitionerService_axl.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Service de partitionnement géométrique de maillage.
 *
 * \warning En cours de développement. Ne pas utiliser en dehors de Arcane.
 */
class BasicGeometricPartitionerService
: public ArcaneBasicGeometricPartitionerServiceObject
, public IMeshPartitioner
{
 public:

  explicit BasicGeometricPartitionerService(const ServiceBuildInfo& sbi);

 public:

  IMesh* mesh() const override { return BasicService::mesh(); }
  void build() override {}
  void partitionMesh(bool initial_partition) override;
  void partitionMesh(bool initial_partition, Int32 nb_part) override
  {
    _partitionMesh(initial_partition, nb_part);
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
    ARCANE_THROW(NotImplementedException, "");
  }

  ILoadBalanceMng* loadBalanceMng() const override
  {
    ARCANE_THROW(NotImplementedException, "");
  }

 private:

  Real m_imbalance = 0.0;
  Real m_max_imbalance = 0.0;
  Real m_max_computation_time = 0.0;
  Int32 m_nb_weight = 0;
  ArrayView<float> m_cells_weight;
  UniqueArray<Real> m_computation_times;
  bool m_do_rcb = true;

 private:

  Real3 _computeBarycenter(const VariableCellReal3& cells_center);
  Real3x3 _computeInertiaTensor(Real3 center, const VariableCellReal3& cells_center);
  Real3 _findPrincipalAxis(Real3x3 tensor);
  void _partitionMesh2();
  void _partitionMesh(bool initial_partition, Int32 nb_part);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

BasicGeometricPartitionerService::
BasicGeometricPartitionerService(const ServiceBuildInfo& sbi)
: ArcaneBasicGeometricPartitionerServiceObject(sbi)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Fonction pour calculer le moment d'inertie du maillage
Real3 BasicGeometricPartitionerService::
_computeBarycenter(const VariableCellReal3& cells_center)
{
  Real3 center;
  CellGroup cells = mesh()->ownCells();
  IParallelMng* pm = mesh()->parallelMng();
  ENUMERATE_ (Cell, icell, cells) {
    center += cells_center[icell];
  }
  Int64 local_nb_cell = cells.size();
  Int64 total_nb_cell = pm->reduce(Parallel::ReduceSum, local_nb_cell);
  Real3 sum_center = pm->reduce(Parallel::ReduceSum, center);
  Real3 global_center = sum_center / static_cast<Real>(total_nb_cell);
  info() << "GlobalCenter=" << global_center;
  return global_center;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Fonction pour calculer le moment d'inertie du maillage
Real3x3 BasicGeometricPartitionerService::
_computeInertiaTensor(Real3 center, const VariableCellReal3& cells_center)
{
  CellGroup cells = mesh()->ownCells();
  IParallelMng* pm = mesh()->parallelMng();

  Real3x3 tensor;
  ENUMERATE_ (Cell, icell, cells) {
    Real3 cell_coord = cells_center[icell];
    double dx = cell_coord.x - center.x;
    double dy = cell_coord.y - center.y;
    double dz = cell_coord.z - center.z;

    tensor[0][0] += dy * dy + dz * dz;
    tensor[1][1] += dx * dx + dz * dz;
    tensor[2][2] += dx * dx + dy * dy;
    tensor[0][1] -= dx * dy;
    tensor[0][2] -= dx * dz;
    tensor[1][2] -= dy * dz;
  }

  Real3x3 sum_tensor = pm->reduce(Parallel::ReduceSum, tensor);

  tensor[1][0] = tensor[0][1];
  tensor[2][0] = tensor[0][2];
  tensor[2][1] = tensor[1][2];

  return sum_tensor;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Fonction pour trouver l'axe principal d'inertie
Real3 BasicGeometricPartitionerService::
_findPrincipalAxis(Real3x3 tensor)
{
  // Utilisation de la méthode des puissances pour trouver le vecteur propre principal
  Real3 v;
  Real3 v_new;

  v.x = 1.0;

  for (int iter = 0; iter < 100; ++iter) {
    v_new.x = tensor[0][0] * v.x + tensor[0][1] * v.y + tensor[0][2] * v.z;
    v_new.y = tensor[1][0] * v.x + tensor[1][1] * v.y + tensor[1][2] * v.z;
    v_new.z = tensor[2][0] * v.x + tensor[2][1] * v.y + tensor[2][2] * v.z;
    v = math::mutableNormalize(v_new);
  }
  return v_new;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Fonction pour partitionner le maillage en deux sous-domaines
void BasicGeometricPartitionerService::
_partitionMesh2()
{
  info() << "** ** DO_PARTITION_MESH2";
  const VariableNodeReal3& nodes_coordinates = mesh()->nodesCoordinates();

  // Calcule le centre des mailles
  VariableCellReal3 cells_center(VariableBuildInfo(mesh(), "ArcaneCellCenter"));
  ENUMERATE_ (Cell, icell, allCells()) {
    Real3 c;
    for (NodeLocalId n : icell->nodeIds())
      c += nodes_coordinates[n];
    cells_center[icell] = c / icell->nbNode();
  }
  // Calculer le centre de masse
  Real3 center = _computeBarycenter(cells_center);

  // Calculer le tenseur d'inertie
  Real3x3 tensor = _computeInertiaTensor(center, cells_center);

  // Trouver l'axe principal d'inertie
  Real3 eigenvector = _findPrincipalAxis(tensor);

  NodeGroup nodes = mesh()->allNodes();

  // Regarde dans quel partie va se trouver la maille
  VariableItemInt32& cells_new_owner = mesh()->cellFamily()->itemsNewOwner();
  ENUMERATE_ (Cell, icell, mesh()->allCells()) {
    Cell cell = *icell;
    const Real3 cell_coord = cells_center[icell];

    Real projection = 0.0;
    projection += (cell_coord.x - center.x) * eigenvector.x;
    projection += (cell_coord.y - center.y) * eigenvector.y;
    projection += (cell_coord.z - center.z) * eigenvector.z;

    Int32 new_owner = (projection < 0) ? 0 : 1;
    cells_new_owner[cell] = new_owner;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicGeometricPartitionerService::
_partitionMesh([[maybe_unused]] bool initial_partition, Int32 nb_part)
{
  // N'est valide que si \a nb_part == mesh()->parallelMng()->commSize();
  if (nb_part != 2)
    ARCANE_FATAL("Only cut in 2 part is supported");

  IPrimaryMesh* mesh = this->mesh()->toPrimaryMesh();

  info() << "Doing mesh partition with BasicGeometricPartitionerService nb_part=" << nb_part;
  IParallelMng* pm = mesh->parallelMng();
  Int32 nb_rank = pm->commSize();

  if (nb_rank == 1) {
    return;
  }

  VariableItemInt32& cells_new_owner = mesh->itemsNewOwner(IK_Cell);

  _partitionMesh2();

  cells_new_owner.synchronize();
  if (mesh->partitionConstraintMng()) {
    // Deal with Tied Cells
    mesh->partitionConstraintMng()->computeAndApplyConstraints();
  }
  mesh->utilities()->changeOwnersFromCells();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicGeometricPartitionerService::
partitionMesh(bool initial_partition)
{
  _partitionMesh(initial_partition, mesh()->parallelMng()->commSize());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE_BASICGEOMETRICPARTITIONERSERVICE(BasicGeometricPartitionerService,
                                                         BasicGeometricPartitionerService);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
