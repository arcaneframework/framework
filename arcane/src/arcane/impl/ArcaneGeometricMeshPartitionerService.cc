// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ArcaneGeometricMeshPartitionerService.cc                    (C) 2000-2025 */
/*                                                                           */
/* Mesh geometric partitioning service.                                      */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/NotImplementedException.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/FixedArray.h"
#include "arcane/utils/SmallArray.h"

#include "arcane/core/IParallelMng.h"
#include "arcane/core/IPrimaryMesh.h"
#include "arcane/core/IMeshUtilities.h"
#include "arcane/core/IMeshPartitioner.h"
#include "arcane/core/IItemFamily.h"
#include "arcane/core/ItemVector.h"
#include "arcane/core/MathUtils.h"

#include "arcane/core/IMeshPartitionConstraintMng.h"
#include "arcane/impl/ArcaneGeometricMeshPartitionerService_axl.h"

#include "arcane_internal_config.h"
#include <limits>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Class to calculate the eigenvalues and eigenvectors of a matrix.
 *
 * The calculation is done using the power method.
 * This is not the fastest or most robust method numerically, but we do not need high precision in
 * the calculation of eigenvalues and eigenvectors. If needed, we could use
 * the Jacobi algorithm or a QR method.
 *
 * The eigenvalues are sorted in ascending order.
 * \note Since the algorithm is iterative, it may happen that the order is not
 * strictly increasing if two eigenvalues are close.
 */
class EigenValueAndVectorComputer
{
 private:

  //! Result of applying the power method
  struct PowerResult
  {
    Real eigen_value;
    Real3 eigen_vector;
  };

 public:

  /*!
   * \brief Calculates the eigenvalues and eigenvectors of \a orig_matrix.
   */
  void computeForMatrix(const Real3x3& orig_matrix)
  {
    Real3x3 matrix = orig_matrix;

    constexpr int nb_value = 3;
    // Iterates to calculate the eigenvalues and eigenvectors
    // The power method calculates the highest eigenvalue. Since we want to sort the eigenvalues
    // in ascending order, we range them in reverse order.

    for (int i = (nb_value - 1); i >= 0; --i) {
      // Calculate the first eigenvector (the largest)
      PowerResult result = _applyPowerIteration(matrix);
      m_eigen_values[i] = result.eigen_value;
      m_eigen_vectors[i] = result.eigen_vector;

      // Apply deflation to eliminate the eigenvector
      // that was just calculated.
      // We do not need to do this for the last iteration
      if (i != 0)
        _deflateMatrix(matrix, result.eigen_value, result.eigen_vector);
    }
  }
  //! Returns the eigenvalues of the matrix in ascending order
  Real3 eigenValues() const { return m_eigen_values; }
  //! Returns the eigenvectors of the matrix in ascending order
  Real3x3 eigenVectors() const { return m_eigen_vectors; }

 private:

  // Subtracts an eigenvector from the matrix for deflation
  static void _deflateMatrix(Real3x3& matrix, double eigenvalue, Real3 eigenvector)
  {
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        matrix[i][j] -= eigenvalue * eigenvector[i] * eigenvector[j];
      }
    }
  }

  // Applies the power method
  static PowerResult _applyPowerIteration(const Real3x3& matrix)
  {
    constexpr Int32 max_iteration = 1000;
    constexpr Real tolerance = 1e-16;
    Real eigenvalue = 0.0;

    // Initialization with an initial vector (it could be random)
    Real3 b{ 1.0, 1.0, 1.0 };
    Real3 b_next;

    // Power method iterations
    for (Int32 iter = 0; iter < max_iteration; ++iter) {
      b_next = math::multiply(matrix, b);

      eigenvalue = b_next.normL2();
      if (math::isNearlyZero(eigenvalue))
        break;
      b_next = b_next / eigenvalue;

      // Check convergence
      Real diff = math::squareNormL2(b_next - b);
      if (diff < tolerance) {
        break; // If the difference is sufficiently small, we have converged
      }

      b = b_next;
    }

    return { eigenvalue, b_next };
  }

 private:

  //! Eigenvalues
  Real3 m_eigen_values;

  //! Eigenvectors
  Real3x3 m_eigen_vectors;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Class to create a binary tree.
 */
class BinaryTree
: public TraceAccessor
{
 public:

  /*!
   * \brief Information about a tree node.
   *
   * A node has 0, 1, or 2 children and 0 or 1 parent.
   * If a node does not have a child on one side, then it
   * is associated with a partition. This means that either
   * \a left_child_index is valid, or \a left_partition_id
   * (and the same for the right side).
   *
   * The partition calculation is done by calling doPartition().
   * After calculation, it is possible to retrieve the array of
   * nodes via the tree() method. The nodes are stored in an
   * array and can therefore be indexed directly.
   */
  struct TreeNode
  {
    //! Linear index in the tree
    Int32 index = -1;
    //! Linear index of the left child (-1 if none)
    Int32 left_child_index = -1;
    //! Linear index of the right child (-1 if none)
    Int32 right_child_index = -1;
    //! Index in the tree of the parent (-1 if none)
    Int32 parent_index = -1;
    //! Number of children on the left (if not terminal)
    Int32 nb_left_child = -1;
    //! Number of children on the right (if not terminal)
    Int32 nb_right_child = -1;
    //! Level in the tree
    Int32 level = -1;
    //! Index of the left partition (only for terminal nodes)
    Int32 left_partition_id = -1;
    //! Index of the right partition (only for terminal nodes)
    Int32 right_partition_id = -1;

   public:

    friend std::ostream& operator<<(std::ostream& o, const TreeNode& t)
    {
      o << "(index=" << t.index << " level=" << t.level
        << " parent=" << t.parent_index << " sum_left=" << t.nb_left_child
        << " sum_right=" << t.nb_right_child << " left=" << t.left_child_index << " right="
        << t.right_child_index
        << " left_id=" << t.left_partition_id << " right_id=" << t.right_partition_id
        << ")";
      return o;
    }
  };

 public:

  explicit BinaryTree(ITraceMng* tm)
  : TraceAccessor(tm)
  {}

 public:

  void doPartition(Int32 nb_part)
  {
    m_tree_info.resize(nb_part);
    m_nb_part = nb_part;
    Int32 sum = 0;
    Int32 level = 0;
    Int32 parent_index = -1;
    Int32 part_id = 0;
    _doRecursivePart(0, part_id, sum, parent_index, level);
    for (const TreeNode& t : m_tree_info) {
      info() << t;
    }
  }

  //! List of tree nodes
  ConstArrayView<TreeNode> tree() const { return m_tree_info; }

 private:

  UniqueArray<TreeNode> m_tree_info;
  Int32 m_nb_part = 0;

  void _doRecursivePart(Int32 partition_index, Int32& part_id, Int32& nb_child, Int32 parent_index, Int32 level)
  {
    Int32 part0_partition_index = partition_index;
    Int32 part1_partition_index = partition_index + 1;

    Int32 nb_child_left = 0;
    Int32 nb_child_right = 0;

    Int32 next_left = (2 * partition_index) + 1;
    Int32 next_right = (2 * partition_index) + 2;

    m_tree_info[part0_partition_index].parent_index = parent_index;

    if ((next_left + 1) < m_nb_part) {
      m_tree_info[part0_partition_index].left_child_index = next_left;
      _doRecursivePart(next_left, part_id, nb_child_left,
                       part0_partition_index, level + 1);
    }
    else {
      info() << "DO_PART LEFT parent=" << parent_index << " ID=" << part_id
             << " nb_child=" << nb_child << " nb_child_left=" << nb_child_left;
      m_tree_info[part0_partition_index].left_partition_id = part_id;
      ++part_id;
      ++nb_child_left;
    }
    if ((next_right + 1) < m_nb_part) {
      m_tree_info[part0_partition_index].right_child_index = next_right;
      _doRecursivePart(next_right, part_id, nb_child_right,
                       part1_partition_index, level + 1);
    }
    else {
      info() << "DO_PART RIGHT parent=" << parent_index << " ID=" << part_id
             << " nb_child" << nb_child << " nb_child_right=" << nb_child_right;
      m_tree_info[part0_partition_index].right_partition_id = part_id;
      ++part_id;
      ++nb_child_right;
    }
    nb_child += (nb_child_left + nb_child_right);
    info() << "End for me parent=" << parent_index
           << " part0_index=" << part0_partition_index
           << " nb_child_left=" << nb_child_left << " nb_child_right=" << nb_child_right
           << " level=" << level;
    m_tree_info[part0_partition_index].nb_left_child = nb_child_left;
    m_tree_info[part0_partition_index].nb_right_child = nb_child_right;
    m_tree_info[part0_partition_index].level = level;
    m_tree_info[part0_partition_index].index = part0_partition_index;
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Mesh geometric partitioning service.
 *
 * This service allows performing a geometric partitioning.
 * Its primary goal is to allow parallel tests even if
 * no specific partitioning service (like ParMetis) is
 * available.
 *
 * The algorithm is as follows:
 *
 * 1. The partitioning is recursive, and at each iteration, a partition is split into two.
 * 2. To split a partition, the center of gravity of
 * the partition, as well as the inertia tensor matrix, is calculated.
 * 3. From this matrix, the eigenvector corresponding to the smallest
 * axis (the one corresponding to the smallest eigenvalue) is taken.
 * 4. The center of gravity and the eigenvector define a plane (or a
 * line in 2D) which serves as a partition: elements on one side of the plane
 * are in one partition and those on the other side are in another.
 * 5. This mechanism is applied to cut into the desired number of partitions.
 *
 * The current algorithm and implementation are very simple and have the following limitations:
 *
 * - Element weights are not taken into account when calculating the partitioning
 * - It is assumed that the number of partitions is a power of 2. If not,
 * some partitions will be larger than others. For example,
 * if we want to split into 3, we first create two identical partitions 1 and 2,
 * then split partition 2 into two parts. Partition 1 will
 * therefore theoretically be twice as large as partitions 2 and 3.
 * - The partition is done according to a plane, which is probably not optimal
 * for limiting the boundary. Using a circle would be more sensible.
 * - The implementation is linear in the number of partitions and all ranks
 * participate. Normally, if we want P partitions, then the number of recursions
 * is N=log2(P) and it would be possible to perform all partitions
 * of rank 1 to N simultaneously.
 *
 * Currently, the algorithm used applies to meshes, but only
 * coordinates are used, which would allow it to be applied without
 * mesh elements. This could be useful for an initial partitioning
 * such as the one used in the parallel MSH reader.
 */
class ArcaneGeometricMeshPartitionerService
: public ArcaneArcaneGeometricMeshPartitionerServiceObject
, public IMeshPartitioner
{
 public:

  explicit ArcaneGeometricMeshPartitionerService(const ServiceBuildInfo& sbi);

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
  Int32 m_nb_part = 0;

 private:

  Real3 _computeBarycenter(const VariableCellReal3& cells_center, CellVectorView cells);
  Real3x3 _computeInertiaTensor(Real3 center, const VariableCellReal3& cells_center,
                                CellVectorView cells);
  Real3 _findPrincipalAxis(Real3x3 tensor);
  void _partitionMesh2();
  void _partitionMesh(bool initial_partition, Int32 nb_part);
  bool _partitionMeshRecursive(const VariableCellReal3& cells_center,
                               CellVectorView cells, Int32 partition_index, Int32& part_id);
  void _partitionMeshRecursive2(ConstArrayView<BinaryTree::TreeNode> tree_nodes, const VariableCellReal3& cells_center,
                                CellVectorView cells, Int32 partition_index);
  void _printOwners();
  Real3 _computeEigenValuesAndVectors(ITraceMng* tm, Real3x3 tensor, Real3x3& eigen_vectors, Real3& eigen_values);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ArcaneGeometricMeshPartitionerService::
ArcaneGeometricMeshPartitionerService(const ServiceBuildInfo& sbi)
: ArcaneArcaneGeometricMeshPartitionerServiceObject(sbi)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Function to calculate the inertia tensor of the mesh
Real3 ArcaneGeometricMeshPartitionerService::
_computeBarycenter(const VariableCellReal3& cells_center, CellVectorView cells)
{
  Real3 center;
  IParallelMng* pm = mesh()->parallelMng();
  ENUMERATE_ (Cell, icell, cells) {
    center += cells_center[icell];
  }
  Int64 local_nb_cell = cells.size();
  Int64 total_nb_cell = pm->reduce(Parallel::ReduceSum, local_nb_cell);
  if (total_nb_cell == 0)
    return {};
  Real3 sum_center = pm->reduce(Parallel::ReduceSum, center);
  Real3 global_center = sum_center / static_cast<Real>(total_nb_cell);
  return global_center;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*
 * \brief Calculates the inertia tensor of the mesh.
 */
Real3x3 ArcaneGeometricMeshPartitionerService::
_computeInertiaTensor(Real3 center, const VariableCellReal3& cells_center, CellVectorView cells)
{
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

  sum_tensor[1][0] = sum_tensor[0][1];
  sum_tensor[2][0] = sum_tensor[0][2];
  sum_tensor[2][1] = sum_tensor[1][2];

  return sum_tensor;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*
 * \brief Finds the principal inertia axis of the mesh.
 */
Real3 ArcaneGeometricMeshPartitionerService::
_findPrincipalAxis(Real3x3 tensor)
{
  info() << "Tensor=" << tensor;

  EigenValueAndVectorComputer eigen_computer;
  eigen_computer.computeForMatrix(tensor);
  info() << "EigenValues  = " << eigen_computer.eigenValues();
  info() << "EigenVectors = " << eigen_computer.eigenVectors();
  Real3x3 eigen_vectors = eigen_computer.eigenVectors();
  Real3 v = eigen_vectors[0];
  // If the smallest eigenvector is zero, take the next one
  // (generally this does not happen unless the algorithm in
  // 'computeForMatrix' did not converge).
  if (math::isNearlyZero(v.normL2())) {
    v = eigen_vectors[1];
    if (math::isNearlyZero(v.normL2()))
      v = eigen_vectors[2];
  }
  info() << "EigenVector=" << v;
  return v;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Function to partition the mesh into two sub-domains
void ArcaneGeometricMeshPartitionerService::
_partitionMesh2()
{
  // Calculates the center of the cells
  VariableCellReal3 cells_center(VariableBuildInfo(mesh(), "ArcaneCellCenter"));
  IItemFamily* cell_family = mesh()->cellFamily();
  CellVector cells_vector(cell_family, cell_family->allItems().own().view().localIds());
  CellVectorView cells = cells_vector.view();

  info() << "** ** DO_PARTITION_MESH2 nb_cell=" << cells.size();

  // Calculates the center of each cell
  {
    const VariableNodeReal3& nodes_coordinates = mesh()->nodesCoordinates();
    ENUMERATE_ (Cell, icell, cells) {
      Real3 c;
      for (NodeLocalId n : icell->nodeIds())
        c += nodes_coordinates[n];
      cells_center[icell] = c / icell->nbNode();
    }
  }

  BinaryTree binary_tree(traceMng());
  binary_tree.doPartition(m_nb_part);

  bool do_new = true;
  if (platform::getEnvironmentVariable("GEOMETRIC_PARTITIONER_IMPL") == "1")
    do_new = false;
  info() << "Using geoemtric partitioner do_new?=" << do_new;
  if (do_new) {
    _partitionMeshRecursive2(binary_tree.tree(), cells_center, cells, 0);
  }
  else {
    Int32 partition_index = 0;
    Int32 part_id = 0;
    _partitionMeshRecursive(cells_center, cells, partition_index, part_id);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool ArcaneGeometricMeshPartitionerService::
_partitionMeshRecursive(const VariableCellReal3& cells_center,
                        CellVectorView cells, Int32 partition_index, Int32& part_id)
{
  Int32 part0_partition_index = partition_index;
  Int32 part1_partition_index = partition_index + 1;
  // For testing. To be removed.
  Int32 total_nb_cell = mesh()->parallelMng()->reduce(Parallel::ReduceSum, cells.size());
  info() << "Doing partition partition_index=" << partition_index << " total_nb_cell=" << total_nb_cell;
  if (part1_partition_index >= m_nb_part)
    return true;

  info() << "Doing partition really partition_index=" << partition_index;
  IItemFamily* cell_family = mesh()->cellFamily();
  VariableItemInt32& cells_new_owner = cell_family->itemsNewOwner();

  // Calculate the center of mass
  Real3 center = _computeBarycenter(cells_center, cells);
  info() << "GlobalCenter=" << center;

  // Calculate the inertia tensor
  Real3x3 tensor = _computeInertiaTensor(center, cells_center, cells);

  // Find the principal inertia axis
  Real3 eigenvector = _findPrincipalAxis(tensor);
  info() << "EigenVector=" << eigenvector;

  const Int32 nb_cell = cells.size();
  UniqueArray<Int32> part0_cells;
  part0_cells.reserve(nb_cell);
  UniqueArray<Int32> part1_cells;
  part1_cells.reserve(nb_cell);

  // Checks which part the cell will belong to
  // by calculating the dot product between the eigenvector
  // and the vector from the center of gravity to the cell.
  // The sign value indicates which part it is in.
  info() << "Doing partition setting nb_cell=" << nb_cell << " partition=" << part0_partition_index << " " << part1_partition_index;
  ENUMERATE_ (Cell, icell, cells) {
    const Real3 cell_coord = cells_center[icell];

    Real projection = 0.0;
    projection += (cell_coord.x - center.x) * eigenvector.x;
    projection += (cell_coord.y - center.y) * eigenvector.y;
    projection += (cell_coord.z - center.z) * eigenvector.z;

    if (projection < 0.0) {
      part0_cells.add(icell.itemLocalId());
    }
    else {
      part1_cells.add(icell.itemLocalId());
    }
  }
  CellVectorView part0(cell_family, part0_cells);
  CellVectorView part1(cell_family, part1_cells);

  // If _partitionMeshRecursive() returns true, then there are no more sub-partitions
  // to perform. In this case, we fill the owners of the cells
  // for the partition.
  if (_partitionMeshRecursive(cells_center, part0, (2 * partition_index) + 1, part_id)) {
    info() << "Filling left part part_index=" << part_id << " nb_cell=" << part0.size();
    ENUMERATE_ (Cell, icell, part0) {
      cells_new_owner[icell] = part_id;
    }
    ++part_id;
  }
  if (_partitionMeshRecursive(cells_center, part1, (2 * partition_index) + 2, part_id)) {
    info() << "Filling right part part_index=" << part_id << " nb_cell=" << part1.size();
    ENUMERATE_ (Cell, icell, part1) {
      cells_new_owner[icell] = part_id;
    }
    ++part_id;
  }
  return false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcaneGeometricMeshPartitionerService::
_partitionMeshRecursive2(ConstArrayView<BinaryTree::TreeNode> tree_nodes,
                         const VariableCellReal3& cells_center,
                         CellVectorView cells, Int32 partition_index)
{
  Int32 part0_partition_index = partition_index;
  Int32 part1_partition_index = partition_index + 1;
  IParallelMng* pm = mesh()->parallelMng();
  // For testing. To be removed later
  Int32 total_nb_cell = pm->reduce(Parallel::ReduceSum, cells.size());
  info() << "Doing partition (V2) partition_index=" << partition_index << " total_nb_cell=" << total_nb_cell;

  info() << "Doing partition (V2) really partition_index=" << partition_index;
  IItemFamily* cell_family = mesh()->cellFamily();
  VariableItemInt32& cells_new_owner = cell_family->itemsNewOwner();

  // Calculate the center of mass
  Real3 center = _computeBarycenter(cells_center, cells);
  info() << "GlobalCenter=" << center;

  // Calculate the inertia tensor
  Real3x3 tensor = _computeInertiaTensor(center, cells_center, cells);

  // Find the principal inertia axis
  Real3 eigenvector = _findPrincipalAxis(tensor);
  info() << "EigenVector=" << eigenvector;

  const Int32 nb_cell = cells.size();

  // Calculates for each element the dot product between
  // the eigenvector and the vector connecting the center of gravity
  // to this element and stores it in projections
  info() << "Doing partition (V2) nb_cell=" << nb_cell << " setting partition=" << part0_partition_index << " " << part1_partition_index;
  UniqueArray<Real> projections(nb_cell);
  Real min_projection = std::numeric_limits<Real>::max();
  Real max_projection = std::numeric_limits<Real>::min();
  ENUMERATE_ (Cell, icell, cells) {
    const Real3 cell_coord = cells_center[icell];
    Real projection = math::dot(cell_coord - center, eigenvector);
    projections[icell.index()] = projection;
    min_projection = math::min(min_projection, projection);
    max_projection = math::max(max_projection, projection);
  }

  // Globally calculates the min and max of the projection.
  min_projection = pm->reduce(Parallel::ReduceMin, min_projection);
  max_projection = pm->reduce(Parallel::ReduceMax, max_projection);

  info() << "min_projection=" << min_projection << " max_projection=" << max_projection;

  // To account for the ratio between the two partitions which is not necessarily 0.5,
  // we determine several projection values that will be used to partition
  // These tested values are in the interval [-min_projection, max_projection].
  // We test nb_to_test values between '-min_projection' and 0, the value 0.0 and nb_to_test
  // between 0 and max_projection. We will therefore test (2*nb_to_test)+1 values.

  UniqueArray<Real> projections_to_test;
  // TODO: make nb_to_test parameterizable.
  // We could also perform a dichotomy on the projection values
  // to better approximate the balance and not necessarily test too many values
  int nb_to_test = 10;
  const int total_nb_to_test = (2 * nb_to_test) + 1;
  projections_to_test.resize(total_nb_to_test);
  for (Int32 i = 0; i < nb_to_test; ++i) {
    Real v1 = min_projection / (i + 1);
    projections_to_test[i] = v1;
    Real v2 = max_projection / (nb_to_test - i);
    projections_to_test[i + 1 + nb_to_test] = v2;
  }
  projections_to_test[nb_to_test] = 0.0; //< Central projection
  info() << "projections_to_test=" << projections_to_test;

  // The binary tree allows knowing how many partitions need to be made
  // on the left and right side. We use it to calculate a ratio
  // ideal which is stored in expected_ratio.
  BinaryTree::TreeNode current_tree_node = tree_nodes[partition_index];
  Int32 nb_left_child = current_tree_node.nb_left_child;
  Int32 nb_right_child = current_tree_node.nb_right_child;
  // Desired ratio of elements between the two parts
  Real expected_ratio = 1.0;
  if (nb_left_child != 0) {
    Real r_nb_left_child = static_cast<Real>(nb_left_child);
    Real r_nb_right_child = static_cast<Real>(nb_right_child);
    expected_ratio = r_nb_left_child / (r_nb_left_child + r_nb_right_child);
  }

  // This array will contain, in the form of a pair, the number of elements
  // of each partition for each test.
  // global_nb_parts[i*2+p] is the value of the left partition (p==0)
  // or right (p==1) for the i-th test.
  SmallArray<Int64> global_nb_parts(total_nb_to_test*2);

  // Tests all partitions and calculates the one whose ratio is the
  // closest to the desired ratio. This is the one we will take for
  // partitioning
  Int32 wanted_projection_index = 0;
  Real best_partition_ratio = std::numeric_limits<Real>::max();
  for (Int32 z = 0; z < total_nb_to_test; ++z) {
    Int32 nb_new_part0 = 0;
    Int32 nb_new_part1 = 0;
    const Real projection_to_test = projections_to_test[z];
    // TODO: Move this loop outside
    ENUMERATE_ (Cell, icell, cells) {
      Real projection = projections[icell.index()];
      if (projection < projection_to_test)
        ++nb_new_part0;
      else
        ++nb_new_part1;
    }
    global_nb_parts[0+(z*2)] = nb_new_part0;
    global_nb_parts[1+(z*2)] = nb_new_part1;
  }

  // Sums the parts across all sub-domains.
  pm->reduce(Parallel::ReduceSum, global_nb_parts.view());

  for (Int32 z = 0; z < total_nb_to_test; ++z) {
    Real ratio_0 = 1.0;
    Int64 nb_part0 = global_nb_parts[0+(z*2)];
    Int64 nb_part1 = global_nb_parts[1+(z*2)];
    if (nb_part0 != 0) {
      Real r_nb_part0 = static_cast<Real>(nb_part0);
      Real r_nb_part1 = static_cast<Real>(nb_part1);
      ratio_0 = r_nb_part0 / (r_nb_part0 + r_nb_part1);
    }
    Real diff_ratio = math::abs(expected_ratio - ratio_0);
    info(4) << "Partition info nb_part0=" << nb_part0 << " nb_part1=" << nb_part1
            << " ratio_0=" << ratio_0
            << " nb_left_child=" << nb_left_child << " nb_right_child=" << nb_right_child
            << " expected_ratio=" << expected_ratio
            << " diff_ratio=" << diff_ratio
            << " best_ratio=" << best_partition_ratio;
    if (diff_ratio < best_partition_ratio) {
      wanted_projection_index = z;
      best_partition_ratio = diff_ratio;
    }
  }

  const Real projection_to_use = projections_to_test[wanted_projection_index];
  info() << "Keep projection index=" << wanted_projection_index << " projection=" << projection_to_use
         << " best_ratio=" << best_partition_ratio;

  UniqueArray<Int32> part0_cells;
  part0_cells.reserve(nb_cell);
  UniqueArray<Int32> part1_cells;
  part1_cells.reserve(nb_cell);

  // Checks which part the cell will belong to
  // by calculating the dot product between the eigenvector
  // and the vector from the center of gravity to the cell.
  // The sign value indicates which part it is in.

  ENUMERATE_ (Cell, icell, cells) {
    Real projection = projections[icell.index()];
    if (projection < projection_to_use) {
      part0_cells.add(icell.itemLocalId());
    }
    else {
      part1_cells.add(icell.itemLocalId());
    }
  }

  CellVectorView part0(cell_family, part0_cells);
  CellVectorView part1(cell_family, part1_cells);

  Int32 child_left = current_tree_node.left_child_index;
  if (child_left >= 0) {
    _partitionMeshRecursive2(tree_nodes, cells_center, part0, child_left);
  }
  Int32 left_partition_id = current_tree_node.left_partition_id;
  if (left_partition_id >= 0) {
    info() << "Filling left part part_index=" << left_partition_id << " nb_cell=" << part0.size();
    ENUMERATE_ (Cell, icell, part0) {
      cells_new_owner[icell] = left_partition_id;
    }
  }
  Int32 child_right = current_tree_node.right_child_index;
  if (child_right >= 0) {
    _partitionMeshRecursive2(tree_nodes, cells_center, part1, child_right);
  }

  Int32 right_partition_id = current_tree_node.right_partition_id;
  if (right_partition_id >= 0) {
    info() << "Filling right part part_index=" << right_partition_id << " nb_cell=" << part1.size();
    ENUMERATE_ (Cell, icell, part1) {
      cells_new_owner[icell] = right_partition_id;
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcaneGeometricMeshPartitionerService::
_partitionMesh([[maybe_unused]] bool initial_partition, Int32 nb_part)
{
  m_nb_part = nb_part;

  IPrimaryMesh* mesh = this->mesh()->toPrimaryMesh();

  info() << "Doing mesh partition with ArcaneGeometricMeshPartitionerService nb_part=" << nb_part;
  IParallelMng* pm = mesh->parallelMng();
  Int32 nb_rank = pm->commSize();

  if (nb_rank == 1) {
    return;
  }

  _partitionMesh2();
  _printOwners();
  VariableItemInt32& cells_new_owner = mesh->itemsNewOwner(IK_Cell);
  cells_new_owner.synchronize();
  if (mesh->partitionConstraintMng()) {
    // Deal with Tied Cells
    mesh->partitionConstraintMng()->computeAndApplyConstraints();
  }
  mesh->utilities()->changeOwnersFromCells();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcaneGeometricMeshPartitionerService::
_printOwners()
{
  IParallelMng* pm = mesh()->parallelMng();
  VariableItemInt32& cells_new_owner = mesh()->toPrimaryMesh()->itemsNewOwner(IK_Cell);
  for (Int32 i = 0; i < m_nb_part; ++i) {
    Int32 n = 0;
    ENUMERATE_ (Cell, icell, mesh()->ownCells()) {
      if (cells_new_owner[icell] == i)
        ++n;
    }
    info() << "NbCell for part=" << i << " total=" << pm->reduce(Parallel::ReduceSum, n);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcaneGeometricMeshPartitionerService::
partitionMesh(bool initial_partition)
{
  _partitionMesh(initial_partition, mesh()->parallelMng()->commSize());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#if ARCANE_DEFAULT_PARTITIONER == ARCANEGEOMETRICMESHPARTITIONER_DEFAULT_PARTITIONER
ARCANE_REGISTER_SERVICE_ARCANEGEOMETRICMESHPARTITIONERSERVICE(DefaultPartitioner,
                                                              ArcaneGeometricMeshPartitionerService);
#endif
ARCANE_REGISTER_SERVICE_ARCANEGEOMETRICMESHPARTITIONERSERVICE(ArcaneGeometricMeshPartitioner,
                                                              ArcaneGeometricMeshPartitionerService);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
