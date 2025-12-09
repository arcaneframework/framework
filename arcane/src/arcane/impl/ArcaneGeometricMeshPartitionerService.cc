// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ArcaneGeometricMeshPartitionerService.cc                    (C) 2000-2025 */
/*                                                                           */
/* Service de partitionnement géométrique de maillage.                       */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/NotImplementedException.h"
#include "arcane/utils/ITraceMng.h"

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

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe pour calculer les valeurs et vecteurs propres d'une matrice.
 *
 * Le calcul se fait par la méthode des puissances.
 * Ce n'est pas la méthode la plus rapide ni la plus robuste au niveau
 * numérique, mais on n'a pas besoin d'une précision importante dans
 * le calcul des valeurs et vecteurs propres. Si besoin, on pourrait utiliser
 * l'algorithme de Jacboi ou une méthode QR.
 *
 * Les valeurs propres sont triées par ordre croissant.
 * \note Comme l'algorithme est itératif, il peut arriver que l'ordre ne soit
 * pas strictement croissant si deux valeurs propres sont proches.
 */
class EigenValueAndVectorComputer
{
 private:

  //! Résultat de l'application de la méthode de la puissance
  struct PowerResult
  {
    Real eigen_value;
    Real3 eigen_vector;
  };

 public:

  /*!
   * \brief Calcule les valeurs et vecteurs propres de \a orig_matrix.
   */
  void computeForMatrix(const Real3x3& orig_matrix)
  {
    Real3x3 matrix = orig_matrix;

    constexpr int nb_value = 3;
    // Itère pour calculer les valeurs et vecteurs propres
    // La méthode de la puissance calcule la valeur propre
    // la plus élevée. Comme on veut trier les valeurs propres
    // par ordre croissant, on les range dans l'ordre inverse.

    for (int i = (nb_value - 1); i >= 0; --i) {
      // Calculer le premier vecteur propre (le plus grand)
      PowerResult result = _applyPowerIteration(matrix);
      m_eigen_values[i] = result.eigen_value;
      m_eigen_vectors[i] = result.eigen_vector;

      // Appliquer la déflation pour éliminer le vecteur propre
      // qu'on vient de calculer.
      // On n'a pas besoin de le faire pour la dernière itération
      if (i != 0)
        _deflateMatrix(matrix, result.eigen_value, result.eigen_vector);
    }
  }
  //! Retourne les valeurs propres de la matrice par ordre croissant
  Real3 eigenValues() const { return m_eigen_values; }
  //! Retourne les vecteurs propres de la matrice par ordre croissant
  Real3x3 eigenVectors() const { return m_eigen_vectors; }

 private:

  // Soustrais un vecteur propre de la matrice pour la déflation
  static void _deflateMatrix(Real3x3& matrix, double eigenvalue, Real3 eigenvector)
  {
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        matrix[i][j] -= eigenvalue * eigenvector[i] * eigenvector[j];
      }
    }
  }

  // Applique la méthode des puissances
  static PowerResult _applyPowerIteration(const Real3x3& matrix)
  {
    constexpr Int32 max_iteration = 1000;
    constexpr Real tolerance = 1e-16;
    Real eigenvalue = 0.0;

    // Initialisation avec un vecteur initial (il pourrait être aléatoire)
    Real3 b{ 1.0, 1.0, 1.0 };
    Real3 b_next;

    // Itérations de la méthode des puissances
    for (Int32 iter = 0; iter < max_iteration; ++iter) {
      b_next = math::multiply(matrix, b);

      eigenvalue = b_next.normL2();
      if (math::isNearlyZero(eigenvalue))
        break;
      b_next = b_next / eigenvalue;

      // Vérifie la convergence
      Real diff = math::squareNormL2(b_next - b);
      if (diff < tolerance) {
        break; // Si la différence est suffisamment petite, on a convergé
      }

      b = b_next;
    }

    return { eigenvalue, b_next };
  }

 private:

  //! Valeurs propres
  Real3 m_eigen_values;

  //! Vecteurs propres
  Real3x3 m_eigen_vectors;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe pour créer un arbre binaire.
 */
class BinaryTree
: public TraceAccessor
{
 public:

  /*!
   * \brief Information sur un nœud de l'arbre.
   *
   * Un nœud a 0, 1 ou 2 enfants et 0 ou 1 parent.
   * Si un nœud n'a pas d'enfant d'un côté, alors il
   * est associé à une partition. Cela signifie que soit
   * \a left_child_index est valide, soit \a left_partition_id
   * (et de même pour le côté droit).
   *
   * Le calcul de la partition se fait par l'appel à doPartition().
   * Après le calcul, il est possible de récupérer le tableau des
   * noeuds via la méthode tree(). Les noeuds sont rangés dans un
   * tableau et peuvent donc être indéxés directement.
   */
  struct TreeNode
  {
    //! Index linéaire dans l'arbre
    Int32 index = -1;
    //! Index linéaire du fils de gauche (-1 si aucune)
    Int32 left_child_index = -1;
    //! Index linéaire du fils de droite (-1 si aucune)
    Int32 right_child_index = -1;
    //! Index dans l'arbre du parent (-1 si aucun)
    Int32 parent_index = -1;
    //! Nombre d'enfants à gauche (si non terminal)
    Int32 nb_left_child = -1;
    //! Nombre d'enfants à droite (si non terminal)
    Int32 nb_right_child = -1;
    //! Niveau dans l'arbre
    Int32 level = -1;
    //! Indice de la partition de gauche (uniquement pour les noeuds terminaux)
    Int32 left_partition_id = -1;
    //! Indice de la partition de droite (uniquement pour les noeuds terminaux)
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

  //! Liste des noeuds de l'arbre
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
 * \brief Service de partitionnement géométrique de maillage.
 *
 * Ce service permet d'effectuer un partitionnement géométrique.
 * Son but premier est de permettre de réaliser des tests parallèles même si
 * aucun service de partitionnement spécifique (comme ParMetis) n'est
 * disponible.
 *
 * L'algorithme est le suivant :
 *
 * 1. Le partitionnement est récursif et à chaque itération, on coupe
 *    une partition deux.
 * 2. Pour couper une partition, on calcule le centre de gravité de
 *    la partition ainsi que la matrice de son moment d'inertie.
 * 3. À partir de cette matrice prend le vecteur propre du plus petit
 *    axe (celui qui correspond à la plus petite valeur propre).
 * 4. Le centre de gravité et le vecteur propre définissent un plan (ou une
 *    droite en 2D) qui sert de partition : les éléments d'un côté du plan
 *    sont dans une partition et ceux de l'autre côté sont dans une autre.
 * 5. On applique ce mécanisme pour découper en le nombre de partitions souhaité.
 *
 * L'algorithme et l'implémentation actuels sont très simples et ont les limitions
 * suivantes :
 *
 * - On ne tient pas compte des poids aux éléments pour calculer le partitionnement
 * - On suppose que le nombre de partitions est une puissance de 2. Si ce n'est
 *   pas le cas, certaines partitions seront plus grosses que d'autres. Par exemple
 *   si on veut découper en 3, on va d'abord créer deux partitions 1 et 2 identiques
 *   puis découper la partition 2 en deux parties. La partition 1 sera
 *   donc en théorie deux fois plus grosses que les partitions 2 et 3.
 * - La partition se fait selon un plan ce qui n'est probablement pas optimal
 *   pour limiter la frontière. Utiliser un cercle serait plus judicieux.
 * - L'implémentation est linéaire en nombre de partitions et tous les rangs
 *   participent. Normalement, si on veut P partition, alors le nombre de récursions
 *   est N=log2(P) et il serait possible de réaliser simultanément toutes les partitions
 *   de rang 1 à N.
 *
 * Actuellement l'algorithme utilisé s'applique aux mailles, mais seules
 * les coordonnées sont utilisées ce qui permettrait de l'appliquer sans
 * élément de maillage. Cela pourrait être utile pour un premier partitionnement
 * tel que celui utilisé dans le lecteur parallèle MSH.
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

// Fonction pour calculer le moment d'inertie du maillage
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
 * \brief Calcule le moment d'inertie du maillage.
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
 * \brief Trouve l'axe d'inertie principal du maillage.
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
  // Si le plus petit vecteur propre est nul, prend le suivant
  // (en général cela n'arrive pas sauf si l'algorithme dans
  // 'computeForMatrix' n'a pas convergé).
  if (math::isNearlyZero(v.normL2())){
    v = eigen_vectors[1];
    if (math::isNearlyZero(v.normL2()))
      v = eigen_vectors[2];
  }
  info() << "EigenVector=" << v;
  return v;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Fonction pour partitionner le maillage en deux sous-domaines
void ArcaneGeometricMeshPartitionerService::
_partitionMesh2()
{
  // Calcule le centre des mailles
  VariableCellReal3 cells_center(VariableBuildInfo(mesh(), "ArcaneCellCenter"));
  IItemFamily* cell_family = mesh()->cellFamily();
  CellVector cells_vector(cell_family, cell_family->allItems().own().view().localIds());
  CellVectorView cells = cells_vector.view();

  info() << "** ** DO_PARTITION_MESH2 nb_cell=" << cells.size();

  // Calcule le centre de chaque maille
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
  // Pour test. A supprimer.
  Int32 total_nb_cell = mesh()->parallelMng()->reduce(Parallel::ReduceSum, cells.size());
  info() << "Doing partition partition_index=" << partition_index << " total_nb_cell=" << total_nb_cell;
  if (part1_partition_index >= m_nb_part)
    return true;

  info() << "Doing partition really partition_index=" << partition_index;
  IItemFamily* cell_family = mesh()->cellFamily();
  VariableItemInt32& cells_new_owner = cell_family->itemsNewOwner();

  // Calculer le centre de masse
  Real3 center = _computeBarycenter(cells_center, cells);
  info() << "GlobalCenter=" << center;

  // Calculer le tenseur d'inertie
  Real3x3 tensor = _computeInertiaTensor(center, cells_center, cells);

  // Trouver l'axe principal d'inertie
  Real3 eigenvector = _findPrincipalAxis(tensor);
  info() << "EigenVector=" << eigenvector;

  const Int32 nb_cell = cells.size();
  UniqueArray<Int32> part0_cells;
  part0_cells.reserve(nb_cell);
  UniqueArray<Int32> part1_cells;
  part1_cells.reserve(nb_cell);

  // Regarde dans quelle partie va se trouver la maille
  // en calculant le produit scalaire entre le vecteur propre
  // et le vecteur du centre de gravité à la maille.
  // La valeur du signe indique dans quelle partie on se trouve.
  info() << "Doing partition setting partition=" << part0_partition_index << " " << part1_partition_index;
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

  // Si _partitionMeshRecursive() retourne \a true, alors il n'y a plus de sous-partition
  // à réaliser. Dans ce cas, on remplit les propriétaires des mailles
  // pour la partition.
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
  // Pour test. À supprimer par la suite
  Int32 total_nb_cell = mesh()->parallelMng()->reduce(Parallel::ReduceSum, cells.size());
  info() << "Doing partition (V2) partition_index=" << partition_index << " total_nb_cell=" << total_nb_cell;

  info() << "Doing partition (V2) really partition_index=" << partition_index;
  IItemFamily* cell_family = mesh()->cellFamily();
  VariableItemInt32& cells_new_owner = cell_family->itemsNewOwner();

  // Calculer le centre de masse
  Real3 center = _computeBarycenter(cells_center, cells);
  info() << "GlobalCenter=" << center;

  // Calculer le tenseur d'inertie
  Real3x3 tensor = _computeInertiaTensor(center, cells_center, cells);

  // Trouver l'axe principal d'inertie
  Real3 eigenvector = _findPrincipalAxis(tensor);
  info() << "EigenVector=" << eigenvector;

  const Int32 nb_cell = cells.size();
  UniqueArray<Int32> part0_cells;
  part0_cells.reserve(nb_cell);
  UniqueArray<Int32> part1_cells;
  part1_cells.reserve(nb_cell);

  // Regarde dans quelle partie va se trouver la maille
  // en calculant le produit scalaire entre le vecteur propre
  // et le vecteur du centre de gravité à la maille.
  // La valeur du signe indique dans quelle partie on se trouve.
  info() << "Doing partition setting partition=" << part0_partition_index << " " << part1_partition_index;
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

  BinaryTree::TreeNode current_tree_node = tree_nodes[partition_index];
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
