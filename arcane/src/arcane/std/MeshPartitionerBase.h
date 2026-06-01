// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshPartitionerBase.h                                       (C) 2000-2025 */
/*                                                                           */
/* Base class for a mesh partitioner.                                        */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_STD_MESHPARTITIONERBASE_H
#define ARCANE_STD_MESHPARTITIONERBASE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Array.h"
#include "arcane/utils/HashTableMap.h"
#include "arcane/utils/ScopedPtr.h"

#include "arcane/core/IMeshPartitioner.h"
#include "arcane/core/AbstractService.h"
#include "arcane/core/ILoadBalanceMng.h"

#include <set>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup StandardService
 * \brief Base class for a load balancing service.
 */
class ARCANE_STD_EXPORT MeshPartitionerBase
: public AbstractService
, public IMeshPartitioner
{
 public:

  explicit MeshPartitionerBase(const ServiceBuildInfo& sbi);
  ~MeshPartitionerBase() override;

 public:

  ISubDomain* subDomain() const { return m_sub_domain; }
  IMesh* mesh() const override { return m_mesh; }

  // DEPRECATED
  void setMaximumComputationTime(Real v) override { m_maximum_computation_time = v; }
  Real maximumComputationTime() const override { return m_maximum_computation_time; }

  void setImbalance(Real v) override { m_imbalance = v; }
  Real imbalance() const override { return m_imbalance; }

  void setMaxImbalance(Real v) override { m_max_imbalance = v; }
  Real maxImbalance() const override { return m_max_imbalance; }

  void setComputationTimes(RealConstArrayView v) override { m_computation_times.copy(v); }
  RealConstArrayView computationTimes() const override { return m_computation_times; }

  void setCellsWeight(ArrayView<float> weights, Integer nb_weight) override;
  ArrayView<float> cellsWeight() const override;

  // CORRECT
  Integer nbCellWeight() const;
  void setILoadBalanceMng(ILoadBalanceMng* mng) override { m_lbMng = mng; }
  ILoadBalanceMng* loadBalanceMng() const override { return m_lbMng; }

 public:

  void notifyEndPartition() override { loadBalanceMng()->notifyEndPartition(); }

 public:

  /*! \brief Positions the new owners of nodes, edges
   * and faces based on the meshes.
   *
   * Assuming the new owners of the meshes are known (and synchronized),
   * determines the new owners of the other
   * entities.
   */
  virtual void changeOwnersFromCells();

  enum eMarkCellWithConstraint
  {
    eCellClassical,
    eCellReference,
    eCellGrouped,
    eCellGhost,
    eCellInAConstraint
  };

  /* \brief Initializes structures for the case with partitioning constraints */
  virtual void initConstraints(bool uidref = true);

  /* \brief Frees temporary arrays */
  virtual void freeConstraints();

  /* \brief Returns the number of Cells after grouping according to constraints, internal */
  virtual Int32 nbOwnCellsWithConstraints() const;

  /* \brief Provides the list of unique IDs of neighboring Cells of a cell, taking constraints into account */
  virtual Real getNeighbourCellsUidWithConstraints(Cell cell, Int64Array& neighbourcells, Array<float>* commWeights = NULL,
                                                   bool noCellContrib = false);

  virtual Integer nbNeighbourCellsWithConstraints(Cell cell);

  /* \brief Provides the list of unique IDs of neighboring Nodes, useful for the HG model */
  virtual void getNeighbourNodesUidWithConstraints(Cell cell, Int64UniqueArray neighbournodes);

  /* \brief Returns the local ID for a Cell, taking constraints into account and compacting
   * the numbering. Returns -1 for cells not to be used */
  virtual Int32 localIdWithConstraints(Cell cell);
  virtual Int32 localIdWithConstraints(Int32 cell_lid);

  /* \brief Inverts the functionality of localIdWithConstraints */
  virtual void invertArrayLid2LidCompacted();

  /* \brief Returns the weight(s) associated with the Cells, taking constraints into
   * account. max_nb_weight is 0 if there are no limits */
  virtual SharedArray<float> cellsWeightsWithConstraints(Int32 max_nb_weight = 0, bool ask_lb_cells = false);

  virtual SharedArray<float> cellsSizeWithConstraints();

  /* \brief Returns true if the mesh is used despite the constraints. If false, it is
   * necessary to pass it during an ENUMERATE_CELL */
  virtual bool cellUsedWithConstraints(Cell cell);

  virtual bool cellUsedWithWeakConstraints(std::pair<Int64, Int64>& paired_item);

  /* \brief Assigns the new process number to the cell and others in the same
   * group/constraint if applicable */
  virtual void changeCellOwner(Item cell, VariableItemInt32& cells_new_owner, Int32 new_owner);

  /* \brief Returns true if there are constraints in the mesh. Requires calling
   * initArrayCellsWithConstraints */
  virtual bool haveConstraints() { return m_cells_with_constraints.size() > 0; }

  virtual bool haveWeakConstraints() { return m_cells_with_weak_constraints.size() > 0; }

 protected:

  //! Dumps the partitioning information to disk
  virtual void dumpObject(String filename = "toto");

  virtual void* getCommunicator() const;
  virtual Parallel::Communicator communicator() const;
  virtual bool cellComm() { return true; }

 protected:

  virtual void _initArrayCellsWithConstraints();
  virtual void _initFilterLidCells();
  virtual void _initUidRef();
  virtual void _initUidRef(VariableCellInteger& cell_renum_uid);
  virtual void _initLid2LidCompacted();
  virtual void _initNbCellsWithConstraints();
  virtual void _clearCellWgt();

 protected:

  bool _isNonManifoldMesh() const { return m_is_non_manifold_mesh; }
  Int32 _meshDimension() const { return m_mesh_dimension; }

 private:

  Real _addNgb(const Cell& cell, const Face& face, Int64Array& neighbourcells, Array<bool>& contrib,
               HashTableMapT<Int64, Int32>& map, Array<float>* ptrcommWeights, Int32 offset,
               HashTableMapT<Int32, Int32>& lids, bool special = false);
  bool _createConstraintsLists(Int64MultiArray2& tied_uid);

  SharedArray<float> _cellsProjectWeights(VariableCellArrayReal& cellWgtIn, Int32 nbWgt) const;
  SharedArray<float> _cellsProjectWeights(VariableCellReal& cellWgtIn) const;

 private:

  ISubDomain* m_sub_domain = nullptr;
  IMesh* m_mesh = nullptr;
  IParallelMng* m_pm_sub = nullptr; // sub communicator for partitioning libraries.

 protected:

  IItemFamily* m_cell_family = nullptr;

 private:

  ILoadBalanceMng* m_lbMng = nullptr;
  ILoadBalanceMngInternal* m_lb_mng_internal = nullptr;

  Real m_maximum_computation_time = 0.0;
  Real m_imbalance = 0.0;
  Real m_max_imbalance = 0.0;
  UniqueArray<Real> m_computation_times;

  // Used internally to build the graph/hypergraph
  UniqueArray<SharedArray<Cell>> m_cells_with_constraints;
  std::set<std::pair<Int64, Int64>> m_cells_with_weak_constraints;
  Integer m_nb_cells_with_constraints = 0;
  UniqueArray<eMarkCellWithConstraint> m_filter_lid_cells;
  UniqueArray<Int32> m_local_id_2_local_id_compacted;
  VariableCellInt64* m_unique_id_reference = nullptr;

  void _checkCreateVar();
  UniqueArray<Int32> m_check;
  bool m_is_non_manifold_mesh = false;
  Int32 m_mesh_dimension = -1;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
