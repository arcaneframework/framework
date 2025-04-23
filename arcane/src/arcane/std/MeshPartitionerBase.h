// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshPartitionerBase.h                                       (C) 2000-2025 */
/*                                                                           */
/* Classe de base d'un partitionneur de maillage.                            */
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
 * \brief Classe de base d'un service d'équilibrage de charge.
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

  void setCellsWeight(ArrayView<float> weights,Integer nb_weight) override;
  ArrayView<float> cellsWeight() const override;

  // CORRECT
  Integer nbCellWeight() const;
  void setILoadBalanceMng(ILoadBalanceMng* mng) override { m_lbMng = mng; }
  ILoadBalanceMng* loadBalanceMng() const override { return m_lbMng; }


 public:

  void notifyEndPartition() override { loadBalanceMng()->notifyEndPartition(); }

 public:

  /*! \brief Positionne les nouveaux propriétaires des noeuds, arêtes
   * et faces à partir des mailles.
   *
   * En considérant que les nouveaux propriétaires des mailles sont
   * connues (et synchronisées), détermine les nouveaux propriétaires des autres
   * entités.
   */
  virtual void changeOwnersFromCells();

  enum eMarkCellWithConstraint {eCellClassical, eCellReference, eCellGrouped, eCellGhost, eCellInAConstraint};

  /* \brief Initialise les structures pour le cas avec des contraintes de découpage */
  virtual void initConstraints(bool uidref=true);

  /* \brief Libération des tableaux temporaires */
  virtual void freeConstraints();

  /* \brief Retourne le nombre de Cell après regroupement suivant les contraintes, internes */
  virtual Int32 nbOwnCellsWithConstraints() const;

  /* \brief Renseigne sur la liste des uniqueId des Cell voisines d'une cell en tenant compte des contraintes */
  virtual Real getNeighbourCellsUidWithConstraints(Cell cell, Int64Array& neighbourcells, Array<float> *commWeights = NULL,
                                                   bool noCellContrib = false);

  virtual Integer nbNeighbourCellsWithConstraints(Cell cell);

  /* \brief Renseigne sur la liste des uniqueId des Nodes voisins, utile pour le modele HG */
  virtual void getNeighbourNodesUidWithConstraints(Cell cell, Int64UniqueArray neighbournodes);

  /* \brief Retourne le local id pour une Cell en tenant compte des contraintes et en compactant la numérotation
     retourne -1 pour les mailles à ne pas utiliser */
  virtual Int32 localIdWithConstraints(Cell cell);
  virtual Int32 localIdWithConstraints(Int32 cell_lid);

  /* \brief Renverse le fonctionnement de  localIdWithConstraints */
  virtual void invertArrayLid2LidCompacted();

  /* \brief Retourne le[s] poids associés aux Cell en tenant compte des contraintes
     max_nb_weight à 0 s'il n'y a pas de limites
  */
  virtual SharedArray<float> cellsWeightsWithConstraints(Int32 max_nb_weight=0, bool ask_lb_cells=false);

  virtual SharedArray<float> cellsSizeWithConstraints();


  /* \brief Retourne vrai si la maille est utilisée malgrès les contraintes
     si c'est faux il est nécessaire de la passer lors d'un ENUMERATE_CELL */
  virtual bool cellUsedWithConstraints(Cell cell);

  virtual bool cellUsedWithWeakConstraints(std::pair<Int64,Int64>& paired_item);

  /* \brief Affecte le nouveau numéro de proc à la mailles et autres du même groupe/contrainte s'il y a lieu */
  virtual void changeCellOwner(Item cell, VariableItemInt32& cells_new_owner, Int32 new_owner);


  /* \brief Retourne vrai s'il a des contraintes dans le maillage
   Nécessite d'avoir fait initArrayCellsWithConstraints */
  virtual bool haveConstraints() {return m_cells_with_constraints.size() > 0;}

  virtual bool haveWeakConstraints() {return m_cells_with_weak_constraints.size() > 0;}

 protected:

  //! Dump les informations de repartitionnement sur le disque
  virtual void dumpObject(String filename="toto");

  virtual void* getCommunicator() const;
  virtual Parallel::Communicator communicator() const;
  virtual bool cellComm() {return true; }

 protected:

  virtual void _initArrayCellsWithConstraints();
  virtual void _initFilterLidCells();
  virtual void _initUidRef();
  virtual void _initUidRef(VariableCellInteger& cell_renum_uid);
  virtual void _initLid2LidCompacted();
  virtual void _initNbCellsWithConstraints();
  virtual void  _clearCellWgt();

 protected:

  bool _isNonManifoldMesh() const { return m_is_non_manifold_mesh; }
  Int32 _meshDimension() const { return m_mesh_dimension; }

 private:

  Real _addNgb(const Cell& cell, const Face& face, Int64Array& neighbourcells, Array<bool>& contrib,
               HashTableMapT<Int64,Int32>& map, Array<float> *ptrcommWeights, Int32 offset,
               HashTableMapT<Int32,Int32>& lids,  bool special=false);
  bool _createConstraintsLists(Int64MultiArray2& tied_uid);

  SharedArray<float> _cellsProjectWeights(VariableCellArrayReal& cellWgtIn, Int32 nbWgt) const;
  SharedArray<float> _cellsProjectWeights(VariableCellReal& cellWgtIn) const;

 private:

  ISubDomain* m_sub_domain = nullptr;
  IMesh* m_mesh = nullptr;
  IParallelMng* m_pm_sub = nullptr; // sous communicateur pour les bibliotheques de partitionnement.

 protected:

  IItemFamily* m_cell_family = nullptr;

 private:

  ILoadBalanceMng* m_lbMng = nullptr;
  ILoadBalanceMngInternal* m_lb_mng_internal = nullptr;

  Real m_maximum_computation_time = 0.0;
  Real m_imbalance = 0.0;
  Real m_max_imbalance = 0.0;
  UniqueArray<Real> m_computation_times;

  // Utile en interne pour construire le graphe/hypergraphe
  UniqueArray<SharedArray<Cell> > m_cells_with_constraints;
  std::set<std::pair<Int64, Int64> > m_cells_with_weak_constraints;
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
