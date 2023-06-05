// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VariableSynchronizerDispatcher.h                            (C) 2000-2023 */
/*                                                                           */
/* Service de synchronisation des variables.                                 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IMPL_VARIABLESYNCHRONIZERDISPATCHER_H
#define ARCANE_IMPL_VARIABLESYNCHRONIZERDISPATCHER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/UniqueArray.h"
#include "arcane/utils/Ref.h"
#include "arccore/base/ReferenceCounterImpl.h"

#include "arcane/core/ArcaneTypes.h"
#include "arcane/core/Parallel.h"
#include "arcane/core/VariableCollection.h"

#include "arcane/impl/IGenericVariableSynchronizerDispatcher.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class VariableSynchronizerDispatcher;
class VariableSynchronizerMultiDispatcher;
class IVariableSynchronizerDispatcher;
class GroupIndexTable;
class INumericDataInternal;
using IVariableSynchronizeDispatcher = IVariableSynchronizerDispatcher;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Informations sur la liste des entités partagées/fantômes pour
 * un rang donné pour une synchronisation.
 *
 * TODO: Utiliser pour toutes les VariableSyncInfo un seul tableau pour les
 *       entités partagées et un seul tableau pour les entités fantômes qui
 *       sera géré par ItemGroupSynchronizeInfo.
 */
class ARCANE_IMPL_EXPORT VariableSyncInfo
{
 public:

  VariableSyncInfo(Int32ConstArrayView share_ids, Int32ConstArrayView ghost_ids, Int32 rank);
  VariableSyncInfo(const VariableSyncInfo& rhs);
  VariableSyncInfo();

 public:

  //! Rang du processeur cible
  Int32 targetRank() const { return m_target_rank; }

  //! localIds() des entités à envoyer au rang targetRank()
  ConstArrayView<Int32> shareIds() const { return m_share_ids; }
  //! localIds() des entités à réceptionner du rang targetRank()
  ConstArrayView<Int32> ghostIds() const { return m_ghost_ids; }

  //! Nombre d'entités partagées
  Int32 nbShare() const { return m_share_ids.size(); }
  //! Nombre d'entités fantômes
  Int32 nbGhost() const { return m_ghost_ids.size(); }

  //! Met à jour les informations lorsque les localId() des entités changent
  void changeLocalIds(Int32ConstArrayView old_to_new_ids);

 private:

  //! localIds() des entités à envoyer au processeur #m_rank
  UniqueArray<Int32> m_share_ids;
  //! localIds() des entités à réceptionner du processeur #m_rank
  UniqueArray<Int32> m_ghost_ids;
  //! Rang du processeur cible
  Int32 m_target_rank = A_NULL_RANK;

 private:

  void _changeIds(Array<Int32>& ids, Int32ConstArrayView old_to_new_ids);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Informations nécessaires pour synchroniser les entités sur un groupe.
 *
 * Il faut appeler recompute() après avoir ajouté ou modifier les instances
 * de VariableSyncInfo.
 */
class ARCANE_IMPL_EXPORT ItemGroupSynchronizeInfo
: private ReferenceCounterImpl
{
 private:

  ItemGroupSynchronizeInfo() = default;

 public:

  ItemGroupSynchronizeInfo(const ItemGroupSynchronizeInfo&) = delete;
  ItemGroupSynchronizeInfo operator=(const ItemGroupSynchronizeInfo&) = delete;
  ItemGroupSynchronizeInfo(ItemGroupSynchronizeInfo&&) = delete;
  ItemGroupSynchronizeInfo operator=(ItemGroupSynchronizeInfo&&) = delete;

 public:

  static Ref<ItemGroupSynchronizeInfo> create()
  {
    return makeRef(new ItemGroupSynchronizeInfo());
  }

 public:

  VariableSyncInfo& operator[](Int32 i) { return m_ranks_info[i]; }
  const VariableSyncInfo& operator[](Int32 i) const { return m_ranks_info[i]; }

  VariableSyncInfo& rankInfo(Int32 i) { return m_ranks_info[i]; }
  const VariableSyncInfo& rankInfo(Int32 i) const { return m_ranks_info[i]; }

  void clear() { m_ranks_info.clear(); }
  Int32 size() const { return m_ranks_info.size(); }
  void add(const VariableSyncInfo& s) { m_ranks_info.add(s); }
  Int64 shareDisplacement(Int32 index) const { return m_share_displacements_base[index]; }
  Int64 ghostDisplacement(Int32 index) const { return m_ghost_displacements_base[index]; }
  Int64 totalNbGhost() const { return m_total_nb_ghost; }
  Int64 totalNbShare() const { return m_total_nb_share; }

  //! Notifie l'instance que les indices locaux ont changé
  void changeLocalIds(Int32ConstArrayView old_to_new_ids);

  //! Notifie l'instance que les valeurs ont changé
  void recompute();

 public:

  void addReference() { ReferenceCounterImpl::addReference(); }
  void removeReference() { ReferenceCounterImpl::removeReference(); }

 public:

  ARCANE_DEPRECATED_REASON("Y2023: use operator[] instead")
  ConstArrayView<VariableSyncInfo> infos() const { return m_ranks_info; }

  ARCANE_DEPRECATED_REASON("Y2023: use operator[] instead")
  ArrayView<VariableSyncInfo> infos() { return m_ranks_info; }

 private:

  UniqueArray<VariableSyncInfo> m_ranks_info;
  //! Déplacement dans le buffer fantôme de chaque rang
  UniqueArray<Int64> m_ghost_displacements_base;
  //! Déplacement dans le buffer partagé de chaque rang
  UniqueArray<Int64> m_share_displacements_base;
  Int64 m_total_nb_ghost = 0;
  Int64 m_total_nb_share = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Infos pour construire un VariableSynchronizeDispatcher.
 */
class ARCANE_IMPL_EXPORT VariableSynchronizeDispatcherBuildInfo
{
 public:

  VariableSynchronizeDispatcherBuildInfo(IParallelMng* pm, GroupIndexTable* table,
                                         Ref<IDataSynchronizeImplementationFactory> factory)
  : m_parallel_mng(pm)
  , m_table(table)
  , m_factory(factory)
  {}

 public:

  IParallelMng* parallelMng() const { return m_parallel_mng; }
  //! Table d'index pour le groupe. Peut-être nul.
  GroupIndexTable* table() const { return m_table; }
  Ref<IDataSynchronizeImplementationFactory> factory() const
  {
    return m_factory;
  }

 private:

  IParallelMng* m_parallel_mng;
  GroupIndexTable* m_table;
  Ref<IDataSynchronizeImplementationFactory> m_factory;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface pour gérer l'envoi de la synchronisation.
 *
 * Il faut utiliser create() pour créer une implémentation pour cette
 * interface.
 *
 * Il faut appeler \a setItemGroupSynchronizeInfo() pour initialiser
 * l'instance.
 */
class ARCANE_IMPL_EXPORT IVariableSynchronizerDispatcher
{
  ARCCORE_DECLARE_REFERENCE_COUNTED_INCLASS_METHODS();

 public:

  virtual ~IVariableSynchronizerDispatcher() = default;

 public:

  virtual void setItemGroupSynchronizeInfo(ItemGroupSynchronizeInfo* sync_info) = 0;

  /*!
   * \brief Recalcule les informations nécessaires après une mise à jour des informations
   * de \a ItemGroupSynchronizeInfo.
   */
  virtual void compute() = 0;

  //! Commence l'exécution pour la synchronisation pour la donnée \a data.
  virtual void beginSynchronize(INumericDataInternal* data) = 0;

  //! Termine la synchronisation.
  virtual void endSynchronize() = 0;

 public:

  static Ref<IVariableSynchronizeDispatcher>
  create(const VariableSynchronizeDispatcherBuildInfo& build_info);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Interface de la synchronisation d'une liste de variables.
 */
class ARCANE_IMPL_EXPORT IVariableSynchronizerMultiDispatcher
{
 public:

  virtual ~IVariableSynchronizerMultiDispatcher() = default;

 public:

  virtual void synchronize(VariableCollection vars, ItemGroupSynchronizeInfo* sync_info) = 0;

 public:

  static IVariableSynchronizerMultiDispatcher* create(const VariableSynchronizeDispatcherBuildInfo& bi);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
