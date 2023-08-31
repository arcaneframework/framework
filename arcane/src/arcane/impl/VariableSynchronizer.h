// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VariableSynchronizer.h                                      (C) 2000-2023 */
/*                                                                           */
/* Service de synchronisation des variables.                                 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IMPL_VARIABLESYNCHRONIZER_H
#define ARCANE_IMPL_VARIABLESYNCHRONIZER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"
#include "arcane/utils/NotSupportedException.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/Event.h"

#include "arcane/Parallel.h"
#include "arcane/ItemGroup.h"
#include "arcane/IVariableSynchronizer.h"
#include "arcane/IParallelMng.h"

#include "arcane/impl/IBufferCopier.h"
#include "arcane/impl/VariableSynchronizerDispatcher.h"

#include "arcane/DataTypeDispatchingDataVisitor.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class Timer;
class INumericDataInternal;
class DataSynchronizeResult;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface d'un service de synchronisation de variable.
 *
 * Une instance de cette classe est créée via
 * IParallelMng::createVariableSynchronizer(). Une instance est associée
 * à un groupe d'entité. Il faut appeller la fonction compute()
 * pour calculer les infos de synchronisation.
 */
class ARCANE_IMPL_EXPORT VariableSynchronizer
: public TraceAccessor
, public IVariableSynchronizer
{
  class RankInfo
  {
   public:
    RankInfo() : m_rank(A_NULL_RANK) {}
    RankInfo(Int32 arank)
    : m_rank(arank) {}
   public:
    Int32 rank() const { return m_rank; }
    void setRank(Int32 arank) { m_rank = arank; }
    /*!
     * \brief Opérateur de comparaison.
     * Une instance est considérée comme inférieure à une autre si
     * son sous-domaine associé est plus petit que celui de l'autre.
     */
    bool operator<(const RankInfo& ar) const
    {
      return m_rank < ar.m_rank;
    }
   private:
    Int32 m_rank;
  };

  class GhostRankInfo : public RankInfo
  {
   public:
    GhostRankInfo() : m_nb_item(0) {}
    GhostRankInfo(Int32 arank)
    : RankInfo(arank), m_nb_item(0) {}
    GhostRankInfo(Int32 arank,Integer nb_item)
    : RankInfo(arank), m_nb_item(nb_item) {}
   public:
    void setInfos(Int32 arank,SharedArray<Int32>& local_ids)
    {
      setRank(arank);
      m_nb_item = local_ids.size();
      m_local_ids = local_ids;
    }
    Int32ConstArrayView localIds() const { return m_local_ids; }
    Integer nbItem() const { return m_nb_item; }
    void resize() { m_unique_ids.resize(m_nb_item); }
    Int64ArrayView uniqueIds() { return m_unique_ids; }

   private:
    Integer m_nb_item;
    SharedArray<Int32> m_local_ids;
    SharedArray<Int64> m_unique_ids;
  };

  class ShareRankInfo : public RankInfo
  {
   public:
    ShareRankInfo() : m_nb_item(0) {}
    ShareRankInfo(Int32 arank,Integer nb_item)
    : RankInfo(arank), m_nb_item(nb_item) {}
    ShareRankInfo(Int32 arank)
    : RankInfo(arank), m_nb_item(0) {}
   public:
    void setInfos(Int32 arank,SharedArray<Int32>& local_ids)
    {
      setRank(arank);
      m_nb_item = local_ids.size();
      m_local_ids = local_ids;
    }
    Int32ConstArrayView localIds() const { return m_local_ids; }
    void setLocalIds(SharedArray<Int32>& v) { m_local_ids = v; }
    Integer nbItem() const { return m_nb_item; }
    void resize() { m_unique_ids.resize(m_nb_item); }
    Int64ArrayView uniqueIds() { return m_unique_ids; }
   private:
    Integer m_nb_item;
    SharedArray<Int32> m_local_ids;
    SharedArray<Int64> m_unique_ids;
  };

 public:

  VariableSynchronizer(IParallelMng* pm,const ItemGroup& group,
                       Ref<IDataSynchronizeImplementationFactory> implementation_factory);
  ~VariableSynchronizer() override;

 public:

  IParallelMng* parallelMng() override
  {
    return m_parallel_mng;
  }

  const ItemGroup& itemGroup() override
  {
    return m_item_group;
  }

  void compute() override;

  void changeLocalIds(Int32ConstArrayView old_to_new_ids) override;

  void synchronize(IVariable* var) override;

  void synchronize(VariableCollection vars) override;

  Int32ConstArrayView communicatingRanks() override;

  Int32ConstArrayView sharedItems(Int32 index) override;

  Int32ConstArrayView ghostItems(Int32 index) override;

  void synchronizeData(IData* data) override;

  EventObservable<const VariableSynchronizerEventArgs&>& onSynchronized() override
  {
    return m_on_synchronized;
  }

 private:

  IParallelMng* m_parallel_mng = nullptr;
  ItemGroup m_item_group;
  Ref<DataSynchronizeInfo> m_sync_list;
  Int32UniqueArray m_communicating_ranks;
  Ref<IVariableSynchronizerDispatcher> m_dispatcher;
  IVariableSynchronizerMultiDispatcher* m_multi_dispatcher = nullptr;
  Timer* m_sync_timer = nullptr;
  bool m_is_verbose = false;
  bool m_allow_multi_sync = true;
  bool m_trace_sync = false;
  bool m_is_compare_sync = false;
  EventObservable<const VariableSynchronizerEventArgs&> m_on_synchronized;
  Ref<IDataSynchronizeImplementationFactory> m_implementation_factory;

 private:

  void _createList(UniqueArray<SharedArray<Int32> >& boundary_items);
  void _checkValid(ArrayView<GhostRankInfo> ghost_ranks_info,
                   ArrayView<ShareRankInfo> share_ranks_info);
  void _printSyncList();
  void _synchronize(IVariable* var);
  void _synchronizeMulti(VariableCollection vars);
  bool _canSynchronizeMulti(const VariableCollection& vars);
  DataSynchronizeResult _synchronize(INumericDataInternal* data);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
