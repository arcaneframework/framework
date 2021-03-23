// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VariableSynchronizer.h                                      (C) 2000-2017 */
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

#include "arcane/DataTypeDispatchingDataVisitor.h"


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class VariableSynchronizer;
class Timer;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class SimpleType>
class VariableSynchronize1D;
class VariableSynchronizerMultiDispatcher;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Information pour une synchro d'une variable avec un processeur.
 */
class ARCANE_IMPL_EXPORT VariableSyncInfo
{
 public:

  VariableSyncInfo() : m_target_rank(NULL_SUB_DOMAIN_ID), m_is_send_first(false) {}
  VariableSyncInfo(Int32ConstArrayView share_ids,Int32ConstArrayView ghost_ids,
                   Int32 rank,bool is_send_first)
  : m_share_ids(share_ids), m_ghost_ids(ghost_ids),
    m_target_rank(rank), m_is_send_first(is_send_first) {}
	
 public:

  Int32 targetRank() const { return m_target_rank; }

 public:

  //! localIds() des entités à envoyer au processeur #m_rank
  SharedArray<Int32> m_share_ids;
  //! localIds() des entités à réceptionner du processeur #m_rank
  SharedArray<Int32> m_ghost_ids;
  //! Rang du processeur cible
  Int32 m_target_rank;
  //!< \a true si on envoie avant de recevoir
  bool m_is_send_first;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_IMPL_EXPORT IVariableSynchronizeDispatcher
{
 public:
  typedef FalseType HasStringDispatch;
 public:
  virtual ~IVariableSynchronizeDispatcher(){}
 public:
  virtual void compute(ConstArrayView<VariableSyncInfo> sync_list) =0;
 protected:
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Infos pour construire un VariableSynchronizeDispatcher.
 */
class ARCANE_IMPL_EXPORT VariableSynchronizeDispatcherBuildInfo
{
 public:
  VariableSynchronizeDispatcherBuildInfo(IParallelMng* pm, GroupIndexTable* table)
  : m_parallel_mng(pm), m_table(table) { }
 public:
  IParallelMng* parallelMng() const{ return m_parallel_mng; }
  //! Table d'index pour le groupe. Peut-être nul.
  GroupIndexTable* table() const { return m_table; }
 private:
  IParallelMng* m_parallel_mng;
  GroupIndexTable* m_table;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class SimpleType>
class ARCANE_IMPL_EXPORT VariableSynchronizeDispatcher
: public IDataTypeDataDispatcherT<SimpleType>
, public IVariableSynchronizeDispatcher
{
 public:
  class SyncBuffer
  {
   public:
    void compute(ConstArrayView<VariableSyncInfo> sync_list,Integer dim2_size);
   public:
    Integer m_dim2_size;
    UniqueArray<SimpleType> m_ghost_buffer;
    UniqueArray<SimpleType> m_share_buffer;
    UniqueArray< ArrayView<SimpleType> > m_ghost_locals_buffer;
    UniqueArray< ArrayView<SimpleType> > m_share_locals_buffer;
  };
 public:
  VariableSynchronizeDispatcher(const VariableSynchronizeDispatcherBuildInfo& bi)
  : m_parallel_mng(bi.parallelMng()), m_is_in_sync(false)
  {
    if (bi.table())
      m_buffer_copier = new TableBufferCopier<SimpleType>(bi.table());
    else
      m_buffer_copier = new DirectBufferCopier<SimpleType>();
  }
  virtual ~VariableSynchronizeDispatcher()
  {
    delete m_buffer_copier;
  }
  virtual void applyDispatch(IScalarDataT<SimpleType>* data)
  {
    ARCANE_UNUSED(data);
    throw NotSupportedException(A_FUNCINFO,"Can not synchronize scalar data");
  }
  virtual void applyDispatch(IArrayDataT<SimpleType>* data);
  virtual void applyDispatch(IArray2DataT<SimpleType>* data);
  virtual void applyDispatch(IMultiArray2DataT<SimpleType>* data)
  {
    ARCANE_UNUSED(data);
    throw NotSupportedException(A_FUNCINFO,"Can not synchronize multiarray2 data");
  }
  virtual void compute(ConstArrayView<VariableSyncInfo> sync_list);
  virtual void beginSynchronize(ArrayView<SimpleType> values,SyncBuffer& sync_buffer);
  virtual void endSynchronize(ArrayView<SimpleType> values,SyncBuffer& sync_buffer);

 protected:
  
  void _copyFromBuffer(Int32ConstArrayView indexes,ConstArrayView<SimpleType> buffer,
                       ArrayView<SimpleType> var_value,Integer dim2_size);
  void _copyToBuffer(Int32ConstArrayView indexes,ArrayView<SimpleType> buffer,
                     ConstArrayView<SimpleType> var_value,Integer dim2_size);
 protected:
  IParallelMng* m_parallel_mng;
  IBufferCopier<SimpleType>* m_buffer_copier;
  ConstArrayView<VariableSyncInfo> m_sync_list;
  UniqueArray<Parallel::Request> m_all_requests;
  SyncBuffer m_1d_buffer;
  SyncBuffer m_2d_buffer;
  bool m_is_in_sync;

 private:
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class VariableSynchronizerDispatcher
{
 public:
  typedef DataTypeDispatchingDataVisitor<IVariableSynchronizeDispatcher> DispatcherType;
 public:
  VariableSynchronizerDispatcher(IParallelMng* pm,DispatcherType* dispatcher)
  : m_parallel_mng(pm), m_dispatcher(dispatcher)
  {
  }
  ~VariableSynchronizerDispatcher()
  {
    delete m_dispatcher;
  }
  void compute(ConstArrayView<VariableSyncInfo> sync_list)
  {
    ConstArrayView<IVariableSynchronizeDispatcher*> dispatchers = m_dispatcher->dispatchers();
    m_parallel_mng->traceMng()->info(4) << "DISPATCH RECOMPUTE";
    for( Integer i=0, is=dispatchers.size(); i<is; ++i )
      dispatchers[i]->compute(sync_list);
  }
  IDataVisitor* visitor()
  {
    return m_dispatcher;
  }
 private:
  IParallelMng* m_parallel_mng;
  DispatcherType* m_dispatcher;
};

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
                       VariableSynchronizerDispatcher* dispatcher);
  virtual ~VariableSynchronizer();

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

  IParallelMng* m_parallel_mng;
  ItemGroup m_item_group;
  UniqueArray<VariableSyncInfo> m_sync_list;
  Int32UniqueArray m_communicating_ranks;
  VariableSynchronizerDispatcher* m_dispatcher;
  VariableSynchronizerMultiDispatcher* m_multi_dispatcher;
  Timer* m_sync_timer;
  bool m_is_verbose;
  bool m_allow_multi_sync;
  bool m_trace_sync;
  EventObservable<const VariableSynchronizerEventArgs&> m_on_synchronized;

 private:

  void _createList(UniqueArray<SharedArray<Int32> >& boundary_items);
  void _checkValid(ArrayView<GhostRankInfo> ghost_ranks_info,
                   ArrayView<ShareRankInfo> share_ranks_info);
  void _printSyncList();
  void _synchronize(IVariable* var);
  void _synchronize(VariableCollection vars);
  bool _canSynchronizeMulti(const VariableCollection& vars);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
