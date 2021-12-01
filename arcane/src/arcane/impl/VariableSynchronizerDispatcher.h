// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VariableSynchronizerDispatcher.h                            (C) 2000-2021 */
/*                                                                           */
/* Service de synchronisation des variables.                                 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IMPL_VARIABLESYNCHRONIZERDISPATCHER_H
#define ARCANE_IMPL_VARIABLESYNCHRONIZERDISPATCHER_H
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

namespace Arcane
{

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

class VariableSynchronizerMultiDispatcher
{
 public:
  VariableSynchronizerMultiDispatcher(IParallelMng* pm)
  : m_parallel_mng(pm)
  {
  }

  void synchronize(VariableCollection vars,ConstArrayView<VariableSyncInfo> sync_infos);
 private:
  IParallelMng* m_parallel_mng;
};
  
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespcae Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
