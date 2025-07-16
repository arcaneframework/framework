// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SharedMemoryParallelMng.h                                   (C) 2000-2024 */
/*                                                                           */
/* Implémentation des messages en mode mémoire partagé.                      */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_PARALLEL_THREAD_SHAREDMEMORYPARALLELMNG_H
#define ARCANE_PARALLEL_THREAD_SHAREDMEMORYPARALLELMNG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/ParallelMngDispatcher.h"

#include "arcane/utils/TraceInfo.h"
#include "arcane/utils/NotImplementedException.h"
#include "arcane/utils/Ref.h"
#include "arccore/base/ReferenceCounter.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class SerializeBuffer;
class IParallelTopology;
}

namespace Arcane::MessagePassing
{
class SharedMemoryMachineMemoryWindowBaseCreator;
class ISharedMemoryMessageQueue;
class SharedMemoryAllDispatcher;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Infos pour construire un SharedMemoryParallelMng.
 */
struct ARCANE_THREAD_EXPORT SharedMemoryParallelMngBuildInfo
{
 public:
  SharedMemoryParallelMngBuildInfo()
  : rank(-1), nb_rank(0), trace_mng(nullptr), thread_mng(nullptr)
  , message_queue(nullptr), thread_barrier(nullptr), all_dispatchers(nullptr){}
 public:
  Int32 rank;
  Int32 nb_rank;
  ITraceMng* trace_mng;
  IThreadMng* thread_mng;
  Ref<IParallelMng> sequential_parallel_mng;
  IParallelMng* world_parallel_mng = nullptr;
  ISharedMemoryMessageQueue* message_queue = nullptr;
  IThreadBarrier* thread_barrier = nullptr;
  SharedMemoryAllDispatcher* all_dispatchers = nullptr;
  IParallelMngContainerFactory* sub_builder_factory = nullptr;
  Ref<IParallelMngContainer> container;
  MP::Communicator communicator;
  SharedMemoryMachineMemoryWindowBaseCreator* window_creator;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Gestionnaire du parallélisme utilisant les threads.
 */
class ARCANE_THREAD_EXPORT SharedMemoryParallelMng
: public ParallelMngDispatcher
{
 public:
  class RequestList;
  class Impl;

 public:

  explicit SharedMemoryParallelMng(const SharedMemoryParallelMngBuildInfo& build_info);
  ~SharedMemoryParallelMng() override;
  
  bool isParallel()  const override { return m_is_parallel; }
  Int32 commRank() const override { return m_rank; }
  Int32 commSize() const override { return m_nb_rank; }
  void* getMPICommunicator() override { return m_mpi_communicator.communicatorAddress(); }
  MP::Communicator communicator() const override { return m_mpi_communicator; }
  bool isThreadImplementation() const override { return true; }
  bool isHybridImplementation() const override { return false; }
  ITraceMng* traceMng() const override { return m_trace.get(); }
  IThreadMng* threadMng() const override { return m_thread_mng; }
  void initialize() override;
  bool isMasterIO() const override { return commRank()==0; }
  Integer masterIORank() const override { return 0; }
  IIOMng* ioMng() const override { return m_io_mng; }
  IParallelMng* worldParallelMng() const override { return m_world_parallel_mng; }
  ITimerMng* timerMng() const override { return m_timer_mng; }

  IParallelMng* sequentialParallelMng() override;
  Ref<IParallelMng> sequentialParallelMngRef() override;
  void sendSerializer(ISerializer* values,Int32 rank) override;
  Request sendSerializer(ISerializer* values,Int32 rank,ByteArray& bytes) override;
  ISerializeMessage* createSendSerializer(Int32 rank) override;

  void recvSerializer(ISerializer* values,Int32 rank) override;
  ISerializeMessage* createReceiveSerializer(Int32 rank) override;

  void freeRequests(ArrayView<Parallel::Request> requests) override;

  void broadcastSerializer(ISerializer* values,Int32 rank) override;
  MessageId probe(const PointToPointMessageInfo& message) override;
  MessageSourceInfo legacyProbe(const PointToPointMessageInfo& message) override;
  Request sendSerializer(const ISerializer* values,const PointToPointMessageInfo& message) override;
  Request receiveSerializer(ISerializer* values,const PointToPointMessageInfo& message) override;

  void printStats() override;
  void barrier() override;
  void waitAllRequests(ArrayView<Request> requests) override;

  ARCANE_DEPRECATED_260  Real reduceRank(eReduceType rt,Real v,Int32* rank)
  {
    Real rv = reduce(rt,v);
    if (rank)
      *rank = 0;
    return rv;
  }

  IParallelNonBlockingCollective* nonBlockingCollective() const override { return 0; }

  void build() override;

  IParallelMngInternal* _internalApi() override { return m_parallel_mng_internal; }

 public:
  
  IThreadBarrier* getThreadBarrier()
  {
    return m_thread_barrier;
  }

 public:

  //! Construit un message avec pour destinataire \a dest
  PointToPointMessageInfo buildMessage(Int32 dest,MP::eBlockingType is_blocking);
  PointToPointMessageInfo buildMessage(const PointToPointMessageInfo& orig_message);

 protected:
  
  IGetVariablesValuesParallelOperation* createGetVariablesValuesOperation() override;
  ITransferValuesParallelOperation* createTransferValuesOperation() override;
  IParallelTopology* createTopology() override;
  IParallelExchanger* createExchanger() override; 
  IVariableSynchronizer* createSynchronizer(IItemFamily* family) override;
  IVariableSynchronizer* createSynchronizer(const ItemGroup& group) override;
  Parallel::IStat* stat() override { return m_stat; }
  IParallelReplication* replication() const override;
  void setReplication(IParallelReplication* v) override;
  Ref<Parallel::IRequestList> createRequestListRef() override;
  Ref<IParallelMng> createSubParallelMngRef(Int32ConstArrayView kept_ranks) override;
  Ref<IParallelMngUtilsFactory> _internalUtilsFactory() const override;

 protected:

  ISerializeMessageList* _createSerializeMessageList() override;
  IParallelMng* _createSubParallelMng(Int32ConstArrayView kept_ranks) override;
  bool _isAcceleratorAware() const override { return true; }

 private:
  
  ReferenceCounter<ITraceMng> m_trace;
  IThreadMng* m_thread_mng;
  Ref<IParallelMng> m_sequential_parallel_mng;
  ITimerMng* m_timer_mng;
  IParallelReplication* m_replication;
  IParallelMng* m_world_parallel_mng;
  IIOMng* m_io_mng;
  ISharedMemoryMessageQueue* m_message_queue;
  bool m_is_parallel;
  Int32 m_rank; //!< Rang de l'instance
  Int32 m_nb_rank; //!< Nombre de rangs
  bool m_is_initialized; //!< \a true si déjà initialisé
  Parallel::IStat* m_stat;
  IThreadBarrier* m_thread_barrier;
  SharedMemoryAllDispatcher* m_all_dispatchers;
  IParallelMngContainerFactory* m_sub_builder_factory;
  Ref<IParallelMngContainer> m_parent_container_ref;
  MP::Communicator m_mpi_communicator;
  Ref<IParallelMngUtilsFactory> m_utils_factory;
  IParallelMngInternal* m_parallel_mng_internal = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::MessagePassing

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
