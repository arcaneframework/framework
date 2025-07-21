// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* HybridParallelMng.h                                         (C) 2000-2025 */
/*                                                                           */
/* Implémentation des messages hybrides MPI/Mémoire partagée.                */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_PARALLEL_THREAD_HYBRIDPARALLELMNG_H
#define ARCANE_PARALLEL_THREAD_HYBRIDPARALLELMNG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/NotImplementedException.h"
#include "arcane/utils/Array.h"
#include "arcane/utils/Ref.h"

#include "arcane/core/ParallelMngDispatcher.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class SerializeBuffer;
class MpiParallelMng;
}

namespace Arcane::MessagePassing
{
class HybridMachineMemoryWindowBaseCreator;
class ISharedMemoryMessageQueue;
class HybridMessageQueue;
class HybridParallelMng;
class MpiThreadAllDispatcher;
class HybridSerializeMessageList;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Infos pour construire un HybridParallelMng.
 */
struct HybridParallelMngBuildInfo
{
 public:
  Int32 local_rank = -1;
  Int32 local_nb_rank = -1;
  MpiParallelMng* mpi_parallel_mng = nullptr;
  ITraceMng* trace_mng = nullptr;
  IThreadMng* thread_mng = nullptr;
  IParallelMng* world_parallel_mng = nullptr;
  ISharedMemoryMessageQueue* message_queue = nullptr;
  IThreadBarrier* thread_barrier = nullptr;
  Array<HybridParallelMng*>* parallel_mng_list = nullptr;
  MpiThreadAllDispatcher* all_dispatchers = nullptr;
  IParallelMngContainerFactory* sub_builder_factory = nullptr;
  Ref<IParallelMngContainer> container;
  HybridMachineMemoryWindowBaseCreator* window_creator;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Gestionnaire du parallélisme utilisant les threads.
 */
class HybridParallelMng
: public ParallelMngDispatcher
{
  friend HybridSerializeMessageList;
  class RequestList;
  class Impl;

 public:

  explicit HybridParallelMng(const HybridParallelMngBuildInfo& bi);
  ~HybridParallelMng() override;
  
  bool isParallel()  const override { return m_is_parallel; }
  Int32 commRank() const override { return m_global_rank; }
  Int32 commSize() const override { return m_global_nb_rank; }
  void* getMPICommunicator() override;
  MP::Communicator communicator() const override;
  bool isThreadImplementation() const override { return true; }
  bool isHybridImplementation() const override { return true; }
  ITraceMng* traceMng() const override { return m_trace; }
  IThreadMng* threadMng() const override { return m_thread_mng; }
  IParallelMng* worldParallelMng() const override { return m_world_parallel_mng; }
  IIOMng* ioMng() const override { return m_io_mng; }

  void initialize() override;
  bool isMasterIO() const override { return commRank()==0; }
  Int32 masterIORank() const override { return 0; }

  ITimerMng* timerMng() const override { return m_timer_mng; }

  IParallelMng* sequentialParallelMng() override;
  Ref<IParallelMng> sequentialParallelMngRef() override;
  void sendSerializer(ISerializer* values,Int32 rank) override;
  Request sendSerializer(ISerializer* values,Int32 rank,ByteArray& bytes) override;
  ISerializeMessage* createSendSerializer(Int32 rank) override;

  void recvSerializer(ISerializer* values,Int32 rank) override;
  ISerializeMessage* createReceiveSerializer(Int32 rank) override;

  void freeRequests(ArrayView<Request> requests) override;

  void broadcastSerializer(ISerializer* values,Int32 rank) override;
  MessageId probe(const PointToPointMessageInfo& message) override;
  MessageSourceInfo legacyProbe(const PointToPointMessageInfo& message) override;
  Request sendSerializer(const ISerializer* values,const PointToPointMessageInfo& message) override;
  Request receiveSerializer(ISerializer* values,const PointToPointMessageInfo& message) override;

  void printStats() override;
  void barrier() override;
  void waitAllRequests(ArrayView<Request> requests) override;

  IParallelNonBlockingCollective* nonBlockingCollective() const override { return nullptr; }

  void build() override;

 public:
  
  Int32 localRank() const { return m_local_rank; }
  Int32 localNbRank() const { return m_local_nb_rank; }
  MpiParallelMng* mpiParallelMng() { return m_mpi_parallel_mng; }
  //! Construit un message avec pour destinataire \a dest
  PointToPointMessageInfo buildMessage(Int32 dest,MP::eBlockingType is_blocking);
  PointToPointMessageInfo buildMessage(const PointToPointMessageInfo& message);

 public:

  IParallelMngInternal* _internalApi() override { return m_parallel_mng_internal; }

 protected:
  
  IGetVariablesValuesParallelOperation* createGetVariablesValuesOperation() override;
  ITransferValuesParallelOperation* createTransferValuesOperation() override;
  IParallelExchanger* createExchanger() override;
  IParallelTopology* createTopology() override;
  IVariableSynchronizer* createSynchronizer(IItemFamily* family) override;
  IVariableSynchronizer* createSynchronizer(const ItemGroup& group) override;
  Parallel::IStat* stat() override { return m_stat; }
  IParallelReplication* replication() const override;
  void setReplication(IParallelReplication* v) override;
  Ref<Parallel::IRequestList> createRequestListRef() override;
  ISerializeMessageList* _createSerializeMessageList() override;
  IParallelMng* _createSubParallelMng(Int32ConstArrayView kept_ranks) override;
  Ref<IParallelMng> createSubParallelMngRef(Int32ConstArrayView kept_ranks) override;
  Ref<IParallelMngUtilsFactory> _internalUtilsFactory() const override;
  bool _isAcceleratorAware() const override;

 public:

  IThreadBarrier* getThreadBarrier()
  {
    return m_thread_barrier;
  }

 private:
  
  ITraceMng* m_trace;
  IThreadMng* m_thread_mng;
  IParallelMng* m_world_parallel_mng;
  IIOMng* m_io_mng;
  Ref<IParallelMng> m_sequential_parallel_mng;
  ITimerMng* m_timer_mng;
  IParallelReplication* m_replication;
  HybridMessageQueue* m_message_queue;
  bool m_is_parallel;
  Int32 m_global_rank; //!< Numéro du processeur actuel
  Int32 m_global_nb_rank; //!< Nombre de rangs globaux
  Int32 m_local_rank; //!< Rang local du processeur actuel
  Int32 m_local_nb_rank; //!< Nombre de rang locaux
  bool m_is_initialized; //!< \a true si déjà initialisé
  Parallel::IStat* m_stat = nullptr;
  IThreadBarrier* m_thread_barrier = nullptr;
  MpiParallelMng* m_mpi_parallel_mng = nullptr;
  MpiThreadAllDispatcher* m_all_dispatchers = nullptr;
  Array<HybridParallelMng*>* m_parallel_mng_list = nullptr;
  IParallelMngContainerFactory* m_sub_builder_factory = nullptr;
  Ref<IParallelMngContainer> m_parent_container_ref;
  Ref<IParallelMngUtilsFactory> m_utils_factory;
  IParallelMngInternal* m_parallel_mng_internal = nullptr;

 private:

  SerializeBuffer* _castSerializer(ISerializer* serializer);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::MessagePassing

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
