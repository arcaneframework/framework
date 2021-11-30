﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MpiParallelMng.h                                            (C) 2000-2021 */
/*                                                                           */
/* Implémentation des messages avec MPI.                                     */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_PARALLEL_MPI_MPIPARALLELMNG_H
#define ARCANE_PARALLEL_MPI_MPIPARALLELMNG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Array.h"
#include "arcane/utils/Ref.h"

#include "arcane/ParallelMngDispatcher.h"

#include "arcane/parallel/mpi/ArcaneMpi.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
using Communicator = MP::Communicator;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class MpiDatatypeList;
class SerializeBuffer;
class ArcaneMpiSerializeMessageList;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Infos pour construire un MpiParallelMng.
 */
struct ARCANE_MPI_EXPORT MpiParallelMngBuildInfo
{
 public:
  MpiParallelMngBuildInfo(MPI_Comm comm);
 public:
  Int32 commRank() const { return comm_rank; }
  Int32 commSize() const { return comm_nb_rank; }
  MPI_Comm mpiComm() const { return mpi_comm; }
  MP::Dispatchers* dispatchers() const { return m_dispatchers; }
  MP::Mpi::MpiMessagePassingMng* messagePassingMng() const { return m_message_passing_mng; }
 public:
  bool is_parallel;
 private:
  Int32 comm_rank;
  Int32 comm_nb_rank;
 public:
  Parallel::IStat* stat = nullptr;
  ITraceMng* trace_mng = nullptr;
  ITimerMng* timer_mng = nullptr;
  IThreadMng* thread_mng = nullptr;
  IParallelMng* world_parallel_mng = nullptr;
 private:
  MPI_Comm mpi_comm;
 public:
  bool is_mpi_comm_owned;
  MpiLock* mpi_lock = nullptr;
 private:
  MP::Dispatchers* m_dispatchers;
  MP::Mpi::MpiMessagePassingMng* m_message_passing_mng;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Gestionnaire du parallélisme utilisant MPI.
 */
class ARCANE_MPI_EXPORT MpiParallelMng
: public ParallelMngDispatcher
{
 public:
  friend ArcaneMpiSerializeMessageList;
  class SendSerializerSubRequest;
  class ReceiveSerializerSubRequest;
  class RequestList;

 public:

  MpiParallelMng(const MpiParallelMngBuildInfo& bi);
  ~MpiParallelMng() override;
  
  bool isParallel()  const override { return m_is_parallel; }
  Int32 commRank() const override { return m_comm_rank; }
  Int32 commSize() const override { return m_comm_size; }
  void* getMPICommunicator() override { return &m_communicator; }
  bool isThreadImplementation() const override { return false; }
  ITraceMng* traceMng() const override { return m_trace; }
  IThreadMng* threadMng() const override { return m_thread_mng; }
  IParallelMng* worldParallelMng() const override { return m_world_parallel_mng; }
  IIOMng* ioMng() const override { return m_io_mng; }

  void initialize() override;
  bool isMasterIO() const override { return commRank()==0; }
  Integer masterIORank() const override { return 0; }

  ITimerMng* timerMng() const override { return m_timer_mng; }

  void sendSerializer(ISerializer* values,Int32 rank) override;
  Request sendSerializer(ISerializer* values,Int32 rank,ByteArray& bytes) override;
  ISerializeMessage* createSendSerializer(Int32 rank) override;
  void allGatherSerializer(ISerializer* send_serializer,ISerializer* recv_serializer) override;

  void recvSerializer(ISerializer* values,Int32 rank) override;
  ISerializeMessage* createReceiveSerializer(Int32 rank) override;

  void freeRequests(ArrayView<Parallel::Request> requests) override;

  void broadcastSerializer(ISerializer* values,Int32 rank) override;
  MessageId probe(const PointToPointMessageInfo& message) override;
  Request sendSerializer(const ISerializer* values,const PointToPointMessageInfo& message) override;
  Request receiveSerializer(ISerializer* values,const PointToPointMessageInfo& message) override;

  void printStats() override;
  IParallelMng* sequentialParallelMng() override;
  Ref<IParallelMng> sequentialParallelMngRef() override;
  void barrier() override;
  void waitAllRequests(ArrayView<Request> requests) override;
  UniqueArray<Integer> waitSomeRequests(ArrayView<Request> requests) override;
  UniqueArray<Integer> testSomeRequests(ArrayView<Request> requests) override;
  ARCANE_DEPRECATED_260 Real reduceRank(eReduceType rt,Real v,Int32* rank)
  {
    Real rv = reduce(rt,v);
    if (rank)
      *rank = 0;
    return rv;
  }

  IParallelNonBlockingCollective* nonBlockingCollective() const override { return m_non_blocking_collective; }

  void build() override;

 public:
  
  MpiAdapter* adapter() { return m_adapter; }
  Communicator communicator() const override { return Communicator(m_communicator); }

  MpiLock* mpiLock() const { return m_mpi_lock; }

  MpiDatatypeList* datatypes() { return m_datatype_list; }

  MpiSerializeDispatcher* serializeDispatcher() const { return m_mpi_serialize_dispatcher; }

 protected:

  ISerializeMessageList* _createSerializeMessageList() override;
  IParallelMng* _createSubParallelMng(Int32ConstArrayView kept_ranks) override;

 public:

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
  Ref<IParallelMngUtilsFactory> _internalUtilsFactory() const override;

 private:
  
  ITraceMng* m_trace;
  IThreadMng* m_thread_mng;
  IParallelMng* m_world_parallel_mng;
  IIOMng* m_io_mng;
  Ref<IParallelMng> m_sequential_parallel_mng;
  ITimerMng* m_timer_mng;
  IParallelReplication* m_replication;
  bool m_is_timer_owned;
  MpiDatatypeList* m_datatype_list;
  MpiAdapter* m_adapter;
  bool m_is_parallel;
  Int32 m_comm_rank; //!< Numéro du processeur actuel
  Int32 m_comm_size; //!< Nombre de sous-domaines
  bool m_is_initialized; //!< \a true si déjà initialisé
  Parallel::IStat* m_stat;
  MPI_Comm m_communicator;
  bool m_is_communicator_owned;
  MpiLock* m_mpi_lock;
  IParallelNonBlockingCollective* m_non_blocking_collective;
  MpiSerializeDispatcher* m_mpi_serialize_dispatcher = nullptr;
  Ref<IParallelMngUtilsFactory> m_utils_factory;

 private:

  void _checkInit();
  SerializeBuffer* _castSerializer(ISerializer* serializer);
  const SerializeBuffer* _castSerializer(const ISerializer* serializer);
  void _checkBigMessage(Int64 message_size);
  void _checkFinishedSubRequests();
  UniqueArray<Integer> _waitSomeRequests(ArrayView<Request> requests, bool is_non_blocking);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
