// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SharedMemoryParallelMng.cc                                  (C) 2000-2026 */
/*                                                                           */
/* Implémentation des messages en mode mémoire partagé.                      */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/parallel/thread/SharedMemoryParallelMng.h"

#include "arcane/utils/NotImplementedException.h"
#include "arcane/utils/NotSupportedException.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/TraceInfo.h"
#include "arcane/utils/NumericTypes.h"
#include "arcane/utils/ArgumentException.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/ITraceMng.h"

#include "arcane/parallel/IStat.h"

#include "arcane/parallel/thread/SharedMemoryParallelDispatch.h"
#include "arcane/parallel/thread/ISharedMemoryMessageQueue.h"
#include "arcane/parallel/thread/internal/SharedMemoryMachineMemoryWindowBaseInternalCreator.h"
#include "arcane/parallel/thread/internal/SharedMemoryMachineMemoryWindowBaseInternal.h"
#include "arcane/parallel/thread/internal/SharedMemoryDynamicMachineMemoryWindowBaseInternal.h"

#include "arcane/core/Timer.h"
#include "arcane/core/IIOMng.h"
#include "arcane/core/ISerializeMessageList.h"
#include "arcane/core/IItemFamily.h"
#include "arcane/core/internal/SerializeMessage.h"
#include "arcane/core/internal/ParallelMngInternal.h"
#include "arcane/core/internal/DynamicMachineMemoryWindowMemoryAllocator.h"

#include "arcane/impl/TimerMng.h"
#include "arcane/impl/ParallelReplication.h"
#include "arcane/impl/internal/ParallelMngUtilsFactoryBase.h"

#include "arccore/message_passing/RequestListBase.h"
#include "arccore/message_passing/internal/SerializeMessageList.h"

#include <map>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
extern "C++" ARCANE_IMPL_EXPORT IIOMng*
arcaneCreateIOMng(IParallelMng* psm);
}

namespace Arcane::MessagePassing
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Implémentation de IRequestList pour SharedMemoryParallelMng.
 */
class SharedMemoryParallelMng::RequestList
: public Arccore::MessagePassing::internal::RequestListBase
{
  using Base = Arccore::MessagePassing::internal::RequestListBase;
 public:

  explicit RequestList(SharedMemoryParallelMng* pm)
  : m_parallel_mng(pm), m_message_queue(pm->m_message_queue),
    m_local_rank(m_parallel_mng->commRank()){}
 public:
  void _wait(Parallel::eWaitType wait_type) override
  {
    switch(wait_type){
    case Parallel::WaitAll:
      return m_message_queue->waitAll(_requests());
    case Parallel::WaitSome:
      return m_message_queue->waitSome(m_local_rank,_requests(),_requestsDone(),false);
    case Parallel::WaitSomeNonBlocking:
      return m_message_queue->waitSome(m_local_rank,_requests(),_requestsDone(),true);
    }
  }
 private:
  SharedMemoryParallelMng* m_parallel_mng;
  ISharedMemoryMessageQueue* m_message_queue;
  Int32 m_local_rank;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class SharedMemoryParallelMng::Impl
: public ParallelMngInternal
{
 public:

  explicit Impl(SharedMemoryParallelMng* pm, SharedMemoryMachineMemoryWindowBaseInternalCreator* window_creator)
  : ParallelMngInternal(pm)
  , m_parallel_mng(pm)
  , m_window_creator(window_creator)
  , m_alloc(makeRef(new DynamicMachineMemoryWindowMemoryAllocator(pm)))
  {}

  ~Impl() override = default;

 public:

  Ref<IMachineMemoryWindowBaseInternal> createMachineMemoryWindowBase(Int64 sizeof_segment, Int32 sizeof_type) override
  {
    return makeRef(m_window_creator->createWindow(m_parallel_mng->commRank(), sizeof_segment, sizeof_type));
  }

  Ref<IDynamicMachineMemoryWindowBaseInternal> createDynamicMachineMemoryWindowBase(Int64 sizeof_segment, Int32 sizeof_type) override
  {
    return makeRef(m_window_creator->createDynamicWindow(m_parallel_mng->commRank(), sizeof_segment, sizeof_type));
  }

  IMemoryAllocator* dynamicMachineMemoryWindowMemoryAllocator() override
  {
    return m_alloc.get();
  }

 private:

  SharedMemoryParallelMng* m_parallel_mng;
  SharedMemoryMachineMemoryWindowBaseInternalCreator* m_window_creator;
  Ref<DynamicMachineMemoryWindowMemoryAllocator> m_alloc;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

SharedMemoryParallelMng::
SharedMemoryParallelMng(const SharedMemoryParallelMngBuildInfo& build_info)
: ParallelMngDispatcher(ParallelMngDispatcherBuildInfo(build_info.rank,build_info.nb_rank))
, m_trace(build_info.trace_mng)
, m_thread_mng(build_info.thread_mng)
, m_sequential_parallel_mng(build_info.sequential_parallel_mng)
, m_timer_mng(nullptr)
, m_replication(new ParallelReplication())
, m_world_parallel_mng(build_info.world_parallel_mng)
, m_io_mng(nullptr)
, m_message_queue(build_info.message_queue)
, m_is_parallel(build_info.nb_rank!=1)
, m_rank(build_info.rank)
, m_nb_rank(build_info.nb_rank)
, m_is_initialized(false)
, m_stat(Parallel::createDefaultStat())
, m_thread_barrier(build_info.thread_barrier)
, m_all_dispatchers(build_info.all_dispatchers)
, m_sub_builder_factory(build_info.sub_builder_factory)
, m_parent_container_ref(build_info.container)
, m_mpi_communicator(build_info.communicator)
, m_utils_factory(createRef<ParallelMngUtilsFactoryBase>())
, m_parallel_mng_internal(new Impl(this, build_info.window_creator))
{
  if (!m_world_parallel_mng)
    m_world_parallel_mng = this;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

SharedMemoryParallelMng::
~SharedMemoryParallelMng()
{
  delete m_parallel_mng_internal;
  delete m_replication;
  m_sequential_parallel_mng.reset();
  delete m_io_mng;
  delete m_timer_mng;
  delete m_stat;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{
// Classe pour créer les différents dispatchers
class DispatchCreator
{
 public:
  DispatchCreator(ITraceMng* tm,SharedMemoryParallelMng* mpm,
                  ISharedMemoryMessageQueue* message_queue,
                  SharedMemoryAllDispatcher* all_dispatchers)
  : m_tm(tm), m_mpm(mpm), m_message_queue(message_queue),
    m_all_dispatchers(all_dispatchers){}
 public:
  template<typename DataType> SharedMemoryParallelDispatch<DataType>*
  create()
  {
    ISharedMemoryMessageQueue* tmq = m_message_queue;
    SharedMemoryAllDispatcher* ad = m_all_dispatchers;
    auto& field = ad->instance((DataType*)nullptr);
    return new SharedMemoryParallelDispatch<DataType>(m_tm,m_mpm,tmq,field);
  }

  ITraceMng* m_tm;
  SharedMemoryParallelMng* m_mpm;
  ISharedMemoryMessageQueue* m_message_queue;
  SharedMemoryAllDispatcher* m_all_dispatchers;
};
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SharedMemoryParallelMng::
build()
{
  m_message_queue->setTraceMng(m_rank,traceMng());
  m_timer_mng = new TimerMng(traceMng());

  DispatchCreator creator(m_trace.get(),this,m_message_queue,m_all_dispatchers);
  this->createDispatchers(creator);

  m_io_mng = arcaneCreateIOMng(this);
}

/*----------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SharedMemoryParallelMng::
initialize()
{
  Trace::Setter mci(m_trace.get(),"Thread");
  if (m_is_initialized){
    m_trace->warning() << "SharedMemoryParallelMng already initialized";
    return;
  }
	
  m_is_initialized = true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IGetVariablesValuesParallelOperation* SharedMemoryParallelMng::
createGetVariablesValuesOperation()
{
  return m_utils_factory->createGetVariablesValuesOperation(this)._release();
}

ITransferValuesParallelOperation* SharedMemoryParallelMng::
createTransferValuesOperation()
{
  return m_utils_factory->createTransferValuesOperation(this)._release();
}

IParallelExchanger* SharedMemoryParallelMng::
createExchanger()
{
  return m_utils_factory->createExchanger(this)._release();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SharedMemoryParallelMng::
sendSerializer(ISerializer* values,Int32 dest_rank)
{
  auto p2p_message = buildMessage(dest_rank,Parallel::Blocking);
  Request r = m_message_queue->addSend(p2p_message,SendBufferInfo(values));
  m_message_queue->waitAll(ArrayView<Request>(1,&r));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Parallel::Request SharedMemoryParallelMng::
sendSerializer(ISerializer* values,Int32 rank,ByteArray& bytes)
{
  ARCANE_UNUSED(bytes);
  return m_message_queue->addSend(buildMessage(rank,Parallel::Blocking),SendBufferInfo(values));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ISerializeMessage* SharedMemoryParallelMng::
createSendSerializer(Int32 rank)
{
  return m_utils_factory->createSendSerializeMessage(this, rank)._release();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SharedMemoryParallelMng::
broadcastSerializer(ISerializer* values,Int32 rank)
{
  // Implementation basique pour l'instant.
  // Le rank qui broadcast envoie le message à tout le monde.
  if (m_rank==rank){
    UniqueArray<Parallel::Request> requests;
    for( Int32 i=0; i<m_nb_rank; ++i ){
      if (i!=m_rank){
        requests.add(m_message_queue->addSend(buildMessage(i,Parallel::NonBlocking),SendBufferInfo(values)));
      }
    }
    m_message_queue->waitAll(requests);
  }
  else{
    recvSerializer(values,rank);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SharedMemoryParallelMng::
recvSerializer(ISerializer* values,Int32 rank)
{
  auto p2p_message = buildMessage(rank,Parallel::Blocking);
  Request r = m_message_queue->addReceive(p2p_message,ReceiveBufferInfo(values));
  m_message_queue->waitAll(ArrayView<Request>(1,&r));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ISerializeMessage* SharedMemoryParallelMng::
createReceiveSerializer(Int32 rank)
{
  return m_utils_factory->createReceiveSerializeMessage(this, rank)._release();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SharedMemoryParallelMng::
freeRequests(ArrayView<Parallel::Request> requests)
{
  ARCANE_UNUSED(requests);
  throw NotImplementedException(A_FUNCINFO);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SharedMemoryParallelMng::
printStats()
{
  if (m_stat)
    m_stat->print(m_trace.get());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SharedMemoryParallelMng::
barrier()
{
  m_thread_barrier->wait();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SharedMemoryParallelMng::
waitAllRequests(ArrayView<Request> requests)
{
  Real begin_time = platform::getRealTime();
  m_message_queue->waitAll(requests);
  Real end_time = platform::getRealTime();
  m_stat->add("WaitAll",(end_time-begin_time),0);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ISerializeMessageList* SharedMemoryParallelMng::
_createSerializeMessageList()
{
  return new MP::internal::SerializeMessageList(messagePassingMng());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MessageId SharedMemoryParallelMng::
probe(const PointToPointMessageInfo& message)
{
  PointToPointMessageInfo p2p_message(message);
  p2p_message.setEmiterRank(MessageRank(m_rank));
  return m_message_queue->probe(p2p_message);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MessageSourceInfo SharedMemoryParallelMng::
legacyProbe(const PointToPointMessageInfo& message)
{
  PointToPointMessageInfo p2p_message(message);
  p2p_message.setEmiterRank(MessageRank(m_rank));
  return m_message_queue->legacyProbe(p2p_message);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

auto SharedMemoryParallelMng::
sendSerializer(const ISerializer* values,const PointToPointMessageInfo& message) -> Request
{
  auto p2p_message = buildMessage(message);
  return m_message_queue->addSend(p2p_message,SendBufferInfo(values));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

auto SharedMemoryParallelMng::
receiveSerializer(ISerializer* values,const PointToPointMessageInfo& message) -> Request
{
  auto p2p_message = buildMessage(message);
  return m_message_queue->addReceive(p2p_message,ReceiveBufferInfo(values));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IVariableSynchronizer* SharedMemoryParallelMng::
createSynchronizer(IItemFamily* family)
{
  return m_utils_factory->createSynchronizer(this,family)._release();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IVariableSynchronizer* SharedMemoryParallelMng::
createSynchronizer(const ItemGroup& group)
{
  return m_utils_factory->createSynchronizer(this,group)._release();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IParallelTopology* SharedMemoryParallelMng::
createTopology()
{
  return m_utils_factory->createTopology(this)._release();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IParallelReplication* SharedMemoryParallelMng::
replication() const
{
  return m_replication;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SharedMemoryParallelMng::
setReplication(IParallelReplication* v)
{
  delete m_replication;
  m_replication = v;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IParallelMng* SharedMemoryParallelMng::
_createSubParallelMng(Int32ConstArrayView kept_ranks)
{
  ARCANE_UNUSED(kept_ranks);
  // On ne peut pas implémenter cette méthode car on passe par
  // IParallelMngContainer::_createParallelMng() qui créé obligatoirement
  // un 'Ref<IParallelMng>'.
  ARCANE_THROW(NotSupportedException,"Use createSubParallelMngRef() instead");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Ref<IParallelMng> SharedMemoryParallelMng::
createSubParallelMngRef(Int32ConstArrayView kept_ranks)
{
  if (kept_ranks.empty())
    ARCANE_FATAL("kept_ranks is empty");
  ARCANE_CHECK_POINTER(m_sub_builder_factory);

  Ref<IParallelMngContainer> builder;
  Int32 nb_rank = kept_ranks.size();

  // Regarde si je suis dans les listes des rangs conservés et si oui
  // détermine mon rang dans le IParallelMng créé
  Int32 my_new_rank = (-1);
  for( Integer i=0; i<nb_rank; ++i )
    if (kept_ranks[i]==m_rank){
      my_new_rank = i;
      break;
    }

  barrier();
  // Le rang 0 créé le builder
  if (m_rank==0){
    builder = m_sub_builder_factory->_createParallelMngBuilder(nb_rank, m_mpi_communicator, m_mpi_communicator);
    // Positionne le builder pour tout le monde
    m_all_dispatchers->m_create_sub_parallel_mng_info.m_builder = builder;
  }
  barrier();

  builder = m_all_dispatchers->m_create_sub_parallel_mng_info.m_builder;
  ARCANE_CHECK_POINTER(builder.get());

  Ref<IParallelMng> new_parallel_mng;
  if (my_new_rank>=0){
    new_parallel_mng = builder->_createParallelMng(my_new_rank,traceMng());
    //auto* new_sm = dynamic_cast<SharedMemoryParallelMng*>(new_parallel_mng.get());
    //if (new_sm)
    //new_sm->m_mpi_communicator = m_mpi_communicator;
  }
  barrier();
  // Ici, tout le monde a créé son IParallelMng. On peut donc
  // supprimer la référence au builder.
  // TODO: il faudra ajouter un compteur de référence sur le builder
  // sinon il ne sera jamais détruit.
  if (m_rank==0){
    m_all_dispatchers->m_create_sub_parallel_mng_info.m_builder.reset();
  }
  barrier();

  return new_parallel_mng;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Ref<Parallel::IRequestList> SharedMemoryParallelMng::
createRequestListRef()
{
  IRequestList* r = new RequestList(this);
  return makeRef(r);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Ref<IParallelMng> SharedMemoryParallelMng::
sequentialParallelMngRef()
{
  return m_sequential_parallel_mng;
}

IParallelMng* SharedMemoryParallelMng::
sequentialParallelMng()
{
  return m_sequential_parallel_mng.get();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

PointToPointMessageInfo SharedMemoryParallelMng::
buildMessage(const PointToPointMessageInfo& orig_message)
{
  PointToPointMessageInfo p2p_message{orig_message};
  p2p_message.setEmiterRank(MessageRank(m_rank));
  return p2p_message;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

PointToPointMessageInfo SharedMemoryParallelMng::
buildMessage(Int32 dest,Parallel::eBlockingType blocking_mode)
{
  return buildMessage({MessageRank(dest),blocking_mode});
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Ref<IParallelMngUtilsFactory> SharedMemoryParallelMng::
_internalUtilsFactory() const
{
  return m_utils_factory;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::MessagePassing

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
