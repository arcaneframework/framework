// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* HybridParallelMng.cc                                        (C) 2000-2026 */
/*                                                                           */
/* Parallelism manager using a mix of MPI/Threads.                           */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/parallel/mpithread/HybridParallelMng.h"

#include "arcane/utils/NotImplementedException.h"
#include "arcane/utils/NotSupportedException.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/NumericTypes.h"
#include "arcane/utils/ArgumentException.h"
#include "arcane/utils/IThreadBarrier.h"
#include "arcane/utils/ITraceMng.h"

#include "arcane/core/IIOMng.h"
#include "arcane/core/Timer.h"
#include "arcane/core/ISerializeMessageList.h"
#include "arcane/core/IItemFamily.h"
#include "arcane/core/IParallelTopology.h"
#include "arcane/core/internal/ParallelMngInternal.h"
#include "arcane/core/internal/SerializeMessage.h"
#include "arcane/core/internal/MachineShMemWinMemoryAllocator.h"
#include "arcane/core/parallel/IStat.h"

#include "arcane/parallel/mpithread/HybridParallelDispatch.h"
#include "arcane/parallel/mpithread/HybridMessageQueue.h"
#include "arcane/parallel/mpithread/internal/HybridMachineShMemWinBaseInternalCreator.h"
#include "arcane/parallel/mpithread/internal/HybridContigMachineShMemWinBaseInternal.h"
#include "arcane/parallel/mpithread/internal/HybridMachineShMemWinBaseInternal.h"

#include "arcane/parallel/mpi/MpiParallelMng.h"

#include "arcane/impl/TimerMng.h"
#include "arcane/impl/ParallelReplication.h"
#include "arcane/impl/SequentialParallelMng.h"
#include "arcane/impl/internal/ParallelMngUtilsFactoryBase.h"

#include "arccore/message_passing/Messages.h"
#include "arccore/message_passing/RequestListBase.h"
#include "arccore/message_passing/internal/SerializeMessageList.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
extern "C++" IIOMng*
arcaneCreateIOMng(IParallelMng* psm);
}

namespace Arcane::MessagePassing
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// NOTE: This class is no longer used. It remains for reference
// and will be removed later
class HybridSerializeMessageList
: public ISerializeMessageList
{
 public:
  class HybridSerializeMessageRequest
  {
   public:
    HybridSerializeMessageRequest(ISerializeMessage* message,Request request)
    : m_message(message), m_request(request){}
   public:
    ISerializeMessage* m_message = nullptr;
    Request m_request;
  };

 public:

  explicit HybridSerializeMessageList(HybridParallelMng* mpm)
  : m_parallel_mng(mpm), m_trace(mpm->traceMng())
  {
  }

 public:

  void addMessage(ISerializeMessage* msg) override
  {
    m_messages_to_process.add(msg);

  }
  void processPendingMessages() override
  {
  }

  Integer waitMessages(Parallel::eWaitType wait_type) override
  {
    switch(wait_type){
    case Parallel::WaitAll:
      // Currently, only the blocking mode is supported.
      //m_parallel_mng->processMessages(m_messages_to_process);
      _wait(Parallel::WaitAll);
      m_messages_to_process.clear();
      return (-1);
    case Parallel::WaitSome:
      ARCANE_THROW(NotImplementedException,"WaitSome");
    case Parallel::WaitSomeNonBlocking:
      ARCANE_THROW(NotImplementedException,"WaitSomeNonBlocking");
    }
    return (-1);
  }

 private:

  HybridParallelMng* m_parallel_mng;
  ITraceMng* m_trace;
  UniqueArray<ISerializeMessage*> m_messages_to_process;

  void _waitAll();
  void _wait(Parallel::eWaitType wait_mode);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void HybridSerializeMessageList::
_wait(Parallel::eWaitType wait_mode)
{
  m_trace->info() << "BEGIN PROCESS MESSAGES";

  // TODO: manage memory without using new.
  ConstArrayView<ISerializeMessage*> messages = m_messages_to_process;
  HybridMessageQueue* message_queue = m_parallel_mng->m_message_queue;
  UniqueArray<Request> all_requests;
  MessageTag HYBRID_MESSAGE_TAG(511);
  for( ISerializeMessage* sm : messages ){
    ISerializer* s = sm->serializer();
    MessageRank orig(sm->source());
    MessageRank dest(sm->destination());
    PointToPointMessageInfo message_info(orig,dest,HYBRID_MESSAGE_TAG,Parallel::NonBlocking);
    Request r;
    if (sm->isSend())
      r = message_queue->addSend(message_info,SendBufferInfo(s));
    else
      r = message_queue->addReceive(message_info,ReceiveBufferInfo(s));
    all_requests.add(r);
  }

  if (wait_mode==Parallel::WaitAll)
    message_queue->waitAll(all_requests);

  for( ISerializeMessage* sm : messages )
    sm->setFinished(true);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class HybridParallelMng::Impl
: public ParallelMngInternal
{
 public:

  explicit Impl(HybridParallelMng* pm, HybridMachineShMemWinBaseInternalCreator* window_creator)
  : ParallelMngInternal(pm)
  , m_parallel_mng(pm)
  , m_window_creator(window_creator)
  , m_alloc(makeRef(new MachineShMemWinMemoryAllocator(pm)))
  {}

  ~Impl() override = default;

 public:

  Int32 masterParallelIORank() const override
  {
    return m_parallel_mng->m_mpi_parallel_mng->commRank() * m_parallel_mng->m_local_nb_rank;
  }
  Int32 nbSendersToMasterParallelIO() const override
  {
    return m_parallel_mng->m_local_nb_rank;
  }

  void initializeWindowCreator() override
  {
    m_parallel_mng->traceMng()->debug() << "initializeWindowCreator() Hybrid";
    m_window_creator->initializeMpiWindowCreator(m_parallel_mng->commRank(), m_parallel_mng->mpiParallelMng());
  }

  bool isMachineShMemWinAvailable() override
  {
    if (m_shmem_available == 1) {
      return true;
    }

    if (m_shmem_available == 0) {
      Ref<IParallelTopology> topo = m_parallel_mng->_internalUtilsFactory()->createTopology(m_parallel_mng);
      if (topo->machineRanks().size() == m_window_creator->machineRanks().size()) {
        m_shmem_available = 1;
        return true;
      }
      // Issue with MPI. May occur if MPICH is compiled in ch3:sock mode.
      m_shmem_available = 2;
      return false;
    }

    return false;
  }

  Ref<IContigMachineShMemWinBaseInternal> createContigMachineShMemWinBase(Int64 sizeof_segment, Int32 sizeof_type) override
  {
    return makeRef(m_window_creator->createWindow(m_parallel_mng->commRank(), sizeof_segment, sizeof_type, m_parallel_mng->mpiParallelMng()));
  }

  Ref<IMachineShMemWinBaseInternal> createMachineShMemWinBase(Int64 sizeof_segment, Int32 sizeof_type) override
  {
    return makeRef(m_window_creator->createDynamicWindow(m_parallel_mng->commRank(), sizeof_segment, sizeof_type, m_parallel_mng->mpiParallelMng()));
  }

  MemoryAllocationOptions machineShMemWinMemoryAllocator() override
  {
    return MemoryAllocationOptions{ m_alloc.get() };
  }

  ConstArrayView<Int32> machineRanks() override
  {
    return m_window_creator->machineRanks();
  }

  void machineBarrier() override
  {
    m_window_creator->machineBarrier(m_parallel_mng->commRank(), m_parallel_mng->mpiParallelMng());
  }

 private:

  HybridParallelMng* m_parallel_mng;
  HybridMachineShMemWinBaseInternalCreator* m_window_creator;
  Ref<MachineShMemWinMemoryAllocator> m_alloc;

  // 0 = Not initialized attribute
  // 1 = Shared memory available
  // 2 = Shared memory not available
  Int8 m_shmem_available = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

HybridParallelMng::
HybridParallelMng(const HybridParallelMngBuildInfo& bi)
: ParallelMngDispatcher(ParallelMngDispatcherBuildInfo(bi.local_rank,bi.local_nb_rank))
, m_trace(bi.trace_mng)
, m_thread_mng(bi.thread_mng)
, m_world_parallel_mng(bi.world_parallel_mng)
, m_io_mng(nullptr)
, m_timer_mng(nullptr)
, m_replication(new ParallelReplication())
, m_message_queue(new HybridMessageQueue(bi.message_queue,bi.mpi_parallel_mng,bi.local_nb_rank))
, m_is_initialized(false)
, m_stat(Parallel::createDefaultStat())
, m_thread_barrier(bi.thread_barrier)
, m_mpi_parallel_mng(bi.mpi_parallel_mng)
, m_all_dispatchers(bi.all_dispatchers)
, m_sub_builder_factory(bi.sub_builder_factory)
, m_parent_container_ref(bi.container)
, m_utils_factory(createRef<ParallelMngUtilsFactoryBase>())
, m_parallel_mng_internal(new Impl(this, bi.window_creator))
{
  if (!m_world_parallel_mng)
    m_world_parallel_mng = this;

  // TODO: verify that all other HybridParallelMng have the same
  // number of local ranks (m_local_nb_rank)
  m_local_rank = bi.local_rank;
  m_local_nb_rank = bi.local_nb_rank;

  Int32 mpi_rank = m_mpi_parallel_mng->commRank();
  Int32 mpi_size = m_mpi_parallel_mng->commSize();

  m_global_rank = m_local_rank + mpi_rank * m_local_nb_rank;
  m_global_nb_rank = mpi_size * m_local_nb_rank;

  m_is_parallel = m_global_nb_rank!=1;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

HybridParallelMng::
~HybridParallelMng()
{
  m_sequential_parallel_mng.reset();
  delete m_replication;
  delete m_io_mng;
  delete m_message_queue;
  delete m_timer_mng;
  delete m_stat;
  delete m_mpi_parallel_mng;
  delete m_parallel_mng_internal;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{
// Class to create the different dispatchers
class DispatchCreator
{
 public:
  DispatchCreator(ITraceMng* tm,HybridParallelMng* mpm,HybridMessageQueue* message_queue,MpiThreadAllDispatcher* all_dispatchers)
  : m_tm(tm), m_mpm(mpm), m_message_queue(message_queue), m_all_dispatchers(all_dispatchers){}
 public:
  template<typename DataType> HybridParallelDispatch<DataType>*
  create()
  {
    HybridMessageQueue* tmq = m_message_queue;
    MpiThreadAllDispatcher* ad = m_all_dispatchers;
    auto field = ad->instance((DataType*)nullptr).view();
    return new HybridParallelDispatch<DataType>(m_tm,m_mpm,tmq,field);
  }

  ITraceMng* m_tm;
  HybridParallelMng* m_mpm;
  HybridMessageQueue* m_message_queue;
  MpiThreadAllDispatcher* m_all_dispatchers;
};
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void HybridParallelMng::
build()
{
  ITraceMng* tm = traceMng();
  tm->info() << "Initialise HybridParallelMng"
             << " global_rank=" << m_global_rank
             << " local_rank=" << m_local_rank
             << " mpi_rank=" << m_mpi_parallel_mng->commRank();

  m_timer_mng = new TimerMng(tm);

  // Created the associated sequential manager.
  {
    SequentialParallelMngBuildInfo bi(timerMng(),worldParallelMng());
    bi.setTraceMng(traceMng());
    bi.setCommunicator(communicator());
    bi.setThreadMng(threadMng());
    m_sequential_parallel_mng = arcaneCreateSequentialParallelMngRef(bi);
  }

  DispatchCreator creator(m_trace,this,m_message_queue,m_all_dispatchers);
  this->createDispatchers(creator);
  m_io_mng = arcaneCreateIOMng(this);

  m_parallel_mng_internal->initializeWindowCreator();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*----------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void HybridParallelMng::
initialize()
{
  Trace::Setter mci(m_trace,"Thread");
  if (m_is_initialized){
    m_trace->warning() << "HybridParallelMng already initialized";
    return;
  }

  m_is_initialized = true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

SerializeBuffer* HybridParallelMng::
_castSerializer(ISerializer* serializer)
{
  auto sbuf = dynamic_cast<SerializeBuffer*>(serializer);
  if (!sbuf)
    ARCANE_THROW(ArgumentException,"can not cast 'ISerializer' to 'SerializeBuffer'");
  return sbuf;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IGetVariablesValuesParallelOperation*  HybridParallelMng::
createGetVariablesValuesOperation()
{
  return m_utils_factory->createGetVariablesValuesOperation(this)._release();
}

ITransferValuesParallelOperation* HybridParallelMng::
createTransferValuesOperation()
{
  return m_utils_factory->createTransferValuesOperation(this)._release();
}

IParallelExchanger* HybridParallelMng::
createExchanger()
{
  return m_utils_factory->createExchanger(this)._release();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void HybridParallelMng::
sendSerializer(ISerializer* s,Int32 rank)
{
  auto p2p_message = buildMessage(rank,Parallel::NonBlocking);
  Request r = m_message_queue->addSend(p2p_message,s);
  m_message_queue->waitAll(ArrayView<Request>(1,&r));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

auto HybridParallelMng::
sendSerializer(ISerializer* s,Int32 rank,ByteArray& bytes) -> Request
{
  ARCANE_UNUSED(bytes);
  auto p2p_message = buildMessage(rank,Parallel::NonBlocking);
  return m_message_queue->addSend(p2p_message,s);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ISerializeMessage* HybridParallelMng::
createSendSerializer(Int32 rank)
{
  return m_utils_factory->createSendSerializeMessage(this, rank)._release();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void HybridParallelMng::
broadcastSerializer(ISerializer* values,Int32 rank)
{
  Timer::Phase tphase(timeStats(),TP_Communication);
  SerializeBuffer* sbuf = _castSerializer(values);

  bool is_broadcaster = (rank==commRank());

  // Send in two phases. First send the number of elements
  // then send the elements.
  // TODO: it would be possible to do it in one go for messages
  // not exceeding a certain size.

  IMessagePassingMng* mpm = this->messagePassingMng();
  if (is_broadcaster){
    Int64 total_size = sbuf->totalSize();
    Span<Byte> bytes = sbuf->globalBuffer();
    this->broadcast(Int64ArrayView(1,&total_size),rank);
    mpBroadcast(mpm,bytes,rank);
  }
  else{
    Int64 total_size = 0;
    this->broadcast(Int64ArrayView(1,&total_size),rank);
    sbuf->preallocate(total_size);
    Span<Byte> bytes = sbuf->globalBuffer();
    mpBroadcast(mpm,bytes,rank);
    sbuf->setFromSizes();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void HybridParallelMng::
recvSerializer(ISerializer* s,Int32 rank)
{
  auto p2p_message = buildMessage(rank,Parallel::NonBlocking);
  Request r = m_message_queue->addReceive(p2p_message,ReceiveBufferInfo(s));
  m_message_queue->waitAll(ArrayView<Request>(1,&r));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ISerializeMessage* HybridParallelMng::
createReceiveSerializer(Int32 rank)
{
  return m_utils_factory->createReceiveSerializeMessage(this, rank)._release();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void HybridParallelMng::
freeRequests(ArrayView<Request> requests)
{
  ARCANE_UNUSED(requests);
  throw NotImplementedException(A_FUNCINFO);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MessageId HybridParallelMng::
probe(const PointToPointMessageInfo& message)
{
  PointToPointMessageInfo p2p_message(message);
  p2p_message.setEmiterRank(MessageRank(m_global_rank));
  return m_message_queue->probe(p2p_message);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MessageSourceInfo HybridParallelMng::
legacyProbe(const PointToPointMessageInfo& message)
{
  PointToPointMessageInfo p2p_message(message);
  p2p_message.setEmiterRank(MessageRank(m_global_rank));
  return m_message_queue->legacyProbe(p2p_message);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Request HybridParallelMng::
sendSerializer(const ISerializer* s,const PointToPointMessageInfo& message)
{
  auto p2p_message = buildMessage(message);
  return m_message_queue->addSend(p2p_message,s);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Request HybridParallelMng::
receiveSerializer(ISerializer* s,const PointToPointMessageInfo& message)
{
  auto p2p_message = buildMessage(message);
  return m_message_queue->addReceive(p2p_message,ReceiveBufferInfo(s));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void HybridParallelMng::
printStats()
{
  if (m_stat)
    m_stat->print(m_trace);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void HybridParallelMng::
barrier()
{
  m_thread_barrier->wait();
  if (m_local_rank==0)
    m_mpi_parallel_mng->barrier();
  m_thread_barrier->wait();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ISerializeMessageList* HybridParallelMng::
_createSerializeMessageList()
{
  auto* x = new MP::internal::SerializeMessageList(messagePassingMng());
  x->setAllowAnyRankReceive(false);
  return x;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IVariableSynchronizer* HybridParallelMng::
createSynchronizer(IItemFamily* family)
{
  return m_utils_factory->createSynchronizer(this,family)._release();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IVariableSynchronizer* HybridParallelMng::
createSynchronizer(const ItemGroup& group)
{
  return m_utils_factory->createSynchronizer(this,group)._release();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IParallelTopology* HybridParallelMng::
createTopology()
{
  return m_utils_factory->createTopology(this)._release();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IParallelReplication* HybridParallelMng::
replication() const
{
  return m_replication;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void HybridParallelMng::
setReplication(IParallelReplication* v)
{
  delete m_replication;
  m_replication = v;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IParallelMng* HybridParallelMng::
sequentialParallelMng()
{
  return m_sequential_parallel_mng.get();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Ref<IParallelMng> HybridParallelMng::
sequentialParallelMngRef()
{
  return m_sequential_parallel_mng;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Implementation of IRequestList for HybridParallelMng.
 */
class HybridParallelMng::RequestList
: public Arccore::MessagePassing::internal::RequestListBase
{
  using Base = Arccore::MessagePassing::internal::RequestListBase;
 public:
  RequestList(HybridParallelMng* pm)
  : m_parallel_mng(pm), m_message_queue(pm->m_message_queue),
    m_local_rank(m_parallel_mng->localRank()) {}
 public:
  void _wait(Parallel::eWaitType wait_type) override
  {
    switch(wait_type){
    case Parallel::WaitAll:
      m_parallel_mng->m_message_queue->waitAll(_requests());
      break;
    case Parallel::WaitSome:
      m_message_queue->waitSome(m_local_rank,_requests(),_requestsDone(),false);
      break;
    case Parallel::WaitSomeNonBlocking:
      m_message_queue->waitSome(m_local_rank,_requests(),_requestsDone(),true);
    }
  }
 private:
  HybridParallelMng* m_parallel_mng;
  HybridMessageQueue* m_message_queue;
  Int32 m_local_rank;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Ref<Parallel::IRequestList> HybridParallelMng::
createRequestListRef()
{
  Parallel::IRequestList* r = new RequestList(this);
  return makeRef(r);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void HybridParallelMng::
waitAllRequests(ArrayView<Request> requests)
{
  m_message_queue->waitAll(requests);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void* HybridParallelMng::
getMPICommunicator()
{
  return m_mpi_parallel_mng->getMPICommunicator();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MP::Communicator HybridParallelMng::
communicator() const
{
  return m_mpi_parallel_mng->communicator();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MP::Communicator HybridParallelMng::
machineCommunicator() const
{
  return m_mpi_parallel_mng->machineCommunicator();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

PointToPointMessageInfo HybridParallelMng::
buildMessage(const PointToPointMessageInfo& message)
{
  PointToPointMessageInfo p2p_message(message);
  p2p_message.setEmiterRank(MessageRank(m_global_rank));
  return p2p_message;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

PointToPointMessageInfo HybridParallelMng::
buildMessage(Int32 dest,Parallel::eBlockingType blocking_mode)
{
  return buildMessage({MessageRank(dest),blocking_mode});
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IParallelMng* HybridParallelMng::
_createSubParallelMng(Int32ConstArrayView kept_ranks)
{
  ARCANE_UNUSED(kept_ranks);
  ARCANE_THROW(NotSupportedException,"Use createSubParallelMngRef() instead");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Ref<IParallelMng> HybridParallelMng::
createSubParallelMngRef(Int32ConstArrayView kept_ranks)
{
  // ATTENTION: This method is called simultaneously by all threads
  // sharing this HybridParallelMng.

  if (kept_ranks.empty())
    ARCANE_FATAL("kept_ranks is empty");
  ARCANE_CHECK_POINTER(m_sub_builder_factory);

  m_trace->info() << "CREATE SUB_PARALLEL_MNG_REF";

  /*
    There are several possibilities:
    1. We just reduce the number of ranks in shared memory for each
    MPI process - we create a HybridParallelMng
    2. We only keep the master rank of each MPI process -> we create an MpiParallelMng.
    3. We only keep the ranks of the same process -> we create a SharedMemoryParallelMng
    4. We only keep a single rank: we create an MpiSequentialParallelMng.
  */
  // For now, we only support cases 1 and 2.
  Int32 nb_kept_rank = kept_ranks.size();

  // Determines the new number of local ranks per MPI rank.

  // Checks if I am in the list of kept ranks and, if so,
  // determines my rank in the created IParallelMng
  Int32 first_global_rank_in_this_mpi = m_global_rank - m_local_rank;
  Int32 last_global_rank_in_this_mpi = first_global_rank_in_this_mpi + m_local_nb_rank - 1;
  // My new global rank. Negative if I am not in the new communicator
  Int32 my_new_global_rank = (-1);
  Int32 new_local_nb_rank = 0;
  Int32 my_new_local_rank = (-1);
  for( Integer i=0; i<nb_kept_rank; ++i ){
    Int32 kept_rank = kept_ranks[i];
    if (kept_rank>=first_global_rank_in_this_mpi && kept_rank<last_global_rank_in_this_mpi)
      ++new_local_nb_rank;
    if (kept_rank==m_global_rank){
      my_new_global_rank = i;
      my_new_local_rank = new_local_nb_rank - 1;
    }
  }
  bool has_new_rank = (my_new_global_rank != (-1));

  // Calculates the min, max, and sum of the new number.
  // Two cases can occur:
  // 1. The min and max are equal and greater than or equal to 2: In this case, we create
  //    a HybridParallelMng.
  // 2. The max is 1. In this case, we create a new IParallelMng via the MpiParallelMng.
  //    The current ranks for which 'new_local_nb_rank' is 0 will not be in this
  //    new communicator. This case also applies when only one rank remains
  //    at the end.

  Int32 min_new_local_nb_rank = -1;
  Int32 max_new_local_nb_rank = -1;
  Int32 sum_new_local_nb_rank = -1;
  Int32 min_rank = A_NULL_RANK;
  Int32 max_rank = A_NULL_RANK;
  computeMinMaxSum(new_local_nb_rank,min_new_local_nb_rank,max_new_local_nb_rank,
                   sum_new_local_nb_rank,min_rank,max_rank);

  m_trace->info() << "CREATE SUB_PARALLEL_MNG_REF new_local_nb_rank=" << new_local_nb_rank
                  << " min=" << min_new_local_nb_rank
                  << " max=" << max_new_local_nb_rank
                  << " sum=" << sum_new_local_nb_rank
                  << " new_global_rank=" << my_new_global_rank;

  // If only one local rank remains, then we only build an MpiParallelMng.
  // Only the PE that has a new rank is concerned and does this
  if (max_new_local_nb_rank==1){
    Integer nb_mpi_rank = m_mpi_parallel_mng->commSize();
    // We must calculate the new MPI ranks.
    // If 'min_new_local_nb_rank' is 1, it's simple because it means we keep
    // all current MPI ranks (equivalent to an MPI_Comm_dup). Otherwise, we
    // retrieve for each MPI rank whether it will be in the new communicator and
    // build the list of kept ranks based on that.
    // NOTE: in all cases, we must ensure that only one thread uses the
    // 'm_mpi_parallel_mng'.
    UniqueArray<Int32> kept_mpi_ranks;
    //! Indicates who will make the MPI calls
    bool do_mpi_call = false;
    if (min_new_local_nb_rank==1){
      if (has_new_rank){
        do_mpi_call = true;
        kept_mpi_ranks.resize(nb_mpi_rank);
        for( Int32 x=0; x<nb_mpi_rank; ++x )
          kept_mpi_ranks[x] = x;
      }
    }
    else{
      // If I am not in the new communicator, local rank 0 must
      // 'gather'.
      UniqueArray<Int16> gathered_ranks(nb_mpi_rank);
      if (has_new_rank || m_local_rank==0){
        do_mpi_call = true;
        Int16 v = (has_new_rank) ? 1 : 0;
        m_mpi_parallel_mng->allGather(ArrayView<Int16>(1,&v),gathered_ranks);
      }
      for( Int32 x=0; x<nb_mpi_rank; ++x )
        if (gathered_ranks[x]==1)
          kept_mpi_ranks.add(x);
    }
    if (do_mpi_call)
      return m_mpi_parallel_mng->createSubParallelMngRef(kept_mpi_ranks);
    else
      return Ref<IParallelMng>();
  }

  if (max_new_local_nb_rank!=new_local_nb_rank)
    ARCANE_FATAL("Not same number of new local ranks on every MPI processus: current={0} max={1}",
                 new_local_nb_rank,max_new_local_nb_rank);

  if (max_new_local_nb_rank<2)
    ARCANE_FATAL("number of local ranks is too low current={0} minimum=2",new_local_nb_rank);

  // Wait a local barrier to ensure everyone waits here.
  m_thread_barrier->wait();

  // NOTE: The builder contains the common parts of the created IParallelMngs. It must
  // therefore be referenced by them, otherwise it will be destroyed at the end
  // of this method.
  Ref<IParallelMngContainer> builder;

  // Rank 0 creates the builder
  if (m_local_rank==0){
    // Assuming we have the same number of MPI ranks as before, we use
    // the MPI communicator we already have.
    MP::Communicator c = communicator();
    MP::Communicator mc = machineCommunicator();
    builder = m_sub_builder_factory->_createParallelMngBuilder(new_local_nb_rank, c, mc);
    // Position the builder for everyone
    m_all_dispatchers->m_create_sub_parallel_mng_info.m_builder = builder;
  }
  // Wait to ensure all threads see the correct builder.
  m_thread_barrier->wait();

  builder = m_all_dispatchers->m_create_sub_parallel_mng_info.m_builder;
  ARCANE_CHECK_POINTER(builder.get());

  Ref<IParallelMng> new_parallel_mng;
  if (my_new_local_rank>=0){
    new_parallel_mng = builder->_createParallelMng(my_new_local_rank,traceMng());
  }
  m_thread_barrier->wait();

  // Here, everyone has created their IParallelMng. We can therefore
  // release the reference to the builder. The created IParallelMngs keep
  // a reference to the builder
  if (m_local_rank==0){
    m_all_dispatchers->m_create_sub_parallel_mng_info.m_builder.reset();
  }
  m_thread_barrier->wait();

  return new_parallel_mng;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Ref<IParallelMngUtilsFactory> HybridParallelMng::
_internalUtilsFactory() const
{
  return m_utils_factory;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool HybridParallelMng::
_isAcceleratorAware() const
{
  return m_mpi_parallel_mng->_internalApi()->isAcceleratorAware();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::MessagePassing

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
