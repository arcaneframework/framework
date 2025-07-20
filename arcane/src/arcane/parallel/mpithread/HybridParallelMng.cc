// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* HybridParallelMng.cc                                        (C) 2000-2025 */
/*                                                                           */
/* Gestionnaire de parallélisme utilisant un mixte MPI/Threads.              */
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
#include "arcane/core/internal/ParallelMngInternal.h"
#include "arcane/core/internal/SerializeMessage.h"
#include "arcane/core/parallel/IStat.h"

#include "arcane/parallel/mpithread/HybridParallelDispatch.h"
#include "arcane/parallel/mpithread/HybridMessageQueue.h"
#include "arcane/parallel/mpithread/internal/HybridMachineMemoryWindowBaseCreator.h"
#include "arcane/parallel/mpithread/internal/HybridMachineMemoryWindowBase.h"

#include "arcane/parallel/mpi/MpiParallelMng.h"

#include "arcane/impl/TimerMng.h"
#include "arcane/impl/ParallelReplication.h"
#include "arcane/impl/SequentialParallelMng.h"
#include "arcane/impl/internal/ParallelMngUtilsFactoryBase.h"

#include "arccore/message_passing/Messages.h"
#include "arccore/message_passing/RequestListBase.h"
#include "arccore/message_passing/SerializeMessageList.h"

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

// NOTE: Cette classe n'est plus utilisée. Elle reste pour référence
// et sera supprimée ultérieurement
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
      // Pour l'instant seul le mode bloquant est supporté.
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

  // TODO: gérer la memoire sans faire de new.
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

  explicit Impl(HybridParallelMng* pm, HybridMachineMemoryWindowBaseCreator* window_creator)
  : ParallelMngInternal(pm)
  , m_parallel_mng(pm)
  , m_window_creator(window_creator)
  {}

  ~Impl() override = default;

 public:

  Ref<IMachineMemoryWindowBase> createMachineMemoryWindowBase(Integer nb_elem_local, Integer sizeof_one_elem) override
  {
    return makeRef(m_window_creator->createWindow(m_parallel_mng->commRank(), nb_elem_local, sizeof_one_elem, m_parallel_mng->mpiParallelMng()));
  }

 private:

  HybridParallelMng* m_parallel_mng;
  HybridMachineMemoryWindowBaseCreator* m_window_creator;
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

  // TODO: vérifier que tous les autres HybridParallelMng ont bien
  // le même nombre de rang locaux (m_local_nb_rank)
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
// Classe pour créer les différents dispatchers
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

  // Créé le gestionnaire séquentiel associé.
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

  // Effectue l'envoie en deux phases. Envoie d'abord le nombre d'éléments
  // puis envoie les éléments.
  // TODO: il serait possible de le faire en une fois pour les messages
  // ne dépassant pas une certaine taille.

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
 * \brief Implémentation de IRequestList pour HybridParallelMng.
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
  // ATTENTION: Cette méthode est appelée simultanément par tous les threads
  // partageant cet HybridParallelMng.

  if (kept_ranks.empty())
    ARCANE_FATAL("kept_ranks is empty");
  ARCANE_CHECK_POINTER(m_sub_builder_factory);

  m_trace->info() << "CREATE SUB_PARALLEL_MNG_REF";

  /*
    Il existe plusieurs possibilités:
    1. on réduit juste le nombre de rangs en mémoire partagé pour chaque
    processus MPI -< on créé un HybridParallelMng
    2. On ne garde que le rang maitre de chaque processus MPI -> on créé un MpiParallelMng.
    3. On ne garde que les rangs d'un même processus -> on créé un SharedMemoryParallelMng
    4. On ne garde qu'un seul rang: on crée un MpiSequentialParallelMng.
  */
  // Pour l'instant, on ne supporte que le cas 1 et 2.
  Int32 nb_kept_rank = kept_ranks.size();

  // Détermine le nouveau nombre de rangs locaux par rang MPI.
  
  // Regarde si je suis dans les listes des rangs conservés et si oui
  // détermine mon rang dans le IParallelMng créé
  Int32 first_global_rank_in_this_mpi = m_global_rank - m_local_rank;
  Int32 last_global_rank_in_this_mpi = first_global_rank_in_this_mpi + m_local_nb_rank - 1;
  // Mon nouveau rang local. Négatif si je ne suis pas dans le nouveau communicateur
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

  // Calcule le min, le max et la somme sur tous les rangs du nombre de nouveaux.
  // Deux cas peuvent se présenter:
  // 1. Le min et le max sont égaux et supérieurs ou égaux à 2: Dans ce cas on créé
  //    un HybridParallelMng.
  // 2. Le max vaut 1. Dans ce cas on créé un nouveau IParallelMng via le MpiParallelMng.
  //    Les rangs actuels pour lequels 'new_local_nb_rank' vaut 0 ne seront pas dans ce
  //    nouveau communicateur. Ce cas concerne aussi le cas où il ne reste plus qu'un
  //    seul rang à la fin.

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

  // S'il ne reste qu'un seul rang local, alors on construit uniquement un MpiParallelMng.
  // Seul le PE qui a un nouveau rang est concerné et fait cela
  if (max_new_local_nb_rank==1){
    Integer nb_mpi_rank = m_mpi_parallel_mng->commSize();
    // Il faut calculer les nouveaux rangs MPI.
    // Si 'min_new_local_nb_rank' vaut 1, alors c'est simple car cela signifie qu'on garde
    // tous les rangs MPI actuels (on fait l'équivalent d'un MPI_Comm_dup). Sinon, on
    // récupère pour chaque rang MPI s'il sera dans le nouveau communicateur et on construit
    // la liste des rangs conservés en fonction de cela.
    // NOTE: dans tous les cas il faut faire attention qu'un seul thread utilise le
    // 'm_mpi_parallel_mng'.
    UniqueArray<Int32> kept_mpi_ranks;
    //! Indique cela qui va faire les appels MPI
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
      // Si je ne suis pas dans le nouveau communicateur, c'est le rang local 0 qui
      // faut le 'gather'.
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

  // Met une barrière locale pour être sur que tout le monde attend ici.
  m_thread_barrier->wait();

  // NOTE: Le builder contient les parties communes aux IParallelMng créés. Il faut
  // donc que ces derniers gardent une référence dessus sinon il sera détruit à la fin
  // de cette méthode.
  Ref<IParallelMngContainer> builder;

  // Le rang 0 créé le builder
  if (m_local_rank==0){
    // Suppose qu'on à le même nombre de rangs MPI qu'avant donc on utilise
    // le communicateur MPI qu'on a déjà.
    MP::Communicator c = communicator();
    builder = m_sub_builder_factory->_createParallelMngBuilder(new_local_nb_rank,c);
    // Positionne le builder pour tout le monde
    m_all_dispatchers->m_create_sub_parallel_mng_info.m_builder = builder;
  }
  // Attend pour être sur que tous les threads voit le bon builder.
  m_thread_barrier->wait();

  builder = m_all_dispatchers->m_create_sub_parallel_mng_info.m_builder;
  ARCANE_CHECK_POINTER(builder.get());

  Ref<IParallelMng> new_parallel_mng;
  if (my_new_local_rank>=0){
    new_parallel_mng = builder->_createParallelMng(my_new_local_rank,traceMng());
  }
  m_thread_barrier->wait();

  // Ici, tout le monde a créé son IParallelMng. On peut donc
  // supprimer la référence au builder. Les IParallelMng créés gardent
  // une référence au builder
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
