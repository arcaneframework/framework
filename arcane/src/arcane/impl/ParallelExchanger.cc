// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ParallelExchanger.cc                                        (C) 2000-2025 */
/*                                                                           */
/* Echange d'informations entre processeurs.                                 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/impl/ParallelExchanger.h"

#include "arcane/utils/NotSupportedException.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/PlatformUtils.h"

#include "arcane/core/MathUtils.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/SerializeBuffer.h"
#include "arcane/core/Timer.h"
#include "arcane/core/ISerializeMessageList.h"
#include "arcane/core/internal/SerializeMessage.h"

#include <algorithm>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ParallelExchanger::
ParallelExchanger(IParallelMng* pm)
: ParallelExchanger(makeRef(pm))
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ParallelExchanger::
ParallelExchanger(Ref<IParallelMng> pm)
: TraceAccessor(pm->traceMng())
, m_parallel_mng(pm)
, m_timer(pm->timerMng(),"ParallelExchangerTimer",Timer::TimerReal)
{
  String use_collective_str = platform::getEnvironmentVariable("ARCANE_PARALLEL_EXCHANGER_USE_COLLECTIVE");
  if (use_collective_str=="1" || use_collective_str=="TRUE")
    m_exchange_mode = EM_Collective;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ParallelExchanger::
~ParallelExchanger()
{
  for( auto* buf : m_comms_buf )
    delete buf;
  m_comms_buf.clear();
  delete m_own_send_message;
  delete m_own_recv_message;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IParallelMng* ParallelExchanger::
parallelMng() const
{
  return m_parallel_mng.get();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool ParallelExchanger::
initializeCommunicationsMessages()
{
  Int32 nb_send_rank = m_send_ranks.size();
  UniqueArray<Int32> gather_input_send_ranks(nb_send_rank+1);
  gather_input_send_ranks[0] = nb_send_rank;
  std::copy(std::begin(m_send_ranks),std::end(m_send_ranks),
            std::begin(gather_input_send_ranks)+1);

  IntegerUniqueArray gather_output_send_ranks;
  Integer nb_rank = m_parallel_mng->commSize();
  m_parallel_mng->allGatherVariable(gather_input_send_ranks,
                                    gather_output_send_ranks);
  
  m_recv_ranks.clear();
  Integer total_comm_rank = 0;
  Int32 my_rank = m_parallel_mng->commRank();
  {
    Integer gather_index = 0;
    for( Integer i=0; i<nb_rank; ++i ){
      Integer nb_comm = gather_output_send_ranks[gather_index];
      total_comm_rank += nb_comm;
      ++gather_index;
      for( Integer z=0; z<nb_comm; ++z ){
        Integer current_rank = gather_output_send_ranks[gather_index+z];
        if (current_rank==my_rank)
          m_recv_ranks.add(i);
      }
      gather_index += nb_comm;
    }
  }

  if (total_comm_rank==0)
    return true;
  
  _initializeCommunicationsMessages();

  return false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParallelExchanger::
initializeCommunicationsMessages(Int32ConstArrayView recv_ranks)
{
  m_recv_ranks.resize(recv_ranks.size());
  m_recv_ranks.copy(recv_ranks);
  _initializeCommunicationsMessages();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParallelExchanger::
_initializeCommunicationsMessages()
{
  if (m_verbosity_level>=1){
    info() << "ParallelExchanger " << m_name << " : nb_send=" << m_send_ranks.size()
           << " nb_recv=" << m_recv_ranks.size();
    if (m_verbosity_level>=2){
      info() << "ParallelExchanger " << m_name << " : send=" << m_send_ranks;
      info() << "ParallelExchanger " << m_name << " : recv=" << m_recv_ranks;
    }
  }

  Int32 my_rank = m_parallel_mng->commRank();

  for( Int32 msg_rank : m_send_ranks ){
    auto* comm = new SerializeMessage(my_rank,msg_rank,ISerializeMessage::MT_Send);
    // Il ne sert à rien de s'envoyer des messages.
    // (En plus ca fait planter certaines versions de MPI...)
    if (my_rank==msg_rank)
      m_own_send_message = comm;
    else
      m_comms_buf.add(comm);
    m_send_serialize_infos.add(comm);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParallelExchanger::
processExchange()
{
  ParallelExchangerOptions options;
  options.setExchangeMode(static_cast<ParallelExchangerOptions::eExchangeMode>(m_exchange_mode));
  processExchange(options);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParallelExchanger::
processExchange(const ParallelExchangerOptions& options)
{
  if (m_verbosity_level>=1)
    info() << "ParallelExchanger " << m_name << ": ProcessExchange (begin)"
           << " date=" << platform::getCurrentDateTime();

  {
    Timer::Sentry sentry(&m_timer);
    _processExchange(options);
  }

  if (m_verbosity_level>=1)
    info() << "ParallelExchanger " << m_name << ": ProcessExchange (end)"
           << " total_time=" << m_timer.lastActivationTime();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParallelExchanger::
_processExchange(const ParallelExchangerOptions& options)
{
  if (m_verbosity_level>=1){
    Int64 total_size = 0;
    for( SerializeMessage* comm : m_send_serialize_infos ){
      Int64 message_size = comm->trueSerializer()->totalSize();
      total_size += message_size;
      if (m_verbosity_level>=2)
        info() << "Send rank=" << comm->destination() << " size=" << message_size;
    }
    info() << "ParallelExchanger " << m_name << ": ProcessExchange"
           << " total_size=" << total_size << " nb_message=" << m_comms_buf.size();
  }

  bool use_all_to_all = false;
  if (options.exchangeMode())
    use_all_to_all = true;
  // TODO: traiter le cas EM_Auto

  // Génère les infos pour chaque processeur de qui on va recevoir
  // des entités
  Int32 my_rank = m_parallel_mng->commRank();
  for( Int32 msg_rank : m_recv_ranks ){
    auto* comm = new SerializeMessage(my_rank,msg_rank,ISerializeMessage::MT_Recv);
    // Il ne sert à rien de s'envoyer des messages.
    // (En plus ca fait planter certaines versions de MPI...)
    if (my_rank==msg_rank)
      m_own_recv_message = comm;
    else
      m_comms_buf.add(comm);
    m_recv_serialize_infos.add(comm);
  }

  if (use_all_to_all)
    _processExchangeCollective();
  else{
    Int32 max_pending = options.maxPendingMessage();
    if (max_pending>0)
      _processExchangeWithControl(max_pending);
    else
      m_parallel_mng->processMessages(m_comms_buf);

    if (m_own_send_message && m_own_recv_message){
      m_own_recv_message->serializer()->copy(m_own_send_message->serializer());
    }
  }

  // Récupère les infos de chaque receveur
  for( SerializeMessage* comm : m_recv_serialize_infos )
    comm->serializer()->setMode(ISerializer::ModeGet);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParallelExchanger::
_processExchangeCollective()
{
  info() << "Using collective exchange in ParallelExchanger";

  IParallelMng* pm = m_parallel_mng.get();
  Int32 nb_rank = pm->commSize();

  Int32UniqueArray send_counts(nb_rank,0);
  Int32UniqueArray send_indexes(nb_rank,0);
  Int32UniqueArray recv_counts(nb_rank,0);
  Int32UniqueArray recv_indexes(nb_rank,0);
 
  // D'abord, détermine pour chaque proc le nombre d'octets à envoyer
  for( SerializeMessage* comm : m_send_serialize_infos ){
    auto* sbuf = comm->trueSerializer();
    Span<Byte> val_buf = sbuf->globalBuffer();
    Int32 rank = comm->destRank();
    send_counts[rank] = arcaneCheckArraySize(val_buf.size());
  }

  // Fait un AllToAll pour connaitre combien de valeurs je dois recevoir des autres.
  {
    Timer::SimplePrinter sp(traceMng(),"ParallelExchanger: sending sizes with AllToAll");
    pm->allToAll(send_counts,recv_counts,1);
  }

  // Détermine le nombre total d'infos à envoyer et recevoir

  // TODO: En cas débordement, il faudrait le faire en plusieurs morceaux
  // ou alors revenir aux échanges point à point.
  Int32 total_send = 0;
  Int32 total_recv = 0;
  Int64 int64_total_send = 0;
  Int64 int64_total_recv = 0;
  for( Integer i=0; i<nb_rank; ++i ){
    send_indexes[i] = total_send;
    recv_indexes[i] = total_recv;
    total_send += send_counts[i];
    total_recv += recv_counts[i];
    int64_total_send += send_counts[i];
    int64_total_recv += recv_counts[i];
  }

  // Vérifie qu'on ne déborde pas.
  if (int64_total_send!=total_send)
    ARCANE_FATAL("Message to send is too big size={0} max=2^31",int64_total_send);
  if (int64_total_recv!=total_recv)
    ARCANE_FATAL("Message to receive is too big size={0} max=2^31",int64_total_recv);

  ByteUniqueArray send_buf(total_send);
  ByteUniqueArray recv_buf(total_recv);
  bool is_verbose = (m_verbosity_level>=1);
  if (m_verbosity_level>=2){
    for( Integer i=0; i<nb_rank; ++i ){
      if (send_counts[i]!=0 || recv_counts[i]!=0)
        info() << "INFOS: rank=" << i << " send_count=" << send_counts[i]
               << " send_idx=" << send_indexes[i]
               << " recv_count=" << recv_counts[i]
               << " recv_idx=" << recv_indexes[i];
    }
  }

  // Copie dans send_buf les infos des sérialisers.
  for( SerializeMessage* comm : m_send_serialize_infos ){
    auto* sbuf = comm->trueSerializer();
    Span<Byte> val_buf = sbuf->globalBuffer();
    Int32 rank = comm->destRank();
    if (is_verbose)
      info() << "SEND rank=" << rank << " size=" << send_counts[rank]
             << " idx=" << send_indexes[rank]
             << " buf_size=" << val_buf.size();
    ByteArrayView dest_buf(send_counts[rank],&send_buf[send_indexes[rank]]);
    dest_buf.copy(val_buf);
  }

  if (is_verbose)
    info() << "AllToAllVariable total_send=" << total_send
           << " total_recv=" << total_recv;

  {
    Timer::SimplePrinter sp(traceMng(),"ParallelExchanger: sending values with AllToAll");
    pm->allToAllVariable(send_buf,send_counts,send_indexes,recv_buf,recv_counts,recv_indexes);
  }
  // Recopie les données reçues dans le message correspondant.
  for( SerializeMessage* comm : m_recv_serialize_infos ){
    auto* sbuf = comm->trueSerializer();
    Int32 rank = comm->destRank();
    if (is_verbose)
      info() << "RECV rank=" << rank << " size=" << recv_counts[rank]
             << " idx=" << recv_indexes[rank];
    ByteArrayView orig_buf(recv_counts[rank],&recv_buf[recv_indexes[rank]]);

    sbuf->preallocate(orig_buf.size());
    sbuf->globalBuffer().copy(orig_buf);
    sbuf->setFromSizes();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ISerializeMessage* ParallelExchanger::
messageToSend(Integer i)
{
  return m_send_serialize_infos[i];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ISerializeMessage* ParallelExchanger::
messageToReceive(Integer i)
{
  return m_recv_serialize_infos[i];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParallelExchanger::
setVerbosityLevel(Int32 v)
{
  if (v<0)
    v = 0;
  m_verbosity_level = v;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParallelExchanger::
setName(const String& name)
{
  m_name = name;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{
class SortFunctor
{
 public:
  /*!
   * \brief Operateur de tri des messages.
   *
   * Le tri se fait comme suit:
   * - d'abord prend 1 rang sur nb_phase pour éviter que tous les messages
   *   aillent sur les mêmes noeuds (car on suppose que les rangs consécutifs sont
   *   sur les mêmes noeuds)
   * - ensuite tri sur le rang de destination
   * - enfin poste les réceptions avant les envois.
   */
  bool operator()(const ISerializeMessage* a,const ISerializeMessage* b)
  {
    const int nb_phase = 4;
    int phase1 = a->destination().value() % nb_phase;
    int phase2 = b->destination().value() % nb_phase;
    if (phase1 != phase2)
      return phase1<phase2;
    if (a->destination() != b->destination())
      return a->destination() < b->destination();
    if (a->isSend() != b->isSend())
      return (a->isSend() ? false : true);
    return a->source() < b->source();
  }
};
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Echange avec contrôle du nombre maximum de messages en vol.
 */
void ParallelExchanger::
_processExchangeWithControl(Int32 max_pending_message)
{
  // L'ensemble des messages sont dans 'm_comms_buf'.
  // On les recopie dans 'sorted_messages' pour qu'ils soient triés.

  auto message_list {m_parallel_mng->createSerializeMessageListRef()};

  UniqueArray<ISerializeMessage*> sorted_messages(m_comms_buf);
  std::sort(sorted_messages.begin(),sorted_messages.end(),SortFunctor{});

  Integer position = 0;
  // Il faut au moins ajouter un minimum de messages pour ne pas avoir de blocage.
  // A priori le minimum est 2 pour qu'il y est au moins un receive et un send
  // mais il est préférable de mettre plus pour ne pas trop dégrader les performances.
  max_pending_message = math::max(4,max_pending_message);

  Integer nb_message = sorted_messages.size();
  Integer nb_to_add = max_pending_message;

  Int32 verbosity_level = m_verbosity_level;

  if (verbosity_level>=1)
    info() << "ParallelExchanger " << m_name << " : process exchange WITH CONTROL"
           << " nb_message=" << nb_message << " max_pending=" << max_pending_message;

  while(position<nb_message){
    for( Integer i=0; i<nb_to_add; ++i ){
      if (position>=nb_message)
        break;
      ISerializeMessage* message = sorted_messages[position];
      if (verbosity_level>=2)
        info() << "Add Message p=" << position << " is_send?=" << message->isSend() << " source=" << message->source()
               << " dest=" << message->destination();
      message_list->addMessage(message);
      ++position;
    }
    // S'il ne reste plus de messages, alors on fait un WaitAll pour attendre*
    // que les messages restants soient tous terminés.
    if (position>=nb_message){
      message_list->waitMessages(Parallel::WaitAll);
      break;
    }
    // Le nombre de messages terminés indique combien de message il faudra
    // ajouter à la liste pour la prochaine itération.
    Integer nb_done = message_list->waitMessages(Parallel::WaitSome);
    if (verbosity_level>=2)
      info() << "Wait nb_done=" << nb_done;
    if (nb_done==(-1))
      nb_done = max_pending_message;
    nb_to_add = nb_done;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Ref<IParallelExchanger>
createParallelExchangerImpl(Ref<IParallelMng> pm)
{
  return makeRef<IParallelExchanger>(new ParallelExchanger(pm));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

