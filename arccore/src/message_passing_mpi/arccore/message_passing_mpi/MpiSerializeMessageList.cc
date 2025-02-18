// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MpiSerializeMessageList.cc                                  (C) 2000-2025 */
/*                                                                           */
/* Gestion des messages de sérialisation via MPI.                            */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing_mpi/internal/MpiSerializeMessageList.h"
#include "arccore/message_passing_mpi/internal/MpiSerializeDispatcher.h"
#include "arccore/message_passing_mpi/internal/MpiAdapter.h"
#include "arccore/message_passing/BasicSerializeMessage.h"
#include "arccore/trace/ITraceMng.h"
#include "arccore/base/FatalErrorException.h"
#include "arccore/base/TimeoutException.h"
#include "arccore/base/NotSupportedException.h"

#include <algorithm>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MessagePassing::Mpi
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class MpiSerializeMessageList::_SortMessages
{
 public:

  bool operator()(const internal::BasicSerializeMessage* m1,
                  const internal::BasicSerializeMessage* m2)
  {
    return _SortMessages::compare(m1,m2);
  }
  
  bool operator()(const ISerializeMessage* pm1,const ISerializeMessage* pm2)
  {
    return compare(pm1,pm2);
  }
  // Note: avec la version 16.5.3 (avril 2020) de VisualStudio, ce
  // comparateur génère une exception en mode débug. Cela signifie qu'il
  // n'est pas cohérent.
  // TODO: corriger le problème
  static bool compare(const ISerializeMessage* pm1,const ISerializeMessage* pm2)
  {
    MessageRank dest_p1 = pm1->destination();
    MessageRank dest_p2 = pm2->destination();
    MessageTag p1_tag = pm1->internalTag();
    MessageTag p2_tag = pm2->internalTag();
      
    // TODO: traiter le cas destRank()==A_NULL_RANK
    if (dest_p1==dest_p2){
      MessageRank orig_p1 = pm1->source();
      MessageRank orig_p2 = pm2->source();

      if (pm1->isSend()){
        if (orig_p1==orig_p2 && (p1_tag!=p2_tag))
          return p1_tag < p2_tag;
        if (orig_p1<dest_p1)
          return true;
      }
      if (!pm1->isSend()){
        if (orig_p1==orig_p2 && (p1_tag!=p2_tag))
          return p1_tag < p2_tag;
        if (dest_p1<orig_p1)
          return true;
      }
      return false;
    }
    if (dest_p1 < dest_p2)
      return true;
    return false;
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MpiSerializeMessageList::
MpiSerializeMessageList(MpiSerializeDispatcher* dispatcher)
: m_dispatcher(dispatcher)
, m_adapter(dispatcher->adapter())
, m_trace(m_adapter->traceMng())
, m_message_passing_phase(timeMetricPhaseMessagePassing(m_adapter->timeMetricCollector()))
{}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiSerializeMessageList::
addMessage(ISerializeMessage* message)
{
  auto* true_message = dynamic_cast<internal::BasicSerializeMessage*>(message);
  if (!true_message)
    ARCCORE_FATAL("Can not convert 'ISerializeMessage' to 'BasicSerializeMessage'");
  if (true_message->isSend() && true_message->source()!=MessageRank(m_adapter->commRank()))
    ARCCORE_FATAL("Invalid source '{0}' for send message (expected={1})",
                  true_message->source(),m_adapter->commRank());
  m_messages_to_process.add(true_message);
}
  
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiSerializeMessageList::
processPendingMessages()
{
  Integer nb_message = m_messages_to_process.size();
  if (nb_message==0)
    return;

  // L'envoie de messages peut se faire en mode bloquant ou non bloquant.
  // Quel que soit le mode, l'ordre d'envoie doit permettre de ne pas
  // avoir de deadlock. Pour cela, on applique l'algorithme suivant:
  // - chaque processeur effectue ces envois et réceptions dans l'ordre
  // croissant des rang de processeurs
  // - lorsque deux processeurs communiquent, c'est celui dont le rang est
  // le plus faible qui envoie ces messages d'abord.
  ITraceMng* msg = m_trace;
  // NOTE (avril 2020): n'appelle plus le tri car il semble que l'opérateur
  // de comparaison ne soit pas cohérent. De plus, il n'est normalement
  // plus nécessaire de faire ce tri car tout est non bloquant.
  //std::stable_sort(std::begin(m_messages_to_process),std::end(m_messages_to_process),_SortMessages());
  const bool print_sorted = false;
  if (print_sorted){
    for( Integer i=0, is=m_messages_to_process.size(); i<is; ++i ){
      ISerializeMessage* pmsg = m_messages_to_process[i];
      msg->debug() << "Sorted message " << i
                   << " orig=" << pmsg->source()
                   << " dest=" << pmsg->destination()
                   << " tag=" << pmsg->internalTag()
                   << " send?=" << pmsg->isSend();
    }
  }

  Int64 serialize_buffer_size = m_dispatcher->serializeBufferSize();
  for( Integer i=0; i<nb_message; ++i ){
    internal::BasicSerializeMessage* mpi_msg = m_messages_to_process[i];
    ISerializeMessage* pmsg = mpi_msg;
    Request new_request;
    MessageRank dest = pmsg->destination();
    MessageTag tag = pmsg->internalTag();
    bool is_one_message_strategy = (pmsg->strategy()==ISerializeMessage::eStrategy::OneMessage);
    if (pmsg->isSend()){
      // TODO: il faut utiliser m_dispatcher->sendSerializer() à la place
      // de legacySendSerializer() mais avant de fair cela il faut envoyer
      // les deux messages potentiels en même temps pour des raisons de
      // performance (voir MpiSerializeDispatcher::sendSerializer())
      const bool do_old = false;
      if (do_old){
        if (is_one_message_strategy)
          ARCCORE_THROW(NotSupportedException,"OneMessage strategy with legacy send serializer");
        new_request = m_dispatcher->legacySendSerializer(pmsg->serializer(),{dest,tag,NonBlocking});
      }
      else
        new_request = m_dispatcher->sendSerializer(pmsg->serializer(),{dest,tag,NonBlocking},is_one_message_strategy);
    }
    else{
      BasicSerializer* sbuf = mpi_msg->trueSerializer();
      sbuf->preallocate(serialize_buffer_size);
      MessageId message_id = pmsg->_internalMessageId();
      if (message_id.isValid()){
        // Message de sérialisation utilisant MPI_Message
        // 'message_id' contient la taille du message final. On préalloue donc
        // le buffer de réception à cette taille ce qui permet si besoin de ne faire
        // qu'un seul message de réception.
        // préallouer le buffer à la taille né
        if (is_one_message_strategy)
          sbuf->preallocate(message_id.sourceInfo().size());
        new_request = m_dispatcher->_recvSerializerBytes(sbuf->globalBuffer(),message_id,false);
      }
      else
        new_request = m_dispatcher->_recvSerializerBytes(sbuf->globalBuffer(),dest,tag,false);
    }
    mpi_msg->setIsProcessed(true);
    m_messages_request.add(MpiSerializeMessageRequest(mpi_msg,new_request));
  }
  // Plus de messages à exécuter
  m_messages_to_process.clear();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer MpiSerializeMessageList::
waitMessages(eWaitType wait_type)
{
  processPendingMessages();
  Integer n = _waitMessages(wait_type);
  m_dispatcher->checkFinishedSubRequests();
  return n;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer MpiSerializeMessageList::
_waitMessages(eWaitType wait_type)
{
  TimeMetricSentry tphase(m_message_passing_phase);
  if (wait_type==WaitAll){
    while (_waitMessages2(WaitSome)!=(-1))
      ;
    return (-1);
  }
  return _waitMessages2(wait_type);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer MpiSerializeMessageList::
_waitMessages2(eWaitType wait_type)
{
  Integer nb_message_finished = 0;
  ITraceMng* msg = m_trace;
  Integer nb_message = m_messages_request.size();
  Int32 comm_rank = m_adapter->commRank();
  UniqueArray<MPI_Status> mpi_status(nb_message);
  UniqueArray<Request> requests(nb_message);
  UniqueArray<bool> done_indexes(nb_message);
  done_indexes.fill(false);
  if (msg->verbosityLevel()>=6)
    m_is_verbose = true;

  for( Integer z=0; z<nb_message; ++z ){
    requests[z] = m_messages_request[z].m_request;
  }

  if (m_is_verbose){
    msg->info() << "Waiting for rank =" << comm_rank << " nb_message=" << nb_message;

    for( Integer z=0; z<nb_message; ++z ){
      internal::BasicSerializeMessage* msm = m_messages_request[z].m_mpi_message;
      msg->info() << "Waiting for message: "
                  << " rank=" << comm_rank
                  << " issend=" << msm->isSend()
                  << " dest=" << msm->destination()
                  << " tag=" << msm->internalTag()
                  << " request=" << requests[z];
    }
  }

  mpi_status.resize(nb_message);
  MpiAdapter* adapter = m_adapter;
  try{
    switch(wait_type){
    case WaitAll:
      ARCCORE_FATAL("Bad value WaitAll");
    case WaitSome:
      msg->debug() << " rank=" << comm_rank << "Wait some " << nb_message;
      if (nb_message>0)
        adapter->waitSomeRequestsMPI(requests,done_indexes,mpi_status,false);
      break;
    case WaitSomeNonBlocking:
      msg->debug() << " rank=" << comm_rank << "Wait some non blocking " << nb_message;
      if (nb_message>0)
        adapter->waitSomeRequestsMPI(requests,done_indexes,mpi_status,true);
      break;
    }
  }
  catch(const TimeoutException&){
    std::ostringstream ostr;
    for( Integer z=0; z<nb_message; ++z ){
      internal::BasicSerializeMessage* message = m_messages_request[z].m_mpi_message;
      ostr << "IndexReturn message: "
           << " issend=" << message->isSend()
           << " dest=" << message->destination()
           << " done_index=" << done_indexes[z]
           << " status_src=" << mpi_status[z].MPI_SOURCE
           << " status_tag=" << mpi_status[z].MPI_TAG
           << " status_err=" << mpi_status[z].MPI_ERROR
           << " request=" << requests[z]
           << "\n";
    }
    msg->pinfo() << "Info messages: myrank=" << comm_rank << " " << ostr.str();
    throw;
  }
  if (m_is_verbose){
    for( Integer z=0; z<nb_message; ++z ){
      internal::BasicSerializeMessage* message = m_messages_request[z].m_mpi_message;
      bool is_send = message->isSend();
      MessageRank destination = message->destination();
      Int64 message_size = message->trueSerializer()->totalSize();
      if (is_send)
        msg->info() << "IndexReturn message: Send: "
                    << " dest=" << destination
                    << " size=" << message_size
                    << " done_index=" << done_indexes[z]
                    << " request=" << requests[z];
      else
        msg->info() << "IndexReturn message: Recv: "
                    << " dest=" << destination
                    << " size=" << message_size
                    << " done_index=" << done_indexes[z]
                    << " request=" << requests[z]
                    << " status_src=" << mpi_status[z].MPI_SOURCE
                    << " status_tag=" << mpi_status[z].MPI_TAG
                    << " status_err=" << mpi_status[z].MPI_ERROR;
    }
  }

  UniqueArray<MpiSerializeMessageRequest> new_messages;

  int mpi_status_index = 0;
  for( Integer i=0; i<nb_message; ++i ){
    internal::BasicSerializeMessage* mpi_msg = m_messages_request[i].m_mpi_message;
    if (done_indexes[i]){
      MPI_Status status = mpi_status[mpi_status_index];
      Request rq = requests[i];
      // NOTE: les valeurs MPI_SOURCE et MPI_TAG de status ne sont
      // valides que pour les réception. On utilise ces valeurs et pas celles
      // de ISerializeMessage car il est possible de spécifier MPI_ANY_TAG ou
      // MPI_ANY_SOURCE dans les messages et on a besoin de connaitre les bonnes
      // valeurs pour réceptionner le second message.
      MessageRank source(status.MPI_SOURCE);
      MessageTag tag(status.MPI_TAG);
      if (m_is_verbose){
        msg->info() << "Message number " << i << " Finished, source=" << source
                    << " tag=" << tag
                    << " err=" << status.MPI_ERROR
                    << " is_send=" << mpi_msg->isSend()
                    << " request=" << rq;
      }
      ++mpi_status_index;
      Request r = _processOneMessage(mpi_msg,source,tag);
      if (r.isValid()){
        if (m_is_verbose)
          msg->info() << "Add new receive operation for message number " << i
                      << " request=" << r;
        new_messages.add(MpiSerializeMessageRequest(mpi_msg,r));
      }
      else{
        mpi_msg->setFinished(true);
        ++nb_message_finished;
      }
    }
    else{
      if (m_is_verbose)
        msg->info() << "Message number " << i << " not finished"
                    << " request=" << requests[i];
      new_messages.add(MpiSerializeMessageRequest(mpi_msg,requests[i]));
    }
  }
  msg->flush();
  m_messages_request = new_messages;
  if (m_messages_request.empty())
    return (-1);
  return nb_message_finished;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Effectue la requête. Retourne une éventuelle requête si non nul.
 */
Request MpiSerializeMessageList::
_processOneMessage(internal::BasicSerializeMessage* message, MessageRank source, MessageTag mpi_tag)
{
  Request request;
  if (m_is_verbose)
    m_trace->info() << "Process one message msg=" << this
                    << " number=" << message->messageNumber()
                    << " is_send=" << message->isSend();
  if (message->isSend())
    return request;
  return _processOneMessageGlobalBuffer(message,source,mpi_tag);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Effectue la requête. Retourne une éventuelle requête si non nul.
 */
Request MpiSerializeMessageList::
_processOneMessageGlobalBuffer(internal::BasicSerializeMessage* message,MessageRank source,MessageTag mpi_tag)
{
  Request request;
  BasicSerializer* sbuf = message->trueSerializer();
  Int64 message_size = sbuf->totalSize();

  MessageRank dest_rank = message->destination();
  if (dest_rank.isNull() && !m_adapter->isAllowNullRankForAnySource())
    ARCCORE_FATAL("Can not use MPI_Mprobe with null rank. Use MessageRank::anySourceRank() instead");

  if (dest_rank.isNull() || dest_rank.isAnySource())
    // Signifie que le message était un MPI_ANY_SOURCE
    dest_rank = source;

  if (m_is_verbose){
    m_trace->info() << "Process one message (GlobalBuffer) msg=" << this
                    << " number=" << message->messageNumber()
                    << " is_send=" << message->isSend()
                    << " dest_rank=" << dest_rank
                    << " size=" << message_size
                    << " (buf_size=" << m_dispatcher->serializeBufferSize() << ")";
  }

  // S'il s'agit du premier message, récupère la longueur totale.
  // et si le message total est trop gros (>m_serialize_buffer_size)
  // poste un nouveau message pour récupèrer les données sérialisées.
  if (message->messageNumber()==0){
    if (message_size<=m_dispatcher->serializeBufferSize()
        || message->strategy()==ISerializeMessage::eStrategy::OneMessage){
      sbuf->setFromSizes();
      return request;
    }
    m_dispatcher->_checkBigMessage(message_size);
    sbuf->preallocate(message_size);
    Span<Byte> bytes = sbuf->globalBuffer();
    MessageTag next_tag = MpiSerializeDispatcher::nextSerializeTag(mpi_tag);
    request = m_dispatcher->_recvSerializerBytes(bytes,dest_rank,next_tag,false);
    message->setMessageNumber(1);
    return request;
  }
  sbuf->setFromSizes();
  return request;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Ref<ISerializeMessage> MpiSerializeMessageList::
createAndAddMessage(MessageRank destination,ePointToPointMessageType type)
{
  MessageRank source(m_adapter->commRank());
  auto x = internal::BasicSerializeMessage::create(source,destination,type);
  addMessage(x.get());
  return x;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore::MessagePassing::Mpi

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
