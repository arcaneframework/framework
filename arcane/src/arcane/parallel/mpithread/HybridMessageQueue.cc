// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* HybridMessageQueue.cc                                       (C) 2000-2025 */
/*                                                                           */
/* File de messages pour une implémentation MPI/Thread.                      */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/utils/Array.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/NotImplementedException.h"
#include "arcane/utils/NotSupportedException.h"
#include "arcane/utils/ArgumentException.h"
#include "arcane/utils/TraceInfo.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/ValueConvert.h"

#include "arcane/parallel/mpithread/HybridMessageQueue.h"
#include "arcane/parallel/mpi/MpiParallelMng.h"

#include "arccore/message_passing_mpi/internal/MpiAdapter.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Macro pour afficher des messages pour debug
#define TRACE_DEBUG(needed_debug_level,format_str,...) \
  if (m_debug_level>=needed_debug_level){ \
    info() << String::format("Hybrid " format_str,__VA_ARGS__);\
    traceMng()->flush();\
  }

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MessagePassing
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

HybridMessageQueue::
HybridMessageQueue(ISharedMemoryMessageQueue* thread_queue,MpiParallelMng* mpi_pm,
                   Int32 local_nb_rank)
: TraceAccessor(mpi_pm->traceMng())
, m_thread_queue(thread_queue)
, m_mpi_parallel_mng(mpi_pm)
, m_mpi_adapter(mpi_pm->adapter())
, m_local_nb_rank(local_nb_rank)
, m_rank_tag_builder(local_nb_rank)
, m_debug_level(0)
{
  if (auto v = Convert::Type<Int32>::tryParseFromEnvironment("ARCCORE_ALLOW_NULL_RANK_FOR_MPI_ANY_SOURCE", true))
    m_is_allow_null_rank_for_any_source = v.value() != 0;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void HybridMessageQueue::
_checkValidRank(MessageRank rank)
{
  if (rank.isNull())
    ARCANE_THROW(ArgumentException,"null rank");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void HybridMessageQueue::
_checkValidSource(const PointToPointMessageInfo& message)
{
  MessageRank source = message.emiterRank();
  if (source.isNull())
    ARCANE_THROW(ArgumentException,"null source");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

PointToPointMessageInfo HybridMessageQueue::
_buildSharedMemoryMessage(const PointToPointMessageInfo& message,
                          const SourceDestinationFullRankInfo& fri)
{
  PointToPointMessageInfo p2p_message(message);
  p2p_message.setEmiterRank(fri.source().localRank());
  p2p_message.setDestinationRank(fri.destination().localRank());
  return p2p_message;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

PointToPointMessageInfo HybridMessageQueue::
_buildMPIMessage(const PointToPointMessageInfo& message,
                 const SourceDestinationFullRankInfo& fri)
{
  PointToPointMessageInfo p2p_message(message);
  p2p_message.setEmiterRank(fri.source().mpiRank());
  p2p_message.setDestinationRank(fri.destination().mpiRank());
  return p2p_message;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void HybridMessageQueue::
waitAll(ArrayView<Request> requests)
{
  // TODO: fusionner ce qui est possible avec waitSome.
  Integer nb_request = requests.size();
  UniqueArray<Request> mpi_requests;
  UniqueArray<Request> thread_requests;
  for( Integer i=0; i<nb_request; ++i ){
    Request r = requests[i];
    if (!r.isValid())
      continue;
    IRequestCreator* creator = r.creator();
    if (creator==m_mpi_adapter) {
      mpi_requests.add(r);
    }
    else if (creator==m_thread_queue)
      thread_requests.add(r);
    else
      ARCANE_FATAL("Invalid IRequestCreator");
  }

  if (mpi_requests.size()!=0)
    m_mpi_adapter->waitAllRequests(mpi_requests);
  if (thread_requests.size()!=0)
    m_thread_queue->waitAll(thread_requests);

  // On remet à zero toutes les requetes pour pouvoir rappeler les fonctions Wait !
  for( Request r : requests )
    r.reset();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void HybridMessageQueue::
waitSome(Int32 rank,ArrayView<Request> requests,ArrayView<bool> requests_done,
         bool is_non_blocking)
{
  Integer nb_done = 0;
  do{
    TRACE_DEBUG(2,"Hybrid: wait some rank={0} requests n={1} nb_done={2} is_non_blocking={3}",
                rank,requests.size(),nb_done,is_non_blocking);
    nb_done = _testOrWaitSome(rank,requests,requests_done);
    if (is_non_blocking || nb_done==(-1))
      break;
  } while (nb_done==0);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer HybridMessageQueue::
_testOrWaitSome(Int32 rank,ArrayView<Request> requests,ArrayView<bool> requests_done)
{
  Integer nb_request = requests.size();
  TRACE_DEBUG(2,"Hybrid: wait some rank={0} requests n={1}",rank,nb_request);

  // Il faut séparer les requêtes MPI des requêtes en mémoire partagée.
  // TODO: avec la notion de requête généralisé de MPI, il serait peut-être
  // possible de fusionner cela.
  UniqueArray<Request> mpi_requests;
  UniqueArray<Request> shm_requests;
  // Indice des requêtes dans la liste globale \a requests
  UniqueArray<Integer> mpi_requests_index;
  UniqueArray<Integer> shm_requests_index;

  Integer nb_done = 0;
  for( Integer i=0; i<nb_request; ++i ){
    Request r = requests[i];
    if (!r.isValid())
      continue;
    IRequestCreator* creator = r.creator();
    if (creator==m_mpi_adapter){
      mpi_requests.add(r);
      mpi_requests_index.add(i);
    }
    else if (creator==m_thread_queue){
      shm_requests.add(r);
      shm_requests_index.add(i);
    }
    else
      ARCANE_FATAL("Invalid IRequestCreator");
  }

  TRACE_DEBUG(2,"Hybrid: wait some rank={0} nb_mpi={1} nb_shm={2}",
              rank,mpi_requests.size(),shm_requests.size());

  // S'il n'y a aucune requête valide, inutile d'aller plus loin.
  // Il ne faut cependant pas retourner '0' car on doit faire
  // la différence entre aucune requête disponible pour le mode 'is_non_blocking'
  // et aucune requête valide.
  if (mpi_requests.size()==0 && shm_requests.size()==0)
    return (-1);

  // Même en mode waitSome, il faut utiliser le mode non bloquant car
  // on ne sait pas entre les threads et MPI quelles seront les requêtes
  // qui sont disponibles

  // Les requêtes ont pu être modifiées si elles ne sont pas terminées.
  // Il faut donc les remettre dans la liste \a requests. Dans notre
  // cas il suffit uniquement de recopier la nouvelle valeur dans
  // l'instance correspondante de HybridMessageRequest.
  UniqueArray<bool> mpi_done_indexes;
  Integer nb_mpi_request = mpi_requests.size();

  if (nb_mpi_request!=0){
    mpi_done_indexes.resize(nb_mpi_request);
    mpi_done_indexes.fill(false);
    m_mpi_adapter->waitSomeRequests(mpi_requests,mpi_done_indexes,true);
    TRACE_DEBUG(2,"Hybrid: MPI wait some requests n={0} after=",nb_mpi_request,mpi_done_indexes);
    for( Integer i=0; i<nb_mpi_request; ++i ){
      Integer index_in_global = mpi_requests_index[i];
      if (mpi_done_indexes[i]){
        requests_done[index_in_global] = true;
        requests[index_in_global].reset();
        ++nb_done;
        TRACE_DEBUG(1,"MPI rank={0} set done i={1} in_global={2}",
                    rank,i,index_in_global);
      }
      else
        requests[index_in_global] = mpi_requests[i];
    }
  }

  UniqueArray<bool> shm_done_indexes;
  Integer nb_shm_request = shm_requests.size();
  TRACE_DEBUG(2,"SHM wait some requests n={0}",nb_shm_request);
  if (shm_requests.size()!=0){
    shm_done_indexes.resize(nb_shm_request);
    shm_done_indexes.fill(false);
    m_thread_queue->waitSome(rank,shm_requests,shm_done_indexes,true);
    for( Integer i=0; i<nb_shm_request; ++i ){
      Integer index_in_global = shm_requests_index[i];
      if (shm_done_indexes[i]){
        requests_done[index_in_global] = true;
        requests[index_in_global].reset();
        ++nb_done;
        TRACE_DEBUG(1,"SHM rank={0} set done i={1} in_global={2}",
                    rank,i,index_in_global);
      }
      else
        requests[index_in_global] = shm_requests[i];
    }
  }
  return nb_done;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Request HybridMessageQueue::
_addReceiveRankTag(const PointToPointMessageInfo& message,ReceiveBufferInfo buf_info)
{
  // On ne supporte pas les réceptions avec ANY_RANK car on ne sait pas
  // s'il faut faire un 'receive' avec MPI ou en mémoire partagée.
  // Dans ce cas, l'utilisateur doit plutôt utiliser probe()
  // pour savoir ce qui est disponible et envoyer faire un addReceive()
  // avec un MessageId.
  if (message.destinationRank().isNull())
    ARCANE_THROW(NotSupportedException,"Receive with any rank. Use probe() and MessageId instead");

  SourceDestinationFullRankInfo fri = _getFullRankInfo(message);
  bool is_same_mpi_rank = fri.isSameMpiRank();

  if (is_same_mpi_rank){
    TRACE_DEBUG(1,"** MPITMQ SHM ADD RECV S queue={0} message={1}",this,message);
    PointToPointMessageInfo p2p_message(_buildSharedMemoryMessage(message,fri));
    return m_thread_queue->addReceive(p2p_message,buf_info);
  }

  ISerializer* serializer = buf_info.serializer();
  if (serializer){
    TRACE_DEBUG(1,"** MPITMQ MPI ADD RECV S queue={0} message={1}",this,message);
    PointToPointMessageInfo p2p_message(_buildMPIMessage(message,fri));
    p2p_message.setTag(m_rank_tag_builder.tagForReceive(MessageTag(message.tag()),fri));
    return m_mpi_parallel_mng->receiveSerializer(serializer,p2p_message);
  }
  else{
    ByteSpan buf = buf_info.memoryBuffer();
    Int64 size = buf.size();

    TRACE_DEBUG(1,"** MPITMQ THREAD ADD RECV B queue={0} message={1} size={2} same_mpi?={3}",
                this,message,size,fri.isSameMpiRank());

    //TODO: utiliser le vrai MPI_Datatype
    MPI_Datatype char_data_type = MpiBuiltIn::datatype(char());
    MessageTag mpi_tag = m_rank_tag_builder.tagForReceive(message.tag(),fri);
    Request r = m_mpi_adapter->directRecv(buf.data(),size,fri.destination().mpiRankValue(),sizeof(char),
                                          char_data_type,mpi_tag.value(),false);
    return r;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Request HybridMessageQueue::
_addReceiveMessageId(const PointToPointMessageInfo& message,ReceiveBufferInfo buf_info)
{
  MessageId message_id = message.messageId();
  MessageId::SourceInfo si(message_id.sourceInfo());

  if (si.rank()!=message.destinationRank())
    ARCANE_FATAL("Incohence between messsage_id rank and destination rank x1={0} x2={1}",
                 si.rank(),message.destinationRank());

  TRACE_DEBUG(1,"** MPITMQ ADD_RECV (message_id) queue={0} message={1}",
              this,message);

  SourceDestinationFullRankInfo fri = _getFullRankInfo(message);
  if (fri.isSameMpiRank()){
    PointToPointMessageInfo p2p_message(_buildSharedMemoryMessage(message,fri));
    return m_thread_queue->addReceive(p2p_message,buf_info);
  }

  TRACE_DEBUG(1,"** MPITMQ MPI ADD RECV (message_id) queue={0} message={1}",this,message);

  ISerializer* serializer = buf_info.serializer();
  if (serializer){
    PointToPointMessageInfo p2p_message(_buildMPIMessage(message,fri));
    //p2p_message.setTag(m_rank_tag_builder.tagForReceive(message.tag(),fri));
    TRACE_DEBUG(1,"** MPI ADD RECV Serializer (message_id) message={0} p2p_message={1}",
                message,p2p_message);
    return m_mpi_parallel_mng->receiveSerializer(serializer,p2p_message);
  }
  else{
    ByteSpan buf = buf_info.memoryBuffer();
    Int64 size = buf.size();

    // TODO: utiliser le vrai MPI_Datatype
    MPI_Datatype char_data_type = MpiBuiltIn::datatype(char());
    MessageId mpi_message(message_id);
    MessageId::SourceInfo mpi_si(si);
    mpi_si.setRank(fri.destination().mpiRank());
    mpi_message.setSourceInfo(mpi_si);
    return m_mpi_adapter->directRecv(buf.data(),size,mpi_message,sizeof(char),
                                     char_data_type,false);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Request HybridMessageQueue::
addReceive(const PointToPointMessageInfo& message,ReceiveBufferInfo buf)
{
  _checkValidSource(message);

  if (!message.isValid())
    return Request();

  if (message.isRankTag())
    return _addReceiveRankTag(message,buf);

  if (message.isMessageId())
    return _addReceiveMessageId(message,buf);

  ARCANE_THROW(NotSupportedException,"Invalid message_info");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Request HybridMessageQueue::
addSend(const PointToPointMessageInfo& message,SendBufferInfo buf_info)
{
  if (!message.isValid())
    return Request();
  if (message.destinationRank().isNull())
    ARCCORE_FATAL("Null destination");
  if (!message.isRankTag())
    ARCCORE_FATAL("Invalid message_info for sending: message.isRankTag() is false");

  SourceDestinationFullRankInfo fri = _getFullRankInfo(message);

  // Même rang donc envoie via la file en mémoire partagée.
  if (fri.isSameMpiRank()){
    TRACE_DEBUG(1,"** MPITMQ SHM ADD SEND S queue={0} message={1}",this,message);
    PointToPointMessageInfo p2p_message(_buildSharedMemoryMessage(message,fri));
    return m_thread_queue->addSend(p2p_message,buf_info);
  }

  // Envoie via MPI
  MessageTag mpi_tag = m_rank_tag_builder.tagForSend(message.tag(),fri);
  const ISerializer* serializer = buf_info.serializer();
  if (serializer){
    PointToPointMessageInfo p2p_message(_buildMPIMessage(message,fri));
    p2p_message.setTag(mpi_tag);
    TRACE_DEBUG(1,"** MPITMQ MPI ADD SEND Serializer queue={0} message={1} p2p_message={2}",
                this,message,p2p_message);
    return m_mpi_parallel_mng->sendSerializer(serializer,p2p_message);
  }
  else{
    ByteConstSpan buf = buf_info.memoryBuffer();
    Int64 size = buf.size();

    // TODO: utiliser m_mpi_parallel_mng mais il faut faire attention
    // d'utiliser le mode bloquant
    // TODO: utiliser le vrai MPI_Datatype
    MPI_Datatype char_data_type = MpiBuiltIn::datatype(char());

    TRACE_DEBUG(1,"** MPITMQ MPI ADD SEND B queue={0} message={1} size={2} mpi_tag={3} mpi_rank={4}",
                this,message,size,mpi_tag,fri.destination().mpiRank());

    return m_mpi_adapter->directSend(buf.data(),size,fri.destination().mpiRankValue(),
                                     sizeof(char),char_data_type,mpi_tag.value(),false);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MP::MessageId HybridMessageQueue::
probe(const MP::PointToPointMessageInfo& message)
{
  TRACE_DEBUG(1,"Probe msg='{0}' queue={1} is_valid={2}",
              message,this,message.isValid());

  MessageRank orig = message.emiterRank();
  if (orig.isNull())
    ARCANE_THROW(ArgumentException,"null sender");

  if (!message.isValid())
    return MessageId();

  // Il faut avoir initialisé le message avec un couple (rang/tag).
  if (!message.isRankTag())
    ARCCORE_FATAL("Invalid message_info: message.isRankTag() is false");

  MessageRank dest = message.destinationRank();
  MessageTag user_tag = message.tag();
  bool is_blocking = message.isBlocking();
  if (is_blocking)
    ARCANE_THROW(NotImplementedException,"blocking probe");
  if (user_tag.isNull())
    ARCANE_THROW(NotImplementedException,"probe with ANY_TAG");
  FullRankInfo orig_fri = m_rank_tag_builder.rank(orig);
  FullRankInfo dest_fri = m_rank_tag_builder.rank(dest);
  MessageId message_id;
  Int32 found_dest = dest.value();
  const bool is_any_source = dest.isNull() || dest.isAnySource();
  if (dest.isNull() && !m_is_allow_null_rank_for_any_source)
    ARCANE_FATAL("Can not use probe() with null rank. Use MessageRank::anySourceRank() instead");
  if (is_any_source) {
    // Comme on ne sait pas de qui on va recevoir, il faut tester à la
    // fois la file de thread et via MPI.
    MP::PointToPointMessageInfo p2p_message(message);
    p2p_message.setEmiterRank(orig_fri.localRank());
    message_id = m_thread_queue->probe(p2p_message);
    if (message_id.isValid()){
      // On a trouvé un message dans la liste de thread.
      // Comme on est dans notre liste de thread, le
      // rang global est notre rang MPI + le rang local trouvé.
      found_dest = orig_fri.mpiRankValue()*m_local_nb_rank + message_id.sourceInfo().rank().value();
      TRACE_DEBUG(2,"Probe with null_rank (thread) orig={0} found_dest={1} tag={2}",
                  orig,found_dest,user_tag);
    }
    else{
      // Recherche via MPI.
      // La difficulté est que le rang local du PE originaire du message
      // est codé dans le tag et qu'on ne connait pas le PE originaire.
      // Il faut donc tester tous les tag potentiels. Leur nombre est
      // égal à 'm_nb_local_rank'.
      for( Integer z=0, zn=m_local_nb_rank; z<zn; ++z ){
        MP::PointToPointMessageInfo mpi_message(message);
        MessageTag mpi_tag = m_rank_tag_builder.tagForReceive(user_tag,orig_fri.localRank(),MessageRank(z));
        mpi_message.setTag(mpi_tag);
        TRACE_DEBUG(2,"Probe with null_rank orig={0} dest={1} tag={2}",orig,dest,mpi_tag);
        message_id = m_mpi_adapter->probeMessage(mpi_message);
        if (message_id.isValid()){
          // On a trouvé un message MPI. Il faut extraire du tag le
          // rang local. Le rang MPI est celui dans le message.
          MessageRank mpi_rank = message_id.sourceInfo().rank();
          MessageTag ret_tag = message_id.sourceInfo().tag();
          Int32 local_rank = m_rank_tag_builder.getReceiveRankFromTag(ret_tag);
          found_dest = mpi_rank.value()*m_local_nb_rank + local_rank;
          TRACE_DEBUG(2,"Probe null rank found mpi_rank={0} local_rank={1} tag={2}",
                      ret_tag,mpi_rank,local_rank,ret_tag);
          break;
        }
      }
    }
  }
  else{
    // Il faut convertir le rang `dest` en le rang attendu par la file de thread
    // ou par MPI.
    if (orig_fri.mpiRank()==dest_fri.mpiRank()){
      MP::PointToPointMessageInfo p2p_message(message);
      p2p_message.setDestinationRank(MP::MessageRank(dest_fri.localRank()));
      p2p_message.setEmiterRank(MessageRank(orig_fri.localRank()));
      message_id = m_thread_queue->probe(p2p_message);
    }
    else{
      MP::PointToPointMessageInfo mpi_message(message);
      MessageTag mpi_tag = m_rank_tag_builder.tagForReceive(user_tag,orig_fri,dest_fri);
      mpi_message.setTag(mpi_tag);
      mpi_message.setDestinationRank(MP::MessageRank(dest_fri.mpiRank()));
      TRACE_DEBUG(2,"Probe orig={0} dest={1} mpi_tag={2} user_tag={3}",orig,dest,mpi_tag,user_tag);
      message_id = m_mpi_adapter->probeMessage(mpi_message);
    }
  }
  if (message_id.isValid()){
    // Il faut transformer le rang local retourné par les méthodes précédentes
    // en un rang global
    MessageId::SourceInfo si = message_id.sourceInfo();
    si.setRank(MessageRank(found_dest));
    message_id.setSourceInfo(si);
  }
  return message_id;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MP::MessageSourceInfo HybridMessageQueue::
legacyProbe(const MP::PointToPointMessageInfo& message)
{
  TRACE_DEBUG(1,"LegacyProbe msg='{0}' queue={1} is_valid={2}",
              message,this,message.isValid());

  MessageRank orig = message.emiterRank();
  if (orig.isNull())
    ARCANE_THROW(ArgumentException,"null sender");

  if (!message.isValid())
    return {};

  // Il faut avoir initialisé le message avec un couple (rang/tag).
  if (!message.isRankTag())
    ARCCORE_FATAL("Invalid message_info: message.isRankTag() is false");

  const MessageRank dest = message.destinationRank();
  const MessageTag user_tag = message.tag();
  const bool is_blocking = message.isBlocking();
  if (is_blocking)
    ARCANE_THROW(NotImplementedException,"blocking probe");
  if (user_tag.isNull())
    ARCANE_THROW(NotImplementedException,"legacyProbe with ANY_TAG");
  FullRankInfo orig_fri = m_rank_tag_builder.rank(orig);
  FullRankInfo dest_fri = m_rank_tag_builder.rank(dest);
  MP::MessageSourceInfo message_source_info;
  Int32 found_dest = dest.value();
  const bool is_any_source = dest.isNull() || dest.isAnySource();
  if (dest.isNull() && !m_is_allow_null_rank_for_any_source)
    ARCANE_FATAL("Can not use legacyProbe() with null rank. Use MessageRank::anySourceRank() instead");
  if (is_any_source) {
    // Comme on ne sait pas de qui on va recevoir, il faut tester à la
    // fois la file de thread et via MPI.
    MP::PointToPointMessageInfo p2p_message(message);
    p2p_message.setEmiterRank(orig_fri.localRank());
    message_source_info = m_thread_queue->legacyProbe(p2p_message);
    if (message_source_info.isValid()){
      // On a trouvé un message dans la liste de thread.
      // Comme on est dans notre liste de thread, le
      // rang global est notre rang MPI + le rang local trouvé.
      found_dest = orig_fri.mpiRankValue()*m_local_nb_rank + message_source_info.rank().value();
      TRACE_DEBUG(2,"LegacyProbe with null_rank (thread) orig={0} found_dest={1} tag={2}",
                  orig,found_dest,user_tag);
    }
    else{
      // Recherche via MPI.
      // La difficulté est que le rang local du PE originaire du message
      // est codé dans le tag et qu'on ne connait pas le PE originaire.
      // Il faut donc tester tous les tag potentiels. Leur nombre est
      // égal à 'm_nb_local_rank'.
      for( Integer z=0, zn=m_local_nb_rank; z<zn; ++z ){
        MP::PointToPointMessageInfo mpi_message(message);
        MessageTag mpi_tag = m_rank_tag_builder.tagForReceive(user_tag,orig_fri.localRank(),MessageRank(z));
        mpi_message.setTag(mpi_tag);
        TRACE_DEBUG(2,"LegacyProbe with null_rank orig={0} dest={1} tag={2}",orig,dest,mpi_tag);
        message_source_info = m_mpi_adapter->legacyProbeMessage(mpi_message);
        if (message_source_info.isValid()){
          // On a trouvé un message MPI. Il faut extraire du tag le
          // rang local. Le rang MPI est celui dans le message.
          MessageRank mpi_rank = message_source_info.rank();
          MessageTag ret_tag = message_source_info.tag();
          Int32 local_rank = m_rank_tag_builder.getReceiveRankFromTag(ret_tag);
          found_dest = mpi_rank.value()*m_local_nb_rank + local_rank;
          TRACE_DEBUG(2,"LegacyProbe null rank found mpi_rank={0} local_rank={1} tag={2}",
                      ret_tag,mpi_rank,local_rank,ret_tag);
          // Remet le tag d'origine pour pouvoir faire un receive avec.
          message_source_info.setTag(user_tag);
          break;
        }
      }
    }
  }
  else{
    // Il faut convertir le rang `dest` en le rang attendu par la file de thread
    // ou par MPI.
    if (orig_fri.mpiRank()==dest_fri.mpiRank()){
      MP::PointToPointMessageInfo p2p_message(message);
      p2p_message.setDestinationRank(MP::MessageRank(dest_fri.localRank()));
      p2p_message.setEmiterRank(MessageRank(orig_fri.localRank()));
      TRACE_DEBUG(2,"LegacyProbe SHM orig={0} dest={1} tag={2}",orig,dest,user_tag);
      message_source_info = m_thread_queue->legacyProbe(p2p_message);
    }
    else{
      MP::PointToPointMessageInfo mpi_message(message);
      MessageTag mpi_tag = m_rank_tag_builder.tagForReceive(user_tag,orig_fri,dest_fri);
      mpi_message.setTag(mpi_tag);
      mpi_message.setDestinationRank(MP::MessageRank(dest_fri.mpiRank()));
      TRACE_DEBUG(2,"LegacyProbe MPI orig={0} dest={1} mpi_tag={2} user_tag={3}",orig,dest,mpi_tag,user_tag);
      message_source_info = m_mpi_adapter->legacyProbeMessage(mpi_message);
      if (message_source_info.isValid()){
        // Remet le tag d'origine pour pouvoir faire un receive avec.
        message_source_info.setTag(user_tag);
      }
    }
  }
  if (message_source_info.isValid()){
    // Il faut transformer le rang local retourné par les méthodes précédentes
    // en un rang global
    message_source_info.setRank(MessageRank(found_dest));
  }
  TRACE_DEBUG(2,"LegacyProbe has matched message? = {0}",message_source_info.isValid());
  return message_source_info;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

std::ostream& operator<<(std::ostream& o,const FullRankInfo& fri)
{
  return o << "(local=" << fri.m_local_rank << ",global="
           << fri.m_global_rank << ",mpi=" << fri.m_mpi_rank << ")";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::MessagePassing

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
