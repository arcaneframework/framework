// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MpiSerializeDispatcher.cc                                   (C) 2000-2025 */
/*                                                                           */
/* Gestion des messages de sérialisation avec MPI.                           */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing_mpi/internal/MpiSerializeDispatcher.h"

#include "arccore/message_passing_mpi/MpiAdapter.h"
#include "arccore/message_passing_mpi/MpiMessagePassingMng.h"
#include "arccore/message_passing_mpi/internal/MpiSerializeMessageList.h"
#include "arccore/message_passing_mpi/MpiLock.h"
#include "arccore/message_passing/Request.h"
#include "arccore/message_passing/internal/SubRequestCompletionInfo.h"
#include "arccore/serialize/BasicSerializer.h"
#include "arccore/base/NotImplementedException.h"
#include "arccore/base/FatalErrorException.h"
#include "arccore/base/NotSupportedException.h"
#include "arccore/base/ArgumentException.h"
#include "arccore/base/PlatformUtils.h"
#include "arccore/trace/ITraceMng.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MessagePassing::Mpi
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Wrappeur pour envoyer un tableau d'octets d'un sérialiseur.
 *
 * \a SpanType doit être un 'Byte' ou un 'const Byte'.
 *
 * Comme MPI utilise un 'int' pour le nombre d'éléments d'un message, on ne
 * peut pas dépasser 2^31 octets pas message. Par contre, les versions 3.0+
 * de MPI supportent des messages dont la longueur dépasse 2^31.
 * On utilise donc un type dérivé MPI contenant N octets (avec N donné
 * par SerializeBuffer::paddingSize()) et on indique à MPI que c'est ce type
 * qu'on envoie. Le nombre d'éléments est donc divisé par N ce qui permet
 * de tenir sur 'int' si la taille du message est inférieure à 2^31 * N octets
 * (en février 2019, N=128 soit des messages de 256Go maximum).
 *
 * \note Pour que cela fonctionne, le tableau \a buffer doit avoir une
 * mémoire allouée arrondie au multiple de N supérieur au nombre d'éléments
 * mais normalement cela est garanti par le SerializeBuffer.
 */
template<typename SpanType>
class SerializeByteConverter
{
 public:
  SerializeByteConverter(Span<SpanType> buffer,MPI_Datatype byte_serializer_datatype)
  : m_buffer(buffer), m_datatype(byte_serializer_datatype), m_final_size(-1)
  {
    Int64 size = buffer.size();
    const Int64 align_size = BasicSerializer::paddingSize();
    if ((size%align_size)!=0)
      ARCCORE_FATAL("Buffer size '{0}' is not a multiple of '{1}' Invalid size",size,align_size);
    m_final_size = size / align_size;
  }
  SpanType* data() { return m_buffer.data(); }
  Int64 size() const { return m_final_size; }
  Int64 messageSize() const { return m_buffer.size() * sizeof(Byte); }
  Int64 elementSize() const { return BasicSerializer::paddingSize(); }
  MPI_Datatype datatype() const { return m_datatype; }
 private:
  Span<SpanType> m_buffer;
  MPI_Datatype m_datatype;
  Int64 m_final_size;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Sous-requête d'envoi.
 *
 * Cette classe est utilisée lorsqu'un message de sérialisation est trop
 * gros pour être envoyé en une seule fois. Dans ce cas, un deuxième message
 * est envoyé. Ce deuxième message contient le message complet de sérialisation
 * car le destinataire connait la taille complète du message et peut donc
 * allouer la mémoire nécessaire.
 */
class MpiSerializeDispatcher::SendSerializerSubRequest
: public ISubRequest
{
 public:

  SendSerializerSubRequest(MpiSerializeDispatcher* pm,BasicSerializer* buf,
                           MessageRank rank,MessageTag mpi_tag)
  : m_dispatcher(pm), m_serialize_buffer(buf), m_rank(rank), m_mpi_tag(mpi_tag) {}

 public:

  Request executeOnCompletion(const SubRequestCompletionInfo&) override
  {
    if (!m_is_message_sent)
      sendMessage();
    return m_send_request;
  }
 public:
  void sendMessage()
  {
    if (m_is_message_sent)
      ARCCORE_FATAL("Message already sent");
    bool do_print  = m_dispatcher->m_is_trace_serializer;
    if (do_print){
      ITraceMng* tm = m_dispatcher->traceMng();
      tm->info() << " SendSerializerSubRequest::sendMessage()"
                 << " rank=" << m_rank << " tag=" << m_mpi_tag;
    }
    Span<Byte> bytes = m_serialize_buffer->globalBuffer();
    m_send_request = m_dispatcher->_sendSerializerBytes(bytes,m_rank,m_mpi_tag,false);
    m_is_message_sent = true;
  }
 private:
  MpiSerializeDispatcher* m_dispatcher;
  BasicSerializer* m_serialize_buffer;
  MessageRank m_rank;
  MessageTag m_mpi_tag;
  Request m_send_request;
  bool m_is_message_sent = false;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class MpiSerializeDispatcher::ReceiveSerializerSubRequest
: public ISubRequest
{
 public:

  ReceiveSerializerSubRequest(MpiSerializeDispatcher* d,BasicSerializer* buf,
                              MessageTag mpi_tag, Integer action)
  : m_dispatcher(d)
  , m_serialize_buffer(buf)
  , m_mpi_tag(mpi_tag)
  , m_action(action)
  {}

 public:

  Request executeOnCompletion(const SubRequestCompletionInfo& completion_info) override
  {
    MessageRank rank = completion_info.sourceRank();
    bool is_trace = m_dispatcher->m_is_trace_serializer;
    ITraceMng* tm = m_dispatcher->traceMng();
    if (is_trace) {
      tm->info() << " ReceiveSerializerSubRequest::executeOnCompletion()"
                 << " rank=" << rank << " wanted_tag=" << m_mpi_tag << " action=" << m_action;
    }
    if (m_action==1){
      BasicSerializer* sbuf = m_serialize_buffer;
      Int64 total_recv_size = sbuf->totalSize();

      if (is_trace) {
        tm->info() << " ReceiveSerializerSubRequest::executeOnCompletion() total_size=" << total_recv_size
                   << BasicSerializer::SizesPrinter(*m_serialize_buffer);
      }
      // Si le message est plus petit que le buffer, le désérialise simplement
      if (total_recv_size<=m_dispatcher->m_serialize_buffer_size){
        sbuf->setFromSizes();
        return {};
      }

      sbuf->preallocate(total_recv_size);
      auto bytes = sbuf->globalBuffer();

      // La nouvelle requête doit utiliser le même rang source que celui de cette requête
      // pour être certain qu'il n'y a pas d'incohérence.
      Request r2 = m_dispatcher->_recvSerializerBytes(bytes, rank, m_mpi_tag, false);
      ISubRequest* sr = new ReceiveSerializerSubRequest(m_dispatcher, m_serialize_buffer, m_mpi_tag, 2);
      r2.setSubRequest(makeRef(sr));
      return r2;
    }
    if (m_action==2){
      m_serialize_buffer->setFromSizes();
    }
    return {};
  }

 private:

  MpiSerializeDispatcher* m_dispatcher = nullptr;
  BasicSerializer* m_serialize_buffer = nullptr;
  MessageTag m_mpi_tag;
  Int32 m_action = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MpiSerializeDispatcher::
MpiSerializeDispatcher(MpiAdapter* adapter)
: m_adapter(adapter)
, m_trace(adapter->traceMng())
, m_serialize_buffer_size(50000)
//, m_serialize_buffer_size(20000000)
, m_max_serialize_buffer_size(m_serialize_buffer_size)
, m_byte_serializer_datatype(MPI_DATATYPE_NULL)
{
  _init();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MpiSerializeDispatcher::
~MpiSerializeDispatcher()
{
  if (m_byte_serializer_datatype!=MPI_DATATYPE_NULL)
    MPI_Type_free(&m_byte_serializer_datatype);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MessageTag MpiSerializeDispatcher::
nextSerializeTag(MessageTag tag)
{
  return MessageTag(tag.value()+1);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiSerializeDispatcher::
_init()
{
  // Type pour la sérialisation en octet.
  MPI_Datatype mpi_datatype;
  MPI_Type_contiguous(BasicSerializer::paddingSize(),MPI_CHAR,&mpi_datatype);
  MPI_Type_commit(&mpi_datatype);
  m_byte_serializer_datatype = mpi_datatype;

  if (!Platform::getEnvironmentVariable("ARCCORE_TRACE_MESSAGE_PASSING_SERIALIZE").empty())
    m_is_trace_serializer = true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Request MpiSerializeDispatcher::
legacySendSerializer(ISerializer* values,const PointToPointMessageInfo& message)
{
  if (!message.isRankTag())
    ARCCORE_FATAL("Only message.isRangTag()==true are allowed for legacy mode");

  MessageRank rank = message.destinationRank();
  MessageTag mpi_tag = message.tag();
  bool is_blocking = message.isBlocking();

  BasicSerializer* sbuf = _castSerializer(values);
  ITraceMng* tm = m_trace;

  Span<Byte> bytes = sbuf->globalBuffer();

  Int64 total_size = sbuf->totalSize();
  _checkBigMessage(total_size);

  if (m_is_trace_serializer)
    tm->info() << "legacySendSerializer(): sending to "
               << " rank=" << rank << " bytes " << bytes.size()
               << BasicSerializer::SizesPrinter(*sbuf)
               << " tag=" << mpi_tag << " is_blocking=" << is_blocking;

  // Si le message est plus petit que le buffer par défaut de sérialisation,
  // envoie tout le message
  if (total_size<=m_serialize_buffer_size){
    if (m_is_trace_serializer)
      tm->info() << "Small message size=" << bytes.size();
    return _sendSerializerBytes(bytes,rank,mpi_tag,is_blocking);
  }

  {
    // le message est trop grand pour tenir dans le buffer, envoie d'abord les tailles,
    // puis le message sérialisé.
    auto x = sbuf->copyAndGetSizesBuffer();
    if (m_is_trace_serializer)
      tm->info() << "Big message first size=" << x.size();
    Request r = _sendSerializerBytes(x,rank,mpi_tag,is_blocking);
    if (!is_blocking){
      SerializeSubRequest* sub_request = new SerializeSubRequest();
      sub_request->m_request = r;
      //m_trace->info() << "** ADD SUB REQUEST r=" << r;
      {
        MpiLock::Section ls(m_adapter->mpiLock());
        m_sub_requests.add(sub_request);
      }
    }
  }

  if (m_is_trace_serializer)
    tm->info() << "Big message second size=" << bytes.size();
  return _sendSerializerBytes(bytes,rank,nextSerializeTag(mpi_tag),is_blocking);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Request MpiSerializeDispatcher::
_recvSerializerBytes(Span<Byte> bytes,MessageId message_id,bool is_blocking)
{
  SerializeByteConverter<Byte> sbc(bytes,m_byte_serializer_datatype);
  MPI_Datatype dt = sbc.datatype();
  if (m_is_trace_serializer)
    m_trace->info() << "_recvSerializerBytes: size=" << bytes.size()
                    << " message_id=" << message_id << " is_blocking=" << is_blocking;
  return m_adapter->directRecv(sbc.data(),sbc.size(),message_id,sbc.elementSize(),dt,is_blocking);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Request MpiSerializeDispatcher::
_recvSerializerBytes(Span<Byte> bytes,MessageRank rank,MessageTag tag,bool is_blocking)
{
  SerializeByteConverter<Byte> sbc(bytes,m_byte_serializer_datatype);
  MPI_Datatype dt = sbc.datatype();
  if (m_is_trace_serializer)
    m_trace->info() << "_recvSerializerBytes: size=" << bytes.size()
                    << " rank=" << rank << " tag=" << tag << " is_blocking=" << is_blocking;
  Request r = m_adapter->directRecv(sbc.data(),sbc.size(),rank.value(),
                                    sbc.elementSize(),dt,tag.value(),is_blocking);
  if (m_is_trace_serializer)
    m_trace->info() << "_recvSerializerBytes: request=" << r;
  return r;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Request MpiSerializeDispatcher::
_sendSerializerBytes(Span<const Byte> bytes,MessageRank rank,MessageTag tag,
                     bool is_blocking)
{
  SerializeByteConverter<const Byte> sbc(bytes,m_byte_serializer_datatype);
  MPI_Datatype dt = sbc.datatype();
  if (m_is_trace_serializer)
    m_trace->info() << "_sendSerializerBytes: orig_size=" << bytes.size()
                    << " rank=" << rank << " tag=" << tag
                    << " second_size=" << sbc.size()
                    << " message_size=" << sbc.messageSize();
  Request  r = m_adapter->directSend(sbc.data(),sbc.size(),rank.value(),
                                     sbc.elementSize(),dt,tag.value(),is_blocking);
  if (m_is_trace_serializer)
    m_trace->info() << "_sendSerializerBytes: request=" << r;
  return r;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiSerializeDispatcher::
legacyReceiveSerializer(ISerializer* values,MessageRank rank,MessageTag mpi_tag)
{
  BasicSerializer* sbuf = _castSerializer(values);
  ITraceMng* tm = m_trace;

  if (m_is_trace_serializer)
    tm->info() << "legacyReceiveSerializer() begin receive"
               << " rank=" << rank << " tag=" << mpi_tag;
  sbuf->preallocate(m_serialize_buffer_size);
  Span<Byte> bytes = sbuf->globalBuffer();

  _recvSerializerBytes(bytes,rank,mpi_tag,true);
  Int64 total_recv_size = sbuf->totalSize();

  if (m_is_trace_serializer)
    tm->info() << "legacyReceiveSerializer total_size=" << total_recv_size
               << " from=" << rank
               << BasicSerializer::SizesPrinter(*sbuf);


  // Si le message est plus petit que le buffer, le désérialise simplement
  if (total_recv_size<=m_serialize_buffer_size){
    sbuf->setFromSizes();
    return;
  }

  if (m_is_trace_serializer)
    tm->info() << "Receive overflow buffer: " << total_recv_size;
  sbuf->preallocate(total_recv_size);
  bytes = sbuf->globalBuffer();
  _recvSerializerBytes(bytes,rank,nextSerializeTag(mpi_tag),true);
  sbuf->setFromSizes();
  if (m_is_trace_serializer)
    tm->info() << "End receive overflow buffer: " << total_recv_size;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiSerializeDispatcher::
checkFinishedSubRequests()
{
  // Regarde si les sous-requêtes sont terminées pour les libérer
  // Cela est uniquement utilisé avec le mode historique où on utilise
  // la classe 'MpiSerializeMessageList'.
  UniqueArray<SerializeSubRequest*> new_sub_requests;
  for( Integer i=0, n=m_sub_requests.size(); i<n; ++i ){
    SerializeSubRequest* ssr = m_sub_requests[i];
    bool is_finished = m_adapter->testRequest(ssr->m_request);
    if (!is_finished){
      new_sub_requests.add(ssr);
    }
    else{
      delete ssr;
    }
  }
  m_sub_requests = new_sub_requests;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiSerializeDispatcher::
_checkBigMessage(Int64 message_size)
{
  if (message_size>m_max_serialize_buffer_size){
    m_max_serialize_buffer_size = message_size;
    m_trace->info() << "big buffer: " << message_size;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Request MpiSerializeDispatcher::
sendSerializer(const ISerializer* s,const PointToPointMessageInfo& message)
{
  return sendSerializer(s,message,false);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Request MpiSerializeDispatcher::
sendSerializer(const ISerializer* s,const PointToPointMessageInfo& message,
               bool force_one_message)
{
  BasicSerializer* sbuf = _castSerializer(const_cast<ISerializer*>(s));

  MessageRank rank = message.destinationRank();
  MessageTag mpi_tag = message.tag();
  bool is_blocking = message.isBlocking();

  ITraceMng* tm = m_trace;

  Span<const Byte> bytes = sbuf->globalBuffer();
  Int64 total_size = sbuf->totalSize();
  _checkBigMessage(total_size);

  if (m_is_trace_serializer)
    tm->info() << "sendSerializer(): sending to "
               << " p2p_message=" << message
               << " rank=" << rank << " bytes " << bytes.size()
               << BasicSerializer::SizesPrinter(*sbuf)
               << " tag=" << mpi_tag
               << " total_size=" << total_size;

  
  // Si le message est plus petit que le buffer par défaut de sérialisation
  // ou qu'on choisit de n'envoyer qu'un seul message, envoie tout le message
  if (total_size<=m_serialize_buffer_size || force_one_message){
    if (m_is_trace_serializer)
      tm->info() << "Small message size=" << bytes.size();
    return _sendSerializerBytes(bytes,rank,mpi_tag,is_blocking);
  }

  // Sinon, envoie d'abord les tailles puis une autre requête qui
  // va envoyer tout le message.
  auto x = sbuf->copyAndGetSizesBuffer();
  Request r1 = _sendSerializerBytes(x,rank,mpi_tag,is_blocking);
  auto* x2 = new SendSerializerSubRequest(this,sbuf,rank,nextSerializeTag(mpi_tag));
  // Envoi directement le message pour des raisons de performance.
  x2->sendMessage();
  r1.setSubRequest(makeRef<ISubRequest>(x2));
  return r1;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Request MpiSerializeDispatcher::
receiveSerializer(ISerializer* s,const PointToPointMessageInfo& message)
{
  BasicSerializer* sbuf = _castSerializer(s);
  MessageRank rank = message.destinationRank();
  MessageTag tag = message.tag();
  bool is_blocking = message.isBlocking();

  sbuf->preallocate(m_serialize_buffer_size);
  Span<Byte> bytes = sbuf->globalBuffer();

  Request r;
  if (message.isRankTag())
    r = _recvSerializerBytes(bytes,rank,tag,is_blocking);
  else if (message.isMessageId())
    r = _recvSerializerBytes(bytes,message.messageId(),is_blocking);
  else
    ARCCORE_THROW(NotSupportedException,"Only message.isRankTag() or message.isMessageId() is supported");
  auto* sr = new ReceiveSerializerSubRequest(this, sbuf, nextSerializeTag(tag), 1);
  r.setSubRequest(makeRef<ISubRequest>(sr));
  return r;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiSerializeDispatcher::
broadcastSerializer(ISerializer* values,MessageRank rank)
{
  BasicSerializer* sbuf = _castSerializer(values);
  ITraceMng* tm = m_trace;
  MessageRank my_rank(m_adapter->commRank());
  bool is_broadcaster = (rank==my_rank);

  MPI_Datatype int64_datatype = MpiBuiltIn::datatype(Int64());
  // Effectue l'envoie en deux phases. Envoie d'abord le nombre d'éléments
  // puis envoie les éléments.
  // TODO: il serait possible de le faire en une fois pour les messages
  // ne dépassant pas une certaine taille.
  if (is_broadcaster){
    Int64 total_size = sbuf->totalSize();
    Span<Byte> bytes = sbuf->globalBuffer();
    _checkBigMessage(total_size);
    ArrayView<Int64> total_size_buf(1,&total_size);
    m_adapter->broadcast(total_size_buf.data(),total_size_buf.size(),rank.value(),int64_datatype);
    if (m_is_trace_serializer)
      tm->info() << "MpiSerializeDispatcher::broadcastSerializer(): sending "
                 << BasicSerializer::SizesPrinter(*sbuf);
    SerializeByteConverter<Byte> sbc(bytes,m_byte_serializer_datatype);
    m_adapter->broadcast(sbc.data(),sbc.size(),rank.value(),sbc.datatype());
  }
  else{
    Int64 total_size = 0;
    ArrayView<Int64> total_size_buf(1,&total_size);
    m_adapter->broadcast(total_size_buf.data(),total_size_buf.size(),rank.value(),int64_datatype);
    sbuf->preallocate(total_size);
    Span<Byte> bytes = sbuf->globalBuffer();
    SerializeByteConverter<Byte> sbc(bytes,m_byte_serializer_datatype);
    m_adapter->broadcast(sbc.data(),sbc.size(),rank.value(),sbc.datatype());
    sbuf->setFromSizes();
    if (m_is_trace_serializer)
      tm->info() << "MpiSerializeDispatcher::broadcastSerializer(): receiving from "
                 << " rank=" << rank << " bytes " << bytes.size()
                 << BasicSerializer::SizesPrinter(*sbuf);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

BasicSerializer* MpiSerializeDispatcher::
_castSerializer(ISerializer* serializer)
{
  BasicSerializer* sbuf = dynamic_cast<BasicSerializer*>(serializer);
  if (!sbuf)
    ARCCORE_THROW(ArgumentException,"Can not cast 'ISerializer' to 'BasicSerializer'");
  return sbuf;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Ref<ISerializeMessageList> MpiSerializeDispatcher::
createSerializeMessageListRef()
{
  ISerializeMessageList* x = new MpiSerializeMessageList(this);
  return makeRef(x);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arccore::MessagePassing::Mpi

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
