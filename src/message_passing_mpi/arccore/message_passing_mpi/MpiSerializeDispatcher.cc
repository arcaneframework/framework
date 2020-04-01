// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
/*---------------------------------------------------------------------------*/
/* MpiSerializeDispatcher.cc                                   (C) 2000-2020 */
/*                                                                           */
/* Gestion des messages de sérialisation avec MPI.                           */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing_mpi/MpiSerializeDispatcher.h"

#include "arccore/message_passing_mpi/MpiAdapter.h"
#include "arccore/message_passing_mpi/MpiMessagePassingMng.h"
#include "arccore/message_passing_mpi/MpiLock.h"
#include "arccore/message_passing/Request.h"
#include "arccore/serialize/BasicSerializer.h"
#include "arccore/base/NotImplementedException.h"
#include "arccore/base/FatalErrorException.h"
#include "arccore/base/NotSupportedException.h"
#include "arccore/base/ArgumentException.h"
#include "arccore/trace/ITraceMng.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore::MessagePassing::Mpi
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

class MpiSerializeDispatcher::SendSerializerSubRequest
: public ISubRequest
{
 public:
  SendSerializerSubRequest(MpiSerializeDispatcher* pm,BasicSerializer* buf,Int32 rank,Int32 mpi_tag)
  : m_parallel_mng(pm), m_serialize_buffer(buf), m_rank(rank), m_mpi_tag(mpi_tag) {}
  ~SendSerializerSubRequest() override
  {
  }
 public:
  Request executeOnCompletion() override
  {
    Span<Byte> bytes = m_serialize_buffer->globalBuffer();
    return m_parallel_mng->_sendSerializerBytes(bytes,m_rank,m_mpi_tag,false);
  }
  MpiSerializeDispatcher* m_parallel_mng;
  BasicSerializer* m_serialize_buffer;
  Int32 m_rank;
  Int32 m_mpi_tag;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class MpiSerializeDispatcher::ReceiveSerializerSubRequest
: public ISubRequest
{
 public:
  ReceiveSerializerSubRequest(MpiSerializeDispatcher* pm,BasicSerializer* buf,Int32 rank,Int32 mpi_tag,Integer action)
  : m_parallel_mng(pm), m_serialize_buffer(buf), m_rank(rank), m_mpi_tag(mpi_tag), m_action(action) {}
 public:
  Request executeOnCompletion() override
  {
    if (m_action==1){
      BasicSerializer* sbuf = m_serialize_buffer;
      Int64 total_recv_size = sbuf->totalSize();

      // Si le message est plus petit que le buffer, le désérialise simplement
      if (total_recv_size<=m_parallel_mng->m_serialize_buffer_size){
        sbuf->setFromSizes();
        return Request();
      }

      sbuf->preallocate(total_recv_size);
      auto bytes = sbuf->globalBuffer();
      Request r2 = m_parallel_mng->recvSerializerBytes(bytes,m_rank,m_mpi_tag,false);
      ISubRequest* sr = new ReceiveSerializerSubRequest(m_parallel_mng,m_serialize_buffer,m_rank,m_mpi_tag,2);
      r2.setSubRequest(makeRef(sr));
      return r2;
    }
    if (m_action==2){
      m_serialize_buffer->setFromSizes();
    }
    return Request();
  }
  MpiSerializeDispatcher* m_parallel_mng;
  BasicSerializer* m_serialize_buffer;
  Int32 m_rank;
  Int32 m_mpi_tag;
  Int32 m_action;
};

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

void MpiSerializeDispatcher::
_init()
{
  // Type pour la sérialisation en octet.
  MPI_Datatype mpi_datatype;
  MPI_Type_contiguous(BasicSerializer::paddingSize(),MPI_CHAR,&mpi_datatype);
  MPI_Type_commit(&mpi_datatype);
  m_byte_serializer_datatype = mpi_datatype;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Request MpiSerializeDispatcher::
sendSerializerWithTag(ISerializer* values,Int32 rank,int mpi_tag,bool is_blocking)
{
  BasicSerializer* sbuf = _castSerializer(values);
  ITraceMng* tm = m_trace;

  Span<Byte> bytes = sbuf->globalBuffer();

  Int64 total_size = sbuf->totalSize();
  _checkBigMessage(total_size);

  if (m_is_trace_serializer)
    tm->info() << "MpiParallelMng::_sendSerializer(): sending to "
               << " rank=" << rank << " bytes " << bytes.size()
               << BasicSerializer::SizesPrinter(*sbuf)
               << " tag=" << mpi_tag;

  
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
  return _sendSerializerBytes(bytes,rank,mpi_tag+1,is_blocking);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Request MpiSerializeDispatcher::
recvSerializerBytes(Span<Byte> bytes,MessageId message_id,bool is_blocking)
{
  SerializeByteConverter<Byte> sbc(bytes,m_byte_serializer_datatype);
  MPI_Datatype dt = sbc.datatype();
  return m_adapter->directRecv(sbc.data(),sbc.size(),message_id,sbc.elementSize(),dt,is_blocking);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Request MpiSerializeDispatcher::
recvSerializerBytes(Span<Byte> bytes,Int32 rank,int tag,bool is_blocking)
{
  SerializeByteConverter<Byte> sbc(bytes,m_byte_serializer_datatype);
  MPI_Datatype dt = sbc.datatype();
  return m_adapter->directRecv(sbc.data(),sbc.size(),rank,sbc.elementSize(),dt,tag,is_blocking);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Request MpiSerializeDispatcher::
_sendSerializerBytes(Span<const Byte> bytes,Int32 rank,int tag,bool is_blocking)
{
  SerializeByteConverter<const Byte> sbc(bytes,m_byte_serializer_datatype);
  MPI_Datatype dt = sbc.datatype();
  m_trace->info(4) << "_sendSerializerBytes: orig_size=" << bytes.size()
                   << " second_size=" << sbc.size()
                   << " message_size=" << sbc.messageSize();
  return m_adapter->directSend(sbc.data(),sbc.size(),rank,sbc.elementSize(),dt,tag,is_blocking);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiSerializeDispatcher::
recvSerializer2(ISerializer* values,Int32 rank,int mpi_tag)
{
  BasicSerializer* sbuf = _castSerializer(values);
  ITraceMng* tm = m_trace;

  if (m_is_trace_serializer)
    tm->info() << "MpiParallelMng::recvSerializer2() begin receive tag=" << mpi_tag;
  sbuf->preallocate(m_serialize_buffer_size);
  Span<Byte> bytes = sbuf->globalBuffer();

  recvSerializerBytes(bytes,rank,mpi_tag,true);
  Int64 total_recv_size = sbuf->totalSize();

  if (m_is_trace_serializer)
    tm->info() << "MpiParallelMng::_recvSerializer2 total_size=" << total_recv_size
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
  recvSerializerBytes(bytes,rank,mpi_tag+1,true);
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
sendSerializer(const ISerializer* s,PointToPointMessageInfo message)
{
  BasicSerializer* sbuf = _castSerializer(const_cast<ISerializer*>(s));

  Int32 rank = message.destinationRank().value();
  Int32 mpi_tag = message.tag().value();
  bool is_blocking = message.isBlocking();

  ITraceMng* tm = m_trace;

  Span<const Byte> bytes = sbuf->globalBuffer();
  Int64 total_size = sbuf->totalSize();
  _checkBigMessage(total_size);

  if (m_is_trace_serializer)
    tm->info() << "MpiParallelMng::_sendSerializer(): sending to "
               << " rank=" << rank << " bytes " << bytes.size()
               << BasicSerializer::SizesPrinter(*sbuf)
               << " tag=" << mpi_tag;

  
  // Si le message est plus petit que le buffer par défaut de sérialisation,
  // envoie tout le message
  if (total_size<=m_serialize_buffer_size){
    if (m_is_trace_serializer)
      tm->info() << "Small message size=" << bytes.size();
    return _sendSerializerBytes(bytes,rank,mpi_tag,is_blocking);
  }

  // Sinon, envoie d'abord les tailles puis une autre requête qui
  // va envoyer tout le message
  auto x = sbuf->copyAndGetSizesBuffer();
  Request r1 = _sendSerializerBytes(x,rank,mpi_tag,is_blocking);
  auto* x2 = new SendSerializerSubRequest(this,sbuf,rank,mpi_tag+1);
  r1.setSubRequest(makeRef<ISubRequest>(x2));
  return r1;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Request MpiSerializeDispatcher::
receiveSerializer(ISerializer* s,PointToPointMessageInfo message)
{
  BasicSerializer* sbuf = _castSerializer(s);
  Int32 rank = message.destinationRank().value();
  Int32 tag = message.tag().value();
  bool is_blocking = message.isBlocking();

  sbuf->preallocate(m_serialize_buffer_size);
  Span<Byte> bytes = sbuf->globalBuffer();

  Request r;
  if (message.isRankTag())
    r = recvSerializerBytes(bytes,rank,tag,is_blocking);
  else if (message.isMessageId())
    r = recvSerializerBytes(bytes,message.messageId(),is_blocking);
  else
    ARCCORE_THROW(NotSupportedException,"Only message.isRankTag() or message.isMessageId() is supported");
  auto* sr = new ReceiveSerializerSubRequest(this,sbuf,rank,tag+1,1);
  r.setSubRequest(makeRef<ISubRequest>(sr));
  return r;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiSerializeDispatcher::
broadcastSerializer(ISerializer* values,Int32 rank)
{
  BasicSerializer* sbuf = _castSerializer(values);
  ITraceMng* tm = m_trace;
  Int32 my_rank = m_adapter->commRank();
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
    Int64ArrayView total_size_buf(1,&total_size);
    m_adapter->broadcast(total_size_buf.data(),total_size_buf.size(),rank,int64_datatype);
    if (m_is_trace_serializer)
      tm->info() << "MpiSerializeDispatcher::broadcastSerializer(): sending "
                 << BasicSerializer::SizesPrinter(*sbuf);
    SerializeByteConverter<Byte> sbc(bytes,m_byte_serializer_datatype);
    m_adapter->broadcast(sbc.data(),sbc.size(),rank,sbc.datatype());
  }
  else{
    Int64 total_size = 0;
    Int64ArrayView total_size_buf(1,&total_size);
    m_adapter->broadcast(total_size_buf.data(),total_size_buf.size(),rank,int64_datatype);
    sbuf->preallocate(total_size);
    Span<Byte> bytes = sbuf->globalBuffer();
    SerializeByteConverter<Byte> sbc(bytes,m_byte_serializer_datatype);
    m_adapter->broadcast(sbc.data(),sbc.size(),rank,sbc.datatype());
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

} // namespace Arccore::MessagePassing::Mpi

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
