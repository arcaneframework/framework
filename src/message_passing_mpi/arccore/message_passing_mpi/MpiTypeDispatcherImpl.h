// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
/*---------------------------------------------------------------------------*/
/* MpiTypeDispatcherImpl.h                                     (C) 2000-2020 */
/*                                                                           */
/* Implémentation de 'MpiTypeDispatcher'.                                    */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_MESSAGEPASSINGMPI_MPITYPEDISPATCHERIMPL_H
#define ARCCORE_MESSAGEPASSINGMPI_MPITYPEDISPATCHERIMPL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing_mpi/MpiTypeDispatcher.h"
#include "arccore/message_passing_mpi/MpiDatatype.h"
#include "arccore/message_passing_mpi/MpiAdapter.h"
#include "arccore/message_passing_mpi/MpiLock.h"

#include "arccore/message_passing/Messages.h"
#include "arccore/message_passing/Request.h"

#include "arccore/base/NotSupportedException.h"
#include "arccore/base/NotImplementedException.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore
{
namespace MessagePassing
{
namespace Mpi
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class Type> MpiTypeDispatcher<Type>::
MpiTypeDispatcher(IMessagePassingMng* parallel_mng,MpiAdapter* adapter,MpiDatatype* datatype)
: m_parallel_mng(parallel_mng)
, m_adapter(adapter)
, m_datatype(datatype)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class Type> MpiTypeDispatcher<Type>::
~MpiTypeDispatcher()
{
  m_datatype = nullptr;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class Type> void MpiTypeDispatcher<Type>::
broadcast(Span<Type> send_buf,Int32 rank)
{
  MPI_Datatype type = m_datatype->datatype();
  m_adapter->broadcast(send_buf.data(),send_buf.size(),rank,type);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class Type> void MpiTypeDispatcher<Type>::
allGather(Span<const Type> send_buf,Span<Type> recv_buf)
{
  MPI_Datatype type = m_datatype->datatype();
  m_adapter->allGather(send_buf.data(),recv_buf.data(),send_buf.size(),type);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class Type> void MpiTypeDispatcher<Type>::
gather(Span<const Type> send_buf,Span<Type> recv_buf,Int32 rank)
{
  MPI_Datatype type = m_datatype->datatype();
  m_adapter->gather(send_buf.data(),recv_buf.data(),send_buf.size(),rank,type);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class Type> void MpiTypeDispatcher<Type>::
allGatherVariable(Span<const Type> send_buf,Array<Type>& recv_buf)
{
  _gatherVariable2(send_buf,recv_buf,-1);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class Type> void MpiTypeDispatcher<Type>::
gatherVariable(Span<const Type> send_buf,Array<Type>& recv_buf,Int32 rank)
{
  _gatherVariable2(send_buf,recv_buf,rank);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class Type> void MpiTypeDispatcher<Type>::
_gatherVariable2(Span<const Type> send_buf,Array<Type>& recv_buf,Int32 rank)
{
  MPI_Datatype type = m_datatype->datatype();

  Int32 comm_size = m_parallel_mng->commSize();
  UniqueArray<int> send_counts(comm_size);
  UniqueArray<int> send_indexes(comm_size);

  Int64 nb_elem = send_buf.size();
  int my_buf_count = (int)nb_elem;
  Span<const int> count_r(&my_buf_count,1);

  // Récupère le nombre d'éléments de chaque processeur
  if (rank!=(-1))
    mpGather(m_parallel_mng,count_r,send_counts,rank);
  else
    mpAllGather(m_parallel_mng,count_r,send_counts);

  // Remplit le tableau des index
  if (rank==(-1) || rank==m_adapter->commRank()){
    Int64 index = 0;
    for( Integer i=0, is=comm_size; i<is; ++i ){
      send_indexes[i] = (int)index;
      index += send_counts[i];
      //info() << " SEND " << i << " index=" << send_indexes[i] << " count=" << send_counts[i];
    }
    Int64 i64_total_elem = index;
    Int64 max_size = ARCCORE_INT64_MAX;
    if (i64_total_elem>max_size){
      ARCCORE_FATAL("Invalid size '{0}'",i64_total_elem);
    }
    Int64 total_elem = i64_total_elem;
    recv_buf.resize(total_elem);
  }

  if (rank!=(-1)){
    m_adapter->gatherVariable(send_buf.data(),recv_buf.data(),send_counts.data(),
                              send_indexes.data(),nb_elem,rank,type);
  }
  else{
    m_adapter->allGatherVariable(send_buf.data(),recv_buf.data(),send_counts.data(),
                                 send_indexes.data(),nb_elem,type);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class Type> void MpiTypeDispatcher<Type>::
scatterVariable(Span<const Type> send_buf,Span<Type> recv_buf,Int32 root)
{
  MPI_Datatype type = m_datatype->datatype();

  Int32 comm_size = m_adapter->commSize();
  UniqueArray<int> recv_counts(comm_size);
  UniqueArray<int> recv_indexes(comm_size);

  Int64 nb_elem = recv_buf.size();
  int my_buf_count = m_adapter->toMPISize(nb_elem);
  Span<const int> count_r(&my_buf_count,1);

  // Récupère le nombre d'éléments de chaque processeur
  mpAllGather(m_parallel_mng,count_r,recv_counts);

  // Remplit le tableau des index
  int index = 0;
  for( Integer i=0, is=comm_size; i<is; ++i ){
    recv_indexes[i] = index;
    index += recv_counts[i];
  }

  m_adapter->scatterVariable(send_buf.data(),recv_counts.data(),recv_indexes.data(),
                             recv_buf.data(),nb_elem,root,type);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class Type> void MpiTypeDispatcher<Type>::
allToAll(Span<const Type> send_buf,Span<Type> recv_buf,Int32 count)
{
  MPI_Datatype type = m_datatype->datatype();
  m_adapter->allToAll(send_buf.data(),recv_buf.data(),count,type);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class Type> void MpiTypeDispatcher<Type>::
allToAllVariable(Span<const Type> send_buf,
                 Int32ConstArrayView send_count,
                 Int32ConstArrayView send_index,
                 Span<Type> recv_buf,
                 Int32ConstArrayView recv_count,
                 Int32ConstArrayView recv_index
                 )
{
  MPI_Datatype type = m_datatype->datatype();

  m_adapter->allToAllVariable(send_buf.data(),send_count.data(),
                              send_index.data(),recv_buf.data(),
                              recv_count.data(),
                              recv_index.data(),type);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class Type> Request MpiTypeDispatcher<Type>::
send(Span<const Type> send_buffer,Int32 rank,bool is_blocked)
{
  MPI_Datatype type = m_datatype->datatype();
  return m_adapter->directSend(send_buffer.data(),send_buffer.size(),
                               rank,sizeof(Type),type,100,is_blocked);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class Type> Request MpiTypeDispatcher<Type>::
receive(Span<Type> recv_buffer,Int32 rank,bool is_blocked)
{
  MPI_Datatype type = m_datatype->datatype();
  MpiLock::Section mls(m_adapter->mpiLock());
  return m_adapter->directRecv(recv_buffer.data(),recv_buffer.size(),
                               rank,sizeof(Type),type,100,is_blocked);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class Type> Request MpiTypeDispatcher<Type>::
send(Span<const Type> send_buffer,PointToPointMessageInfo message)
{
  MPI_Datatype type = m_datatype->datatype();
  Int64 sizeof_type = sizeof(Type);
  MpiLock::Section mls(m_adapter->mpiLock());
  bool is_blocking = message.isBlocking();
  if (message.isRankTag()){
    return m_adapter->directSend(send_buffer.data(),send_buffer.size(),message.rank(),
                                 sizeof_type,type,message.tag().value(),is_blocking);
  }
  if (message.isMessageId()){
    // Le send avec un MessageId n'existe pas.
    ARCCORE_THROW(NotSupportedException,"Invalid generic send with MessageId");
  }
  ARCCORE_THROW(NotSupportedException,"Invalid message_info");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class Type> Request MpiTypeDispatcher<Type>::
receive(Span<Type> recv_buffer,PointToPointMessageInfo message)
{
  MPI_Datatype type = m_datatype->datatype();
  Int64 sizeof_type = sizeof(Type);
  MpiLock::Section mls(m_adapter->mpiLock());
  bool is_blocking = message.isBlocking();
  if (message.isRankTag()){
    return m_adapter->directRecv(recv_buffer.data(),recv_buffer.size(),
                                 message.rank(),sizeof_type,type,message.tag().value(),
                                 is_blocking);
  }
  if (message.isMessageId()){
    MessageId message_id = message.messageId();
    return m_adapter->directRecv(recv_buffer.data(),recv_buffer.size(),
                                 message_id,sizeof_type,type,is_blocking);
  }
  ARCCORE_THROW(NotSupportedException,"Invalid message_info");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class Type> Type MpiTypeDispatcher<Type>::
allReduce(eReduceType op,Type send_buf)
{
  MPI_Datatype type = m_datatype->datatype();
  Type recv_buf = send_buf;
  MPI_Op operation = m_datatype->reduceOperator(op);
  m_adapter->allReduce(&send_buf,&recv_buf,1,type,operation);
  return recv_buf;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class Type> void MpiTypeDispatcher<Type>::
allReduce(eReduceType op,Span<Type> send_buf)
{
  MPI_Datatype type = m_datatype->datatype();
  Int64 s = send_buf.size();
  UniqueArray<Type> recv_buf(s);
  MPI_Op operation = m_datatype->reduceOperator(op);
  {
    MpiLock::Section mls(m_adapter->mpiLock());
    m_adapter->allReduce(send_buf.data(),recv_buf.data(),s,type,operation);
  }
  send_buf.copy(recv_buf);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Mpi
} // End namespace MessagePassing
} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
