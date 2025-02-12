// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MpiTypeDispatcherImpl.h                                     (C) 2000-2025 */
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
#include "arccore/message_passing/GatherMessageInfo.h"

#include "arccore/base/NotSupportedException.h"
#include "arccore/base/NotImplementedException.h"

#include "arccore/collections/Array.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MessagePassing::Mpi
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
  if (m_is_destroy_datatype)
    delete m_datatype;
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
  Int32 comm_size = m_parallel_mng->commSize();
  UniqueArray<int> send_counts(comm_size);
  UniqueArray<int> send_indexes(comm_size);

  Int64 nb_elem = send_buf.size();
  int my_buf_count = (int)nb_elem;
  Span<const int> count_r(&my_buf_count,1);

  // Récupère le nombre d'éléments de chaque processeur
  if (rank!=A_NULL_RANK)
    mpGather(m_parallel_mng,count_r,send_counts,rank);
  else
    mpAllGather(m_parallel_mng,count_r,send_counts);

  // Remplit le tableau des index
  if (rank==A_NULL_RANK || rank==m_adapter->commRank()){
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
  gatherVariable(send_buf,recv_buf,send_counts,send_indexes,rank);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class Type> void MpiTypeDispatcher<Type>::
gatherVariable(Span<const Type> send_buf,Span<Type> recv_buf,Span<const Int32> send_counts,
               Span<const Int32> displacements,Int32 rank)
{
  MPI_Datatype type = m_datatype->datatype();
  Int32 nb_elem = send_buf.smallView().size();
  if (rank!=A_NULL_RANK){
    m_adapter->gatherVariable(send_buf.data(),recv_buf.data(),send_counts.data(),
                              displacements.data(),nb_elem,rank,type);
  }
  else{
    m_adapter->allGatherVariable(send_buf.data(),recv_buf.data(),send_counts.data(),
                                 displacements.data(),nb_elem,type);
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
                 ConstArrayView<Int32> send_count,
                 ConstArrayView<Int32> send_index,
                 Span<Type> recv_buf,
                 ConstArrayView<Int32> recv_count,
                 ConstArrayView<Int32> recv_index
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
send(Span<const Type> send_buffer,const PointToPointMessageInfo& message)
{
  MPI_Datatype type = m_datatype->datatype();
  Int64 sizeof_type = sizeof(Type);
  MpiLock::Section mls(m_adapter->mpiLock());
  bool is_blocking = message.isBlocking();
  if (message.isRankTag()){
    return m_adapter->directSend(send_buffer.data(),send_buffer.size(),
                                 message.destinationRank().value(),
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
receive(Span<Type> recv_buffer,const PointToPointMessageInfo& message)
{
  MPI_Datatype type = m_datatype->datatype();
  Int64 sizeof_type = sizeof(Type);
  MpiLock::Section mls(m_adapter->mpiLock());
  bool is_blocking = message.isBlocking();
  if (message.isRankTag()){
    return m_adapter->directRecv(recv_buffer.data(),recv_buffer.size(),
                                 message.destinationRank().value(),sizeof_type,type,
                                 message.tag().value(),
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

template<class Type> Request MpiTypeDispatcher<Type>::
nonBlockingAllReduce(eReduceType op,Span<const Type> send_buf,Span<Type> recv_buf)
{
  MPI_Datatype type = m_datatype->datatype();
  Int64 s = send_buf.size();
  MPI_Op operation = m_datatype->reduceOperator(op);
  Request request;
  {
    MpiLock::Section mls(m_adapter->mpiLock());
    request = m_adapter->nonBlockingAllReduce(send_buf.data(),recv_buf.data(),s,type,operation);
  }
  return request;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class Type> Request MpiTypeDispatcher<Type>::
nonBlockingAllToAll(Span<const Type> send_buf,Span<Type> recv_buf,Int32 count)
{
  MPI_Datatype type = m_datatype->datatype();
  return m_adapter->nonBlockingAllToAll(send_buf.data(),recv_buf.data(),count,type);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class Type> Request MpiTypeDispatcher<Type>::
nonBlockingAllToAllVariable(Span<const Type> send_buf,
                            ConstArrayView<Int32> send_count,
                            ConstArrayView<Int32> send_index,
                            Span<Type> recv_buf,
                            ConstArrayView<Int32> recv_count,
                            ConstArrayView<Int32> recv_index
                            )
{
  MPI_Datatype type = m_datatype->datatype();

  return m_adapter->nonBlockingAllToAllVariable(send_buf.data(),send_count.data(),
                                                send_index.data(),recv_buf.data(),
                                                recv_count.data(),
                                                recv_index.data(),type);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class Type> Request MpiTypeDispatcher<Type>::
nonBlockingBroadcast(Span<Type> send_buf,Int32 rank)
{
  MPI_Datatype type = m_datatype->datatype();
  return m_adapter->nonBlockingBroadcast(send_buf.data(),send_buf.size(),rank,type);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class Type> Request MpiTypeDispatcher<Type>::
nonBlockingAllGather(Span<const Type> send_buf,Span<Type> recv_buf)
{
  MPI_Datatype type = m_datatype->datatype();
  return m_adapter->nonBlockingAllGather(send_buf.data(),recv_buf.data(),send_buf.size(),type);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class Type> Request MpiTypeDispatcher<Type>::
nonBlockingGather(Span<const Type> send_buf,Span<Type> recv_buf,Int32 rank)
{
  MPI_Datatype type = m_datatype->datatype();
  return m_adapter->nonBlockingGather(send_buf.data(),recv_buf.data(),send_buf.size(),rank,type);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class Type> Request MpiTypeDispatcher<Type>::
gather(GatherMessageInfo<Type>& gather_info)
{
  if (!gather_info.isValid())
    return {};

  bool is_blocking = gather_info.isBlocking();
  MessageRank dest_rank = gather_info.destinationRank();
  bool is_all_variant = dest_rank.isNull();
  MessageRank my_rank(m_parallel_mng->commRank());

  auto send_buf = gather_info.sendBuffer();

  // GatherVariable avec envoi gather préliminaire pour connaitre la taille
  // que doit envoyer chaque rang.
  if (gather_info.mode()==GatherMessageInfoBase::Mode::GatherVariableNeedComputeInfo) {
    if (!is_blocking)
      ARCCORE_THROW(NotSupportedException,"non blocking version of AllGatherVariable or GatherVariable with compute info");
    Array<Type>* receive_array = gather_info.localReceptionBuffer();
    if (is_all_variant){
      if (!receive_array)
        ARCCORE_FATAL("local reception buffer is null");
      this->allGatherVariable(send_buf, *receive_array);
    }
    else{
      UniqueArray<Type> unused_array;
      if (dest_rank==my_rank)
        this->gatherVariable(send_buf, *receive_array, dest_rank.value());
      else
        this->gatherVariable(send_buf, unused_array, dest_rank.value());
    }
    return {};
  }

  // GatherVariable classique avec connaissance du déplacement et des tailles
  if (gather_info.mode() == GatherMessageInfoBase::Mode::GatherVariable) {
    if (!is_blocking)
      ARCCORE_THROW(NotImplementedException, "non blocking version of AllGatherVariable or GatherVariable");
    auto receive_buf = gather_info.receiveBuffer();
    auto displacements = gather_info.receiveDisplacement();
    auto receive_counts = gather_info.receiveCounts();
    gatherVariable(send_buf, receive_buf, receive_counts, displacements, dest_rank.value());
    return {};
  }

  // Gather classique
  if (gather_info.mode() == GatherMessageInfoBase::Mode::Gather) {
    auto receive_buf = gather_info.receiveBuffer();
    if (is_blocking) {
      if (is_all_variant)
        this->allGather(send_buf, receive_buf);
      else
        this->gather(send_buf, receive_buf, dest_rank.value());
      return {};
    }
    else{
      if (is_all_variant)
        return this->nonBlockingAllGather(send_buf, receive_buf);
      else
        return this->nonBlockingGather(send_buf, receive_buf, dest_rank.value());
    }
  }

  ARCCORE_THROW(NotImplementedException,"Unknown type() for GatherMessageInfo");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore::MessagePassing::Mpi

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
