// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* TransferValuesParallelOperation.cc                          (C) 2000-2025 */
/*                                                                           */
/* Transfert de valeurs sur différents processeurs.                          */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Array.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/NotImplementedException.h"
#include "arcane/utils/ArgumentException.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/CheckedConvert.h"

#include "arcane/core/IParallelMng.h"
#include "arcane/core/Timer.h"
#include "arcane/core/ISerializer.h"
#include "arcane/core/internal/SerializeMessage.h"

#include "arcane/impl/TransferValuesParallelOperation.h"

#include "arccore/message_passing/PointToPointSerializerMng.h"

#include <map>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
using namespace Arccore::MessagePassing;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TransferValuesParallelOperation::
TransferValuesParallelOperation(IParallelMng* pm)
: m_parallel_mng(pm)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TransferValuesParallelOperation::
~TransferValuesParallelOperation()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IParallelMng* TransferValuesParallelOperation::
parallelMng()
{
  return m_parallel_mng;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TransferValuesParallelOperation::
setTransferRanks(Int32ConstArrayView ranks)
{
  m_ranks = ranks;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TransferValuesParallelOperation::
addArray(Int32ConstArrayView send_values,SharedArray<Int32> recv_values)
{
  m_send32_values.add(send_values);
  m_recv32_values.add(recv_values);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TransferValuesParallelOperation::
addArray(Int64ConstArrayView send_values,SharedArray<Int64> recv_values)
{
  m_send64_values.add(send_values);
  m_recv64_values.add(recv_values);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TransferValuesParallelOperation::
addArray(RealConstArrayView send_values,SharedArray<Real> recv_values)
{
  m_send_real_values.add(send_values);
  m_recv_real_values.add(recv_values);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename U> void TransferValuesParallelOperation::
_putArray(ISerializer* s,
          Span<const Int32> z_indexes,
          UniqueArray< ConstArrayView<U> >& arrays,
          Array<U>& tmp_values)
{
  Int64 nb = z_indexes.size();
  tmp_values.resize(nb);
  for( Integer z=0, zs=arrays.size(); z<zs; ++z ){
    Span<const U> v = arrays[z];
    for( Integer zz=0; zz<nb; ++zz )
      tmp_values[zz] = v[z_indexes[zz]];
    s->putSpan(tmp_values);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename U> void TransferValuesParallelOperation::
_getArray(ISerializer* s, Integer nb, UniqueArray< SharedArray<U> >& arrays,
          Array<U>& tmp_values)
{
  tmp_values.resize(nb);
  for( Integer z=0, zs=arrays.size(); z<zs; ++z ){
    s->getSpan(tmp_values);
    for( Integer zz=0; zz<nb; ++zz )
      arrays[z].add(tmp_values[zz]);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void TransferValuesParallelOperation::
transferValues()
{
  String func_name = "TransfertValuesParallelOperation::transferValues()";
  IParallelMng* pm = m_parallel_mng;
  Timer::Phase tphase(pm->timeStats(),TP_Communication);

  if (!pm->isParallel()){
    throw NotImplementedException(func_name,"in sequential");
  }

  ITraceMng* trace = pm->traceMng();


  Int32 my_rank = pm->commRank();
  Int32 nb_send = m_ranks.size();

  Int64 nb_send_int32 = m_send32_values.size();
  Int64 nb_send_int64 = m_send64_values.size();
  Int64 nb_send_real = m_send_real_values.size();

  if (nb_send_int32!=m_recv32_values.size())
    throw ArgumentException(func_name,"Int32 send_array and Int32 recv_array do not have the same size");
  if (nb_send_int64!=m_recv64_values.size())
    throw ArgumentException(func_name,"Int64 send_array and Int64 recv_array do not have the same size");
  if (nb_send_real!=m_recv_real_values.size())
    throw ArgumentException(func_name,"Real send_array and Real recv_array do not have the same size");

  for( Int64 i=0; i<nb_send_int32; ++i ){
    if (m_send32_values[i].size()!=nb_send)
      throw ArgumentException(func_name,"Int32 array and ranks do not have the same size");
  }
  for( Int64 i=0; i<nb_send_int64; ++i ){
    if (m_send64_values[i].size()!=nb_send)
      throw ArgumentException(func_name,"Int64 array and ranks do not have the same size");
  }
  for( Int64 i=0; i<nb_send_real; ++i ){
    if (m_send_real_values[i].size()!=nb_send)
      throw ArgumentException(func_name,"Real array and ranks do not have the same size");
  }
  
  typedef std::map<Int32,SharedArray<Int32> > SubDomainIndexMap;
  SubDomainIndexMap sub_domain_list;
  
  for( Integer i=0; i<nb_send; ++i ){
    Int32 sd = m_ranks[i];
    if (sd==NULL_SUB_DOMAIN_ID)
      ARCANE_FATAL("null sub_domain_id");
    if (sd==my_rank)
      ARCANE_FATAL("can not transfer to myself");
  }

  for( Integer i=0; i<nb_send; ++i ){
    Int32 sd = m_ranks[i];
    Int32Array& indexes = sub_domain_list[sd];
    indexes.add(i);
  }

  UniqueArray<Int64> sub_domain_nb_to_send;
  for( SubDomainIndexMap::const_iterator b=sub_domain_list.begin();
       b!=sub_domain_list.end(); ++b ){
    Int32 sd = b->first;
    Int64 n = b->second.size();
    sub_domain_nb_to_send.add(my_rank);
    sub_domain_nb_to_send.add(sd);
    sub_domain_nb_to_send.add(n);
  }
  UniqueArray<Int64> total_sub_domain_nb_to_send;
  pm->allGatherVariable(sub_domain_nb_to_send,total_sub_domain_nb_to_send);
  SharedArray<Int32> tmp_values_int32;
  SharedArray<Int64> tmp_values_int64;
  SharedArray<Real> tmp_values_real;
  PointToPointSerializerMng serializer_mng(pm->messagePassingMng());
  for( Int64 i=0, n=total_sub_domain_nb_to_send.size(); i<n; i+=3 ){
    Int32 rank_send = CheckedConvert::toInt32(total_sub_domain_nb_to_send[i]);
    Int32 rank_recv = CheckedConvert::toInt32(total_sub_domain_nb_to_send[i+1]);
    //Integer nb_exchange = total_sub_domain_nb_to_send[i+2];
    if (rank_send==rank_recv)
      continue;
    if (rank_recv==my_rank){
      serializer_mng.addReceiveMessage(MessageRank(rank_send));
      trace->info() << " ADD RECV MESSAGE recv=" << rank_recv << " send=" << rank_send;
    }
    else if (rank_send==my_rank){
      trace->info() << " ADD SEND MESSAGE recv=" << rank_recv << " send=" << rank_send;
      //sm = new SerializeMessage(rank_send,rank_recv,ISerializeMessage::MT_Send);
      auto sm = serializer_mng.addSendMessage(MessageRank(rank_recv));
      ISerializer* s = sm->serializer();
      s->setMode(ISerializer::ModeReserve);
      IntegerConstArrayView z_indexes = sub_domain_list.find(rank_recv)->second;
      Integer nb = z_indexes.size();
      trace->info() << " ADD SIZE int32=" << nb*nb_send_int32
                    << " int64=" << nb*nb_send_int64
                    << " real=" << nb*nb_send_real;
      s->reserveInteger(1); // Pour la taille
      for( Integer k=0; k<nb; ++k ){
        s->reserveSpan(eBasicDataType::Int32,nb_send_int32);
        s->reserveSpan(eBasicDataType::Int64,nb_send_int64);
        s->reserveSpan(eBasicDataType::Real,nb_send_real);
      }
      s->allocateBuffer();
      s->setMode(ISerializer::ModePut);
      s->putInteger(nb);

      _putArray(s,z_indexes,m_send32_values,tmp_values_int32);
      _putArray(s,z_indexes,m_send64_values,tmp_values_int64);
      _putArray(s,z_indexes,m_send_real_values,tmp_values_real);
    }
  }

  auto func = [&](ISerializeMessage* sm)
  {
    if (!sm->isSend()){
      trace->info() << " GET RECV MESSAGE recv=" << sm->destination();
      ISerializer* s = sm->serializer();
      s->setMode(ISerializer::ModeGet);
      Integer nb = s->getInteger();
      trace->info() << " GET SIZE nb=" << nb;

      _getArray(s,nb,m_recv32_values,tmp_values_int32);
      _getArray(s,nb,m_recv64_values,tmp_values_int64);
      _getArray(s,nb,m_recv_real_values,tmp_values_real);
    }
  };
  serializer_mng.waitMessages(Parallel::WaitAll,func);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
