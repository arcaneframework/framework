// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ParallelSuperMngDispatcher.cc                               (C) 2000-2005 */
/*                                                                           */
/* Redirection de la gestion des messages suivant le type des arguments.     */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/


#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/utils/ArrayView.h"
#include "arcane/utils/Iostream.h"

#include "arcane/ParallelSuperMngDispatcher.h"
#include "arcane/IParallelDispatch.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ParallelSuperMngDispatcher::
ParallelSuperMngDispatcher()
: m_byte(0)
, m_int32(0)
, m_int64(0)
, m_real(0)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ParallelSuperMngDispatcher::
~ParallelSuperMngDispatcher()
{
  delete m_byte;
  delete m_int32;
  delete m_int64;
  delete m_real;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParallelSuperMngDispatcher::
_setDispatchers(IParallelDispatchT<Byte>* c,IParallelDispatchT<Int32>* i32,
                IParallelDispatchT<Int64>* i64,IParallelDispatchT<Real>* r)
{
  m_byte = c;
  m_int32 = i32;
  m_int64 = i64;
  m_real = r;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void  ParallelSuperMngDispatcher::
_finalize()
{
  if (m_byte)
    m_byte->finalize();
  if (m_int32)
    m_int32->finalize();
  if (m_int64)
    m_int64->finalize();
  if (m_real)
    m_real->finalize();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParallelSuperMngDispatcher::
allGather(ByteConstArrayView send_buf,ByteArrayView recv_buf)
{ m_byte->allGather(send_buf,recv_buf); }
void ParallelSuperMngDispatcher::
allGather(Int32ConstArrayView send_buf,Int32ArrayView recv_buf)
{ m_int32->allGather(send_buf,recv_buf); }
void ParallelSuperMngDispatcher::
allGather(Int64ConstArrayView send_buf,Int64ArrayView recv_buf)
{ m_int64->allGather(send_buf,recv_buf); }
void ParallelSuperMngDispatcher::
allGather(RealConstArrayView send_buf,RealArrayView recv_buf)
{ m_real->allGather(send_buf,recv_buf); }

Int32 ParallelSuperMngDispatcher::
reduce(eReduceType rt,Int32 v)
{ return m_int32->allReduce(rt,v); }
Int64 ParallelSuperMngDispatcher::
reduce(eReduceType rt,Int64 v)
{ return m_int64->allReduce(rt,v); }
Real ParallelSuperMngDispatcher::
reduce(eReduceType rt,Real v)
{ return m_real->allReduce(rt,v); }

void ParallelSuperMngDispatcher::
reduce(eReduceType rt,Int32ArrayView v)
{ m_int32->allReduce(rt,v); }
void ParallelSuperMngDispatcher::
reduce(eReduceType rt,Int64ArrayView v)
{ m_int64->allReduce(rt,v); }
void ParallelSuperMngDispatcher::
reduce(eReduceType rt,RealArrayView v)
{ m_real->allReduce(rt,v); }

void ParallelSuperMngDispatcher::
broadcast(ByteArrayView send_buf,Integer id)
{ m_byte->broadcast(send_buf,id); }
void ParallelSuperMngDispatcher::
broadcast(Int32ArrayView send_buf,Integer id)
{ m_int32->broadcast(send_buf,id); }
void ParallelSuperMngDispatcher::
broadcast(Int64ArrayView send_buf,Integer id)
{ m_int64->broadcast(send_buf,id); }
void ParallelSuperMngDispatcher::
broadcast(RealArrayView send_buf,Integer id)
{ m_real->broadcast(send_buf,id); }

void ParallelSuperMngDispatcher::
send(ByteConstArrayView values,Integer id)
{ m_byte->send(values,id); }
void ParallelSuperMngDispatcher::
send(Int32ConstArrayView values,Integer id)
{ m_int32->send(values,id); }
void ParallelSuperMngDispatcher::
send(Int64ConstArrayView values,Integer id)
{ m_int64->send(values,id); }
void ParallelSuperMngDispatcher::
send(RealConstArrayView values,Integer id)
{ m_real->send(values,id); }

void ParallelSuperMngDispatcher::
recv(ByteArrayView values,Integer id)
{ m_byte->recv(values,id); }
void ParallelSuperMngDispatcher::
recv(Int32ArrayView values,Integer id)
{ m_int32->recv(values,id); }
void ParallelSuperMngDispatcher::
recv(Int64ArrayView values,Integer id)
{ m_int64->recv(values,id); }
void ParallelSuperMngDispatcher::
recv(RealArrayView values,Integer id)
{ m_real->recv(values,id); }

Parallel::Request ParallelSuperMngDispatcher::
send(ByteConstArrayView values,Integer id,bool is_blocked)
{ return m_byte->send(values,id,is_blocked); }
Parallel::Request ParallelSuperMngDispatcher::
send(Int32ConstArrayView values,Integer id,bool is_blocked)
{ return m_int32->send(values,id,is_blocked); }
Parallel::Request ParallelSuperMngDispatcher::
send(Int64ConstArrayView values,Integer id,bool is_blocked)
{ return m_int64->send(values,id,is_blocked); }
Parallel::Request ParallelSuperMngDispatcher::
send(RealConstArrayView values,Integer id,bool is_blocked)
{ return m_real->send(values,id,is_blocked); }

Parallel::Request ParallelSuperMngDispatcher::
recv(ByteArrayView values,Integer id,bool is_blocked)
{ return m_byte->recv(values,id,is_blocked); }
Parallel::Request ParallelSuperMngDispatcher::
recv(Int32ArrayView values,Integer id,bool is_blocked)
{ return m_int32->recv(values,id,is_blocked); }
Parallel::Request ParallelSuperMngDispatcher::
recv(Int64ArrayView values,Integer id,bool is_blocked)
{ return m_int64->recv(values,id,is_blocked); }
Parallel::Request ParallelSuperMngDispatcher::
recv(RealArrayView values,Integer id,bool is_blocked)
{ return m_real->recv(values,id,is_blocked); }

void ParallelSuperMngDispatcher::
sendRecv(ByteConstArrayView send_buf,ByteArrayView recv_buf,Integer id)
{ m_byte->sendRecv(send_buf,recv_buf,id); }
void ParallelSuperMngDispatcher::
sendRecv(Int32ConstArrayView send_buf,Int32ArrayView recv_buf,Integer id)
{ m_int32->sendRecv(send_buf,recv_buf,id); }
void ParallelSuperMngDispatcher::
sendRecv(Int64ConstArrayView send_buf,Int64ArrayView recv_buf,Integer id)
{ m_int64->sendRecv(send_buf,recv_buf,id); }
void ParallelSuperMngDispatcher::
sendRecv(RealConstArrayView send_buf,RealArrayView recv_buf,Integer id)
{ m_real->sendRecv(send_buf,recv_buf,id); }

void ParallelSuperMngDispatcher::
allToAll(ByteConstArrayView send_buf,ByteArrayView recv_buf,Integer count)
{ m_byte->allToAll(send_buf,recv_buf,count); }
void ParallelSuperMngDispatcher::
allToAll(Int32ConstArrayView send_buf,Int32ArrayView recv_buf,Integer count)
{ m_int32->allToAll(send_buf,recv_buf,count); }
void ParallelSuperMngDispatcher::
allToAll(Int64ConstArrayView send_buf,Int64ArrayView recv_buf,Integer count)
{ m_int64->allToAll(send_buf,recv_buf,count); }
void ParallelSuperMngDispatcher::
allToAll(RealConstArrayView send_buf,RealArrayView recv_buf,Integer count)
{ m_real->allToAll(send_buf,recv_buf,count); }

Int32 ParallelSuperMngDispatcher::
scan(eReduceType rt,Int32 v)
{ return m_int32->scan(rt,v); }
Int64 ParallelSuperMngDispatcher::
scan(eReduceType rt,Int64 v)
{ return m_int64->scan(rt,v); }
Real ParallelSuperMngDispatcher::
scan(eReduceType rt,Real v)
{ return m_real->scan(rt,v); }

void ParallelSuperMngDispatcher::
scan(eReduceType rt,Int32ArrayView v)
{ m_int32->scan(rt,v); }
void ParallelSuperMngDispatcher::
scan(eReduceType rt,Int64ArrayView v)
{ m_int64->scan(rt,v); }
void ParallelSuperMngDispatcher::
scan(eReduceType rt,RealArrayView v)
{ m_real->scan(rt,v); }

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

