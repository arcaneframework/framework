// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ParallelNonBlockingCollectiveDispatcher.cc                  (C) 2000-2016 */
/*                                                                           */
/* Redirection de la gestion des messages suivant le type des arguments.     */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/utils/ArrayView.h"

#include "arcane/utils/Real2.h"
#include "arcane/utils/Real3.h"
#include "arcane/utils/Real2x2.h"
#include "arcane/utils/Real3x3.h"
#include "arcane/utils/HPReal.h"

#include "arcane/ParallelNonBlockingCollectiveDispatcher.h"
#include "arcane/IParallelMng.h"
#include "arcane/Timer.h"

#include <iostream>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ParallelNonBlockingCollectiveDispatcher::
ParallelNonBlockingCollectiveDispatcher(IParallelMng* pm)
: m_parallel_mng(pm)
, m_char(0)
, m_unsigned_char(0)
, m_signed_char(0)
, m_short(0)
, m_unsigned_short(0)
, m_int(0)
, m_unsigned_int(0)
, m_long(0)
, m_unsigned_long(0)
, m_long_long(0)
, m_unsigned_long_long(0)
, m_float(0)
, m_double(0)
, m_long_double(0)
, m_real2(0)
, m_real3(0)
, m_real2x2(0)
, m_real3x3(0)
, m_hpreal(0)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ParallelNonBlockingCollectiveDispatcher::
~ParallelNonBlockingCollectiveDispatcher()
{
  delete m_char;
  delete m_signed_char;
  delete m_unsigned_char;
  delete m_short;
  delete m_unsigned_short;
  delete m_int;
  delete m_unsigned_int;
  delete m_long;
  delete m_unsigned_long;
  delete m_long_long;
  delete m_unsigned_long_long;
#ifdef ARCANE_REAL_NOT_BUILTIN
  delete m_real;
#endif
  delete m_float;
  delete m_double;
  delete m_long_double;
  delete m_real2;
  delete m_real3;
  delete m_real2x2;
  delete m_real3x3;
  delete m_hpreal;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParallelNonBlockingCollectiveDispatcher::
_setDispatchers(IParallelNonBlockingCollectiveDispatchT<char>* c,
                IParallelNonBlockingCollectiveDispatchT<signed char>* sc,
                IParallelNonBlockingCollectiveDispatchT<unsigned char>* uc,
                IParallelNonBlockingCollectiveDispatchT<short>* s,
                IParallelNonBlockingCollectiveDispatchT<unsigned short>* us,
                IParallelNonBlockingCollectiveDispatchT<int>* i,
                IParallelNonBlockingCollectiveDispatchT<unsigned int>* ui,
                IParallelNonBlockingCollectiveDispatchT<long>* l,
                IParallelNonBlockingCollectiveDispatchT<unsigned long>* ul,
                IParallelNonBlockingCollectiveDispatchT<long long>* ll,
                IParallelNonBlockingCollectiveDispatchT<unsigned long long>* ull,
#ifdef ARCANE_REAL_NOT_BUILTIN
                IParallelNonBlockingCollectiveDispatchT<Real>* r,
#endif
                IParallelNonBlockingCollectiveDispatchT<float>* f,
                IParallelNonBlockingCollectiveDispatchT<double>* d,
                IParallelNonBlockingCollectiveDispatchT<long double>* ld,
                IParallelNonBlockingCollectiveDispatchT<Real2>* r2,
                IParallelNonBlockingCollectiveDispatchT<Real3>* r3,
                IParallelNonBlockingCollectiveDispatchT<Real2x2>* r22,
                IParallelNonBlockingCollectiveDispatchT<Real3x3>* r33,
                IParallelNonBlockingCollectiveDispatchT<HPReal>* hpr)
{
  m_char = c;
  m_signed_char = sc;
  m_unsigned_char = uc;
  m_short = s;
  m_unsigned_short = us;
  m_int = i;
  m_unsigned_int = ui;
  m_long = l;
  m_unsigned_long = ul;
  m_long_long = ll;
  m_unsigned_long_long = ull;
#ifdef ARCANE_REAL_NOT_BUILTIN
  m_real = r;
#endif
  m_float = f;
  m_double = d;
  m_long_double = ld;
  m_real2 = r2;
  m_real3 = r3;
  m_real2x2 = r22;
  m_real3x3 = r33;
  m_hpreal = hpr;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ITimeStats* ParallelNonBlockingCollectiveDispatcher::
timeStats()
{
  return m_parallel_mng->timeStats();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#define ARCANE_PARALLEL_MANAGER_DISPATCH(field,type)\
Parallel::Request ParallelNonBlockingCollectiveDispatcher:: \
allGather(ConstArrayView<type> send_buf,ArrayView<type> recv_buf)\
{\
  Timer::Phase tphase(timeStats(),TP_Communication);\
  return field->allGather(send_buf,recv_buf);       \
}\
Parallel::Request ParallelNonBlockingCollectiveDispatcher::\
gather(ConstArrayView<type> send_buf,ArrayView<type> recv_buf,Integer rank) \
{\
  Timer::Phase tphase(timeStats(),TP_Communication);\
  return field->gather(send_buf,recv_buf,rank);                \
}\
Parallel::Request ParallelNonBlockingCollectiveDispatcher::\
allGatherVariable(ConstArrayView<type> send_buf,Array<type>& recv_buf)\
{\
  Timer::Phase tphase(timeStats(),TP_Communication);\
  return field->allGatherVariable(send_buf,recv_buf);\
}\
Parallel::Request ParallelNonBlockingCollectiveDispatcher::\
gatherVariable(ConstArrayView<type> send_buf,Array<type>& recv_buf,Integer rank) \
{\
  Timer::Phase tphase(timeStats(),TP_Communication);\
  return field->gatherVariable(send_buf,recv_buf,rank);        \
}\
Parallel::Request ParallelNonBlockingCollectiveDispatcher::\
scatterVariable(ConstArrayView<type> send_buf,ArrayView<type> recv_buf,Integer root)\
{\
  Timer::Phase tphase(timeStats(),TP_Communication);\
  return field->scatterVariable(send_buf,recv_buf,root);\
}\
Parallel::Request ParallelNonBlockingCollectiveDispatcher::\
allReduce(eReduceType rt,ConstArrayView<type> send_buf,ArrayView<type> v)\
{\
  Timer::Phase tphase(timeStats(),TP_Communication);\
  return field->allReduce(rt,send_buf,v);\
}\
Parallel::Request ParallelNonBlockingCollectiveDispatcher::\
broadcast(ArrayView<type> send_buf,Integer id)\
{ \
  Timer::Phase tphase(timeStats(),TP_Communication);\
  return field->broadcast(send_buf,id);\
}\
Parallel::Request ParallelNonBlockingCollectiveDispatcher::\
allToAll(ConstArrayView<type> send_buf,ArrayView<type> recv_buf,Integer count)\
{\
  Timer::Phase tphase(timeStats(),TP_Communication);\
  return field->allToAll(send_buf,recv_buf,count);  \
}\
Parallel::Request ParallelNonBlockingCollectiveDispatcher::\
allToAllVariable(ConstArrayView<type> send_buf,Int32ConstArrayView send_count,\
                 Int32ConstArrayView send_index,ArrayView<type> recv_buf,\
                 Int32ConstArrayView recv_count,Int32ConstArrayView recv_index)\
{\
  Timer::Phase tphase(timeStats(),TP_Communication);\
  return field->allToAllVariable(send_buf,send_count,send_index,recv_buf,recv_count,recv_index);\
}\
IParallelNonBlockingCollectiveDispatchT<type>* ParallelNonBlockingCollectiveDispatcher::\
dispatcher(type*)\
{\
  return field;\
}\

ARCANE_PARALLEL_MANAGER_DISPATCH(m_char,char)
ARCANE_PARALLEL_MANAGER_DISPATCH(m_unsigned_char,unsigned char)
ARCANE_PARALLEL_MANAGER_DISPATCH(m_signed_char,signed char)
ARCANE_PARALLEL_MANAGER_DISPATCH(m_short,short)
ARCANE_PARALLEL_MANAGER_DISPATCH(m_unsigned_short,unsigned short)
ARCANE_PARALLEL_MANAGER_DISPATCH(m_int,int)
ARCANE_PARALLEL_MANAGER_DISPATCH(m_unsigned_int,unsigned int)
ARCANE_PARALLEL_MANAGER_DISPATCH(m_long,long)
ARCANE_PARALLEL_MANAGER_DISPATCH(m_unsigned_long,unsigned long)
ARCANE_PARALLEL_MANAGER_DISPATCH(m_long_long,long long)
ARCANE_PARALLEL_MANAGER_DISPATCH(m_unsigned_long_long,unsigned long long)
ARCANE_PARALLEL_MANAGER_DISPATCH(m_float,float)
ARCANE_PARALLEL_MANAGER_DISPATCH(m_double,double)
ARCANE_PARALLEL_MANAGER_DISPATCH(m_long_double,long double)
#ifdef ARCANE_REAL_NOT_BUILTIN
ARCANE_PARALLEL_MANAGER_DISPATCH(m_real,Real)
#endif
ARCANE_PARALLEL_MANAGER_DISPATCH(m_real2,Real2)
ARCANE_PARALLEL_MANAGER_DISPATCH(m_real3,Real3)
ARCANE_PARALLEL_MANAGER_DISPATCH(m_real2x2,Real2x2)
ARCANE_PARALLEL_MANAGER_DISPATCH(m_real3x3,Real3x3)
ARCANE_PARALLEL_MANAGER_DISPATCH(m_hpreal,HPReal)

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
