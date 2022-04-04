﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ParallelMngDispatcher.cc                                    (C) 2000-2022 */
/*                                                                           */
/* Redirection de la gestion des messages suivant le type des arguments.     */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Array.h"
#include "arcane/utils/Real2.h"
#include "arcane/utils/Real3.h"
#include "arcane/utils/Real2x2.h"
#include "arcane/utils/Real3x3.h"
#include "arcane/utils/HPReal.h"
#include "arcane/utils/NotImplementedException.h"
#include "arcane/utils/ScopedPtr.h"
#include "arcane/utils/Ref.h"
#include "arcane/utils/ITraceMng.h"

#include "arcane/ParallelMngDispatcher.h"
#include "arcane/IParallelDispatch.h"
#include "arcane/Timer.h"
#include "arcane/ITimeStats.h"
#include "arcane/IParallelNonBlockingCollective.h"

#include "arccore/message_passing/Dispatchers.h"
#include "arccore/message_passing/MessagePassingMng.h"
#include "arccore/message_passing/IControlDispatcher.h"
#include "arccore/message_passing/ISerializeDispatcher.h"
#include "arccore/message_passing/PointToPointMessageInfo.h"
#include "arccore/message_passing/IRequestList.h"
#include "arccore/message_passing/ISerializeMessageList.h"
#include "arccore/trace/TimeMetric.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
using namespace Arccore::MessagePassing;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ParallelMngDispatcherBuildInfo::
ParallelMngDispatcherBuildInfo(MP::Dispatchers* dispatchers,
                               MP::MessagePassingMng* mpm)
: m_comm_rank(mpm->commRank())
, m_comm_size(mpm->commSize())
, m_dispatchers(dispatchers)
, m_message_passing_mng(mpm)
{
  _init();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ParallelMngDispatcherBuildInfo::
ParallelMngDispatcherBuildInfo(Int32 comm_rank,Int32 comm_size)
: m_comm_rank(comm_rank)
, m_comm_size(comm_size)
, m_dispatchers(nullptr)
, m_message_passing_mng(nullptr)
{
  _init();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParallelMngDispatcherBuildInfo::
_init()
{
  if (!m_dispatchers)
    m_dispatchers = new MP::Dispatchers();
  if (!m_message_passing_mng)
    m_message_passing_mng = new MP::MessagePassingMng(m_comm_rank,m_comm_size,m_dispatchers);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ParallelMngDispatcher::DefaultControlDispatcher::
DefaultControlDispatcher(IParallelMng* pm)
: m_parallel_mng(pm)
{
}


void ParallelMngDispatcher::DefaultControlDispatcher::
waitAllRequests(ArrayView<Request> requests)
{
  m_parallel_mng->waitAllRequests(requests);
}

void ParallelMngDispatcher::DefaultControlDispatcher::
waitSomeRequests(ArrayView<Request> requests,ArrayView<bool> indexes,
                 bool is_non_blocking)
{
  UniqueArray<Integer> done_requests;
  if (is_non_blocking)
    done_requests = m_parallel_mng->testSomeRequests(requests);
  else
    done_requests = m_parallel_mng->waitSomeRequests(requests);
  indexes.fill(false);
  for( int x : done_requests )
    indexes[x] = true;
}

IMessagePassingMng* ParallelMngDispatcher::DefaultControlDispatcher::
commSplit(bool keep)
{
  ARCANE_UNUSED(keep);
  ARCANE_THROW(NotImplementedException,"split from MessagePassing::IControlDispatcher");
}

void ParallelMngDispatcher::DefaultControlDispatcher::
barrier()
{
  m_parallel_mng->barrier();
}

Request ParallelMngDispatcher::DefaultControlDispatcher::
nonBlockingBarrier()
{
  return m_parallel_mng->nonBlockingCollective()->barrier();
}

MessageId ParallelMngDispatcher::DefaultControlDispatcher::
probe(const PointToPointMessageInfo& message)
{
  return m_parallel_mng->probe(message);
}

Ref<Parallel::IRequestList> ParallelMngDispatcher::DefaultControlDispatcher::
createRequestListRef()
{
  return m_parallel_mng->createRequestListRef();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParallelMngDispatcher::DefaultControlDispatcher::
setProfiler(MP::IProfiler* p)
{
  ARCANE_UNUSED(p);
  ARCANE_THROW(NotImplementedException,"setProfiler() from MessagePassing::IControlDispatcher");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ParallelMngDispatcher::SerializeDispatcher
: public MP::ISerializeDispatcher
{
 public:
  SerializeDispatcher(IParallelMng* pm) : m_parallel_mng(pm){}
 public:
  Ref<ISerializeMessageList> createSerializeMessageListRef() override
  {
    return m_parallel_mng->createSerializeMessageListRef();
  }
  Request sendSerializer(const ISerializer* s,const PointToPointMessageInfo& message) override
  {
    return m_parallel_mng->sendSerializer(s,message);
  }
  Request receiveSerializer(ISerializer* s,const PointToPointMessageInfo& message) override
  {
    return m_parallel_mng->receiveSerializer(s,message);
  }
 private:
  IParallelMng* m_parallel_mng;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ParallelMngDispatcher::
ParallelMngDispatcher(const ParallelMngDispatcherBuildInfo& bi)
: m_char(nullptr)
, m_unsigned_char(nullptr)
, m_signed_char(nullptr)
, m_short(nullptr)
, m_unsigned_short(nullptr)
, m_int(nullptr)
, m_unsigned_int(nullptr)
, m_long(nullptr)
, m_unsigned_long(nullptr)
, m_long_long(nullptr)
, m_unsigned_long_long(nullptr)
, m_float(nullptr)
, m_double(nullptr)
, m_long_double(nullptr)
, m_apreal(nullptr)
, m_real2(nullptr)
, m_real3(nullptr)
, m_real2x2(nullptr)
, m_real3x3(nullptr)
, m_hpreal(nullptr)
, m_time_stats(nullptr)
, m_mp_dispatchers(bi.dispatchers())
, m_message_passing_mng(bi.messagePassingMng())
, m_control_dispatcher(new DefaultControlDispatcher(this))
, m_serialize_dispatcher(new SerializeDispatcher(this))
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ParallelMngDispatcher::
~ParallelMngDispatcher()
{
  delete m_mp_dispatchers;
  delete m_message_passing_mng;
  delete m_serialize_dispatcher;
  delete m_control_dispatcher;
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
  delete m_apreal;
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

void ParallelMngDispatcher::
_setControlDispatcher(MP::IControlDispatcher* d)
{
  delete m_control_dispatcher;
  m_control_dispatcher = d;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParallelMngDispatcher::
_setSerializeDispatcher(MP::ISerializeDispatcher* d)
{
  delete m_serialize_dispatcher;
  m_serialize_dispatcher = d;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParallelMngDispatcher::
_setArccoreDispatchers()
{
  m_mp_dispatchers->setDispatcher(m_char->toArccoreDispatcher());
  m_mp_dispatchers->setDispatcher(m_signed_char->toArccoreDispatcher());
  m_mp_dispatchers->setDispatcher(m_unsigned_char->toArccoreDispatcher());
  m_mp_dispatchers->setDispatcher(m_short->toArccoreDispatcher());
  m_mp_dispatchers->setDispatcher(m_unsigned_short->toArccoreDispatcher());
  m_mp_dispatchers->setDispatcher(m_int->toArccoreDispatcher());
  m_mp_dispatchers->setDispatcher(m_unsigned_int->toArccoreDispatcher());
  m_mp_dispatchers->setDispatcher(m_long->toArccoreDispatcher());
  m_mp_dispatchers->setDispatcher(m_unsigned_long->toArccoreDispatcher());
  m_mp_dispatchers->setDispatcher(m_long_long->toArccoreDispatcher());
  m_mp_dispatchers->setDispatcher(m_unsigned_long_long->toArccoreDispatcher());
  m_mp_dispatchers->setDispatcher(m_float->toArccoreDispatcher());
  m_mp_dispatchers->setDispatcher(m_double->toArccoreDispatcher());
  m_mp_dispatchers->setDispatcher(m_long_double->toArccoreDispatcher());

  ARCANE_CHECK_POINTER(m_control_dispatcher);
  m_mp_dispatchers->setDispatcher(m_control_dispatcher);
  ARCANE_CHECK_POINTER(m_serialize_dispatcher);
  m_mp_dispatchers->setDispatcher(m_serialize_dispatcher);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IMessagePassingMng* ParallelMngDispatcher::
messagePassingMng() const
{
  return m_message_passing_mng;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ITimeMetricCollector* ParallelMngDispatcher::
timeMetricCollector() const
{
  ITimeMetricCollector* c = nullptr;
  ITimeStats* s = m_time_stats;
  if (s)
    c = s->metricCollector();
  return c;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParallelMngDispatcher::
setTimeStats(ITimeStats* ts)
{
  m_time_stats = ts;
  if (ts){
    ITimeMetricCollector* c = ts->metricCollector();
    _messagePassingMng()->setTimeMetricCollector(c);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TimeMetricAction ParallelMngDispatcher::
_communicationTimeMetricAction() const
{
  return Timer::phaseAction(timeStats(),TP_Communication);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParallelMngDispatcher::
broadcastString(String& str,Int32 rank)
{
  Int64 len_info[1];
  Int32 my_rank = commRank();
  Span<const Byte> bytes = str.bytes();
  if (rank==my_rank){
    len_info[0] = bytes.size();
    broadcast(Int64ArrayView(1,len_info),rank);
    ByteUniqueArray utf8_array(bytes);
    broadcast(utf8_array,rank);
  }
  else{
    broadcast(Int64ArrayView(1,len_info),rank);
    ByteUniqueArray utf8_array(len_info[0]);
    broadcast(utf8_array,rank);
    str = String::fromUtf8(utf8_array);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParallelMngDispatcher::
broadcastMemoryBuffer(ByteArray& bytes,Int32 rank)
{
  Int64 size = 0;
  if (commRank()==rank){
    size = bytes.largeSize();
  }
  {
    Int64ArrayView bs(1,&size);
    broadcast(bs,rank);
  }
  if (commRank()!=rank)
    bytes.resize(size);
  if (size!=0)
    broadcast(bytes,rank);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParallelMngDispatcher::
processMessages(ConstArrayView<ISerializeMessage*> messages)
{
  TimeMetricSentry tphase(Timer::phaseAction(timeStats(),TP_Communication));
  ScopedPtrT<ISerializeMessageList> message_list(createSerializeMessageList());

  Integer nb_message = messages.size();
  for( Integer i=0; i<nb_message; ++i ){
    ISerializeMessage* m = messages[i];
    message_list->addMessage(m);
  }
  message_list->processPendingMessages();

  message_list->waitMessages(Parallel::WaitAll);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Ref<ISerializeMessageList> ParallelMngDispatcher::
createSerializeMessageListRef()
{
  return makeRef(_createSerializeMessageList());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ISerializeMessageList* ParallelMngDispatcher::
createSerializeMessageList()
{
  return _createSerializeMessageList();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IParallelMng* ParallelMngDispatcher::
createSubParallelMng(Int32ConstArrayView kept_ranks)
{
  return _createSubParallelMng(kept_ranks);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Ref<IParallelMng> ParallelMngDispatcher::
createSubParallelMngRef(Int32ConstArrayView kept_ranks)
{
  return makeRef(_createSubParallelMng(kept_ranks));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

UniqueArray<Integer> ParallelMngDispatcher::
_doWaitRequests(ArrayView<Request> requests,eWaitType wait_type)
{
  Timer::Phase tphase(timeStats(),TP_Communication);
  Ref<IRequestList> request_list(createRequestListRef());
  Integer nb_request = requests.size();
  request_list->add(requests);
  request_list->wait(wait_type);
  // Ne pas oublier de recopier les requêtes car elles ont pu être modifiées
  for (Integer i=0; i<nb_request; ++i )
    requests[i] = request_list->request(i);
  return request_list->doneRequestIndexes();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

UniqueArray<Integer> ParallelMngDispatcher::
waitSomeRequests(ArrayView<Request> requests)
{
  return _doWaitRequests(requests,Parallel::WaitSome);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

UniqueArray<Integer> ParallelMngDispatcher::
testSomeRequests(ArrayView<Request> requests)
{
  return _doWaitRequests(requests,Parallel::WaitSomeNonBlocking);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#define ARCANE_PARALLEL_MANAGER_DISPATCH(field,type)\
void ParallelMngDispatcher::\
allGather(ConstArrayView<type> send_buf,ArrayView<type> recv_buf)\
{\
  Timer::Phase tphase(timeStats(),TP_Communication);\
  field->allGather(send_buf,recv_buf);\
}\
void ParallelMngDispatcher::\
gather(ConstArrayView<type> send_buf,ArrayView<type> recv_buf,Integer rank) \
{\
  Timer::Phase tphase(timeStats(),TP_Communication);\
  field->gather(send_buf,recv_buf,rank);                \
}\
void ParallelMngDispatcher::\
allGatherVariable(ConstArrayView<type> send_buf,Array<type>& recv_buf)\
{\
  Timer::Phase tphase(timeStats(),TP_Communication);\
  field->allGatherVariable(send_buf,recv_buf);\
}\
void ParallelMngDispatcher::\
gatherVariable(ConstArrayView<type> send_buf,Array<type>& recv_buf,Integer rank) \
{\
  Timer::Phase tphase(timeStats(),TP_Communication);\
  field->gatherVariable(send_buf,recv_buf,rank);        \
}\
void ParallelMngDispatcher::\
scatterVariable(ConstArrayView<type> send_buf,ArrayView<type> recv_buf,Integer root)\
{\
  Timer::Phase tphase(timeStats(),TP_Communication);\
  field->scatterVariable(send_buf,recv_buf,root);\
}\
type ParallelMngDispatcher::\
reduce(eReduceType rt,type v)\
{\
  Timer::Phase tphase(timeStats(),TP_Communication);\
  return field->allReduce(rt,v);\
}\
void ParallelMngDispatcher::\
reduce(eReduceType rt,ArrayView<type> v)\
{\
  Timer::Phase tphase(timeStats(),TP_Communication);\
  field->allReduce(rt,v);\
}\
void ParallelMngDispatcher::\
broadcast(ArrayView<type> send_buf,Integer id)\
{ \
  Timer::Phase tphase(timeStats(),TP_Communication);\
  field->broadcast(send_buf,id);\
}\
void ParallelMngDispatcher::\
send(ConstArrayView<type> values,Integer id)\
{\
  Timer::Phase tphase(timeStats(),TP_Communication);\
  field->send(values,id);\
}\
void ParallelMngDispatcher::\
recv(ArrayView<type> values,Integer id)\
{\
  Timer::Phase tphase(timeStats(),TP_Communication);\
  field->recv(values,id);\
}\
Parallel::Request ParallelMngDispatcher::\
send(ConstArrayView<type> values,Integer id,bool is_blocked)\
{\
  Timer::Phase tphase(timeStats(),TP_Communication);\
  return field->send(values,id,is_blocked);\
}\
Request ParallelMngDispatcher::\
send(Span<const type> values,const PointToPointMessageInfo& message)  \
{\
  Timer::Phase tphase(timeStats(),TP_Communication);\
  return field->send(values,message);\
}\
Parallel::Request ParallelMngDispatcher::\
recv(ArrayView<type> values,Integer id,bool is_blocked)\
{\
  Timer::Phase tphase(timeStats(),TP_Communication);\
  return field->recv(values,id,is_blocked);\
}\
Request ParallelMngDispatcher::\
receive(Span<type> values,const PointToPointMessageInfo& message)  \
{\
  Timer::Phase tphase(timeStats(),TP_Communication);\
  return field->receive(values,message);\
}\
void ParallelMngDispatcher::\
sendRecv(ConstArrayView<type> send_buf,ArrayView<type> recv_buf,Integer id)\
{\
  Timer::Phase tphase(timeStats(),TP_Communication);\
  field->sendRecv(send_buf,recv_buf,id);\
}\
void ParallelMngDispatcher::\
allToAll(ConstArrayView<type> send_buf,ArrayView<type> recv_buf,Integer count)\
{\
  Timer::Phase tphase(timeStats(),TP_Communication);\
  field->allToAll(send_buf,recv_buf,count);\
}\
void ParallelMngDispatcher::\
allToAllVariable(ConstArrayView<type> send_buf,Int32ConstArrayView send_count,\
                 Int32ConstArrayView send_index,ArrayView<type> recv_buf,\
                 Int32ConstArrayView recv_count,Int32ConstArrayView recv_index)\
{\
  Timer::Phase tphase(timeStats(),TP_Communication);\
  field->allToAllVariable(send_buf,send_count,send_index,recv_buf,recv_count,recv_index);\
}\
type ParallelMngDispatcher::\
scan(eReduceType rt,type v)\
{\
  Timer::Phase tphase(timeStats(),TP_Communication);\
  return field->scan(rt,v);\
}\
void ParallelMngDispatcher::\
computeMinMaxSum(type val,type& min_val,type& max_val,type& sum_val,Int32& min_proc,Int32& max_proc)\
{\
  Timer::Phase tphase(timeStats(),TP_Communication);\
  field->computeMinMaxSum(val,min_val,max_val,sum_val,min_proc,max_proc);\
}\
IParallelDispatchT<type>* ParallelMngDispatcher::\
dispatcher(type*)\
{\
  return field;\
}\
void ParallelMngDispatcher::\
computeMinMaxSum(ConstArrayView<type> values,    \
                 ArrayView<type> min_values,     \
                 ArrayView<type> max_values,     \
                 ArrayView<type> sum_values,     \
                 ArrayView<Int32> min_ranks,     \
                 ArrayView<Int32> max_ranks)     \
{\
  Timer::Phase tphase(timeStats(),TP_Communication);\
  field->computeMinMaxSum(values,min_values,max_values,sum_values,min_ranks,max_ranks);\
}\
void ParallelMngDispatcher::\
scan(eReduceType rt,ArrayView<type> v)\
{\
  Timer::Phase tphase(timeStats(),TP_Communication);\
  field->scan(rt,v);\
}

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
ARCANE_PARALLEL_MANAGER_DISPATCH(m_apreal,APReal)
ARCANE_PARALLEL_MANAGER_DISPATCH(m_real2,Real2)
ARCANE_PARALLEL_MANAGER_DISPATCH(m_real3,Real3)
ARCANE_PARALLEL_MANAGER_DISPATCH(m_real2x2,Real2x2)
ARCANE_PARALLEL_MANAGER_DISPATCH(m_real3x3,Real3x3)
ARCANE_PARALLEL_MANAGER_DISPATCH(m_hpreal,HPReal)

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore
{
ARCCORE_DEFINE_REFERENCE_COUNTED_CLASS(Arcane::IParallelMng);
ARCCORE_DEFINE_REFERENCE_COUNTED_CLASS(Arcane::IParallelMngContainer);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
