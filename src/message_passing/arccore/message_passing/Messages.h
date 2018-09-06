/*---------------------------------------------------------------------------*/
/* Messages.h                                                  (C) 2000-2018 */
/*                                                                           */
/* Interface du gestionnaire des échanges de messages.                       */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_MESSAGEPASSING_MESSAGES_H
#define ARCCORE_MESSAGEPASSING_MESSAGES_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/messagepassing/IMessagePassingMng.h"
#include "arccore/messagepassing/IDispatchers.h"
#include "arccore/messagepassing/ITypeDispatcher.h"
#include "arccore/messagepassing/Request.h"
#include "arccore/base/ArrayView.h"
#include "arccore/collections/Array.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore
{
namespace MessagePassing
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#define ARCCORE_GENERATE_MESSAGEPASSING_PROTOTYPE(type) \
  inline void mpAllGather(IMessagePassingMng* pm,ConstArrayView<type> send_buf,ArrayView<type> recv_buf) \
  { type* x = nullptr; pm->dispatchers()->dispatcher(x)->allGather(send_buf,recv_buf); } \
  inline void mpGather(IMessagePassingMng* pm,ConstArrayView<type> send_buf,ArrayView<type> recv_buf,Int32 rank) \
  { type* x = nullptr; pm->dispatchers()->dispatcher(x)->gather(send_buf,recv_buf,rank); } \
  inline void mpAllGatherVariable(IMessagePassingMng* pm,ConstArrayView<type> send_buf,Array<type>& recv_buf) \
  { type* x = nullptr; pm->dispatchers()->dispatcher(x)->allGatherVariable(send_buf,recv_buf); } \
  inline void mpGatherVariable(IMessagePassingMng* pm,ConstArrayView<type> send_buf,Array<type>& recv_buf,Int32 rank)\
  { type* x = nullptr; pm->dispatchers()->dispatcher(x)->gatherVariable(send_buf,recv_buf,rank); } \
  inline void mpScatterVariable(IMessagePassingMng* pm,ConstArrayView<type> send_buf,ArrayView<type> recv_buf,Int32 root) ; \
  inline type mpAllReduce(IMessagePassingMng* pm,eReduceType rt,type v) \
  { type* x = nullptr; return pm->dispatchers()->dispatcher(x)->allReduce(rt,v); }\
  inline void mpAllReduce(IMessagePassingMng* pm,eReduceType rt,ArrayView<type> v) \
  { type* x = nullptr; pm->dispatchers()->dispatcher(x)->allReduce(rt,v); }\
  inline void mpBroadcast(IMessagePassingMng* pm,ArrayView<type> send_buf,Int32 rank) \
  { type* x = nullptr; pm->dispatchers()->dispatcher(x)->broadcast(send_buf,rank); } \
  inline void mpSend(IMessagePassingMng* pm,ConstArrayView<type> values,Int32 rank) \
  { type* x = nullptr; pm->dispatchers()->dispatcher(x)->send(values,rank,true); } \
  inline void mpReceive(IMessagePassingMng* pm,ArrayView<type> values,Int32 rank) \
  { type* x = nullptr; pm->dispatchers()->dispatcher(x)->receive(values,rank,true); } \
  inline Request mpSend(IMessagePassingMng* pm,ConstArrayView<type> values,Int32 rank,bool is_blocked) \
  { type* x = nullptr; return pm->dispatchers()->dispatcher(x)->send(values,rank,is_blocked); } \
  inline Request mpReceive(IMessagePassingMng* pm,ArrayView<type> values,Int32 rank,bool is_blocked)\
  { type* x = nullptr; return pm->dispatchers()->dispatcher(x)->receive(values,rank,is_blocked); } \
  inline void mpAllToAll(IMessagePassingMng* pm,ConstArrayView<type> send_buf,ArrayView<type> recv_buf,Integer count) \
  { type* x = nullptr; return pm->dispatchers()->dispatcher(x)->allToAll(send_buf,recv_buf,count); } \
  inline void mpAllToAllVariable(IMessagePassingMng* pm,ConstArrayView<type> send_buf,Int32ConstArrayView send_count, \
                                 Int32ConstArrayView send_index,ArrayView<type> recv_buf, \
                                 Int32ConstArrayView recv_count,Int32ConstArrayView recv_index)\
  { type* x = nullptr;\
    auto d = pm->dispatchers()->dispatcher(x);\
    d->allToAllVariable(send_buf,send_count,send_index,recv_buf,recv_count,recv_index); }

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCCORE_GENERATE_MESSAGEPASSING_PROTOTYPE(char)
ARCCORE_GENERATE_MESSAGEPASSING_PROTOTYPE(signed char)
ARCCORE_GENERATE_MESSAGEPASSING_PROTOTYPE(unsigned char)

ARCCORE_GENERATE_MESSAGEPASSING_PROTOTYPE(short)
ARCCORE_GENERATE_MESSAGEPASSING_PROTOTYPE(unsigned short)
ARCCORE_GENERATE_MESSAGEPASSING_PROTOTYPE(int)
ARCCORE_GENERATE_MESSAGEPASSING_PROTOTYPE(unsigned int)
ARCCORE_GENERATE_MESSAGEPASSING_PROTOTYPE(long)
ARCCORE_GENERATE_MESSAGEPASSING_PROTOTYPE(unsigned long)
ARCCORE_GENERATE_MESSAGEPASSING_PROTOTYPE(long long)
ARCCORE_GENERATE_MESSAGEPASSING_PROTOTYPE(unsigned long long)

ARCCORE_GENERATE_MESSAGEPASSING_PROTOTYPE(float)
ARCCORE_GENERATE_MESSAGEPASSING_PROTOTYPE(double)
ARCCORE_GENERATE_MESSAGEPASSING_PROTOTYPE(long double)

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace MessagePassing
} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
