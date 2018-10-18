/*---------------------------------------------------------------------------*/
/* Messages.h                                                  (C) 2000-2018 */
/*                                                                           */
/* Interface du gestionnaire des échanges de messages.                       */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_MESSAGEPASSING_MESSAGES_H
#define ARCCORE_MESSAGEPASSING_MESSAGES_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing/IMessagePassingMng.h"
#include "arccore/message_passing/IDispatchers.h"
#include "arccore/message_passing/ITypeDispatcher.h"
#include "arccore/message_passing/Request.h"
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
  inline void mpAllGather(IMessagePassingMng* pm,Span<const type> send_buf,Span<type> recv_buf) \
  { type* x = nullptr; pm->dispatchers()->dispatcher(x)->allGather(send_buf,recv_buf); } \
  inline void mpGather(IMessagePassingMng* pm,Span<const type> send_buf,Span<type> recv_buf,Int32 rank) \
  { type* x = nullptr; pm->dispatchers()->dispatcher(x)->gather(send_buf,recv_buf,rank); } \
  inline void mpAllGatherVariable(IMessagePassingMng* pm,Span<const type> send_buf,Array<type>& recv_buf) \
  { type* x = nullptr; pm->dispatchers()->dispatcher(x)->allGatherVariable(send_buf,recv_buf); } \
  inline void mpGatherVariable(IMessagePassingMng* pm,Span<const type> send_buf,Array<type>& recv_buf,Int32 rank)\
  { type* x = nullptr; pm->dispatchers()->dispatcher(x)->gatherVariable(send_buf,recv_buf,rank); } \
  inline void mpScatterVariable(IMessagePassingMng* pm,Span<const type> send_buf,Span<type> recv_buf,Int32 root); \
  inline type mpAllReduce(IMessagePassingMng* pm,eReduceType rt,type v) \
  { type* x = nullptr; return pm->dispatchers()->dispatcher(x)->allReduce(rt,v); }\
  inline void mpAllReduce(IMessagePassingMng* pm,eReduceType rt,Span<type> v) \
  { type* x = nullptr; pm->dispatchers()->dispatcher(x)->allReduce(rt,v); }\
  inline void mpBroadcast(IMessagePassingMng* pm,Span<type> send_buf,Int32 rank) \
  { type* x = nullptr; pm->dispatchers()->dispatcher(x)->broadcast(send_buf,rank); } \
  inline void mpSend(IMessagePassingMng* pm,Span<const type> values,Int32 rank) \
  { type* x = nullptr; pm->dispatchers()->dispatcher(x)->send(values,rank,true); } \
  inline void mpReceive(IMessagePassingMng* pm,Span<type> values,Int32 rank) \
  { type* x = nullptr; pm->dispatchers()->dispatcher(x)->receive(values,rank,true); } \
  inline Request mpSend(IMessagePassingMng* pm,Span<const type> values,Int32 rank,bool is_blocked) \
  { type* x = nullptr; return pm->dispatchers()->dispatcher(x)->send(values,rank,is_blocked); } \
  inline Request mpReceive(IMessagePassingMng* pm,Span<type> values,Int32 rank,bool is_blocked)\
  { type* x = nullptr; return pm->dispatchers()->dispatcher(x)->receive(values,rank,is_blocked); } \
  inline void mpAllToAll(IMessagePassingMng* pm,Span<const type> send_buf,Span<type> recv_buf,Int32 count) \
  { type* x = nullptr; return pm->dispatchers()->dispatcher(x)->allToAll(send_buf,recv_buf,count); } \
  inline void mpAllToAllVariable(IMessagePassingMng* pm,Span<const type> send_buf,ConstArrayView<Int32> send_count, \
                                 ConstArrayView<Int32> send_index,Span<type> recv_buf, \
                                 ConstArrayView<Int32> recv_count,ConstArrayView<Int32> recv_index) \
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
