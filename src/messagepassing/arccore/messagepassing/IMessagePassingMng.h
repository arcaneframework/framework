/*---------------------------------------------------------------------------*/
/* IMessagePassingMng.h                                        (C) 2000-2018 */
/*                                                                           */
/* Interface du gestionnaire des échanges de messages.                       */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_MESSAGEPASSING_IMESSAGEPASSINGMNG_H
#define ARCCORE_MESSAGEPASSING_IMESSAGEPASSINGMNG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/messagepassing/MessagePassingGlobal.h"
#include "arccore/base/BaseTypes.h"
#include "arccore/collections/CollectionsGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore
{
namespace MessagePassing
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface du gestionnaire des échanges de messages.
 */
class ARCCORE_MESSAGEPASSING_EXPORT IMessagePassingMng
{
 public:

  virtual ~IMessagePassingMng(){}

 public:

  virtual IDispatchers* dispatchers() =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#define ARCCORE_GENERATE_MESSAGEPASSING_PROTOTYPE(type) \
template<typename Type> void mpAllGather(IMessagePassingMng* pm,ConstArrayView<type> send_buf,ArrayView<type> recv_buf); \
template<typename Type> void mpGather(IMessagePassingMng* pm,ConstArrayView<type> send_buf,ArrayView<type> recv_buf,Int32 rank) ; \
template<typename Type> void mpAllGatherVariable(IMessagePassingMng* pm,ConstArrayView<type> send_buf,Array<type>& recv_buf) ; \
template<typename Type> void mpGatherVariable(IMessagePassingMng* pm,ConstArrayView<type> send_buf,Array<type>& recv_buf,Int32 rank) ; \
template<typename Type> void mpScatterVariable(IMessagePassingMng* pm,ConstArrayView<type> send_buf,ArrayView<type> recv_buf,Int32 root) ; \
template<typename Type> type mpAllReduce(IMessagePassingMng* pm,eReduceType rt,type v) ; \
template<typename Type> void mpAllReduce(IMessagePassingMng* pm,eReduceType rt,ArrayView<type> v) ;\
template<typename Type> void mpBroadcast(IMessagePassingMng* pm,ArrayView<type> send_buf,Int32 rank) ;\
template<typename Type> void mpSend(IMessagePassingMng* pm,ConstArrayView<type> values,Int32 rank) ;\
template<typename Type> void mpReceive(IMessagePassingMng* pm,ArrayView<type> values,Int32 rank) ; \
template<typename Type> Request mpSend(IMessagePassingMng* pm,ConstArrayView<type> values,Int32 rank,bool is_blocked) ;\
template<typename Type> Request mpReceive(IMessagePassingMng* pm,ArrayView<type> values,Int32 rank,bool is_blocked) ; \
template<typename Type> void mpAllToAll(IMessagePassingMng* pm,ConstArrayView<type> send_buf,ArrayView<type> recv_buf,Integer count) ;\
template<typename Type> void mpAllToAllVariable(IMessagePassingMng* pm,ConstArrayView<type> send_buf,Int32ConstArrayView send_count,\
                                                Int32ConstArrayView send_index,ArrayView<type> recv_buf, \
                                                Int32ConstArrayView recv_count,Int32ConstArrayView recv_index)

ARCCORE_GENERATE_MESSAGEPASSING_PROTOTYPE(char);
ARCCORE_GENERATE_MESSAGEPASSING_PROTOTYPE(signed char);
ARCCORE_GENERATE_MESSAGEPASSING_PROTOTYPE(unsigned char);

ARCCORE_GENERATE_MESSAGEPASSING_PROTOTYPE(short);
ARCCORE_GENERATE_MESSAGEPASSING_PROTOTYPE(unsigned short);
ARCCORE_GENERATE_MESSAGEPASSING_PROTOTYPE(int);
ARCCORE_GENERATE_MESSAGEPASSING_PROTOTYPE(unsigned int);
ARCCORE_GENERATE_MESSAGEPASSING_PROTOTYPE(long);
ARCCORE_GENERATE_MESSAGEPASSING_PROTOTYPE(unsigned long);
ARCCORE_GENERATE_MESSAGEPASSING_PROTOTYPE(long long);
ARCCORE_GENERATE_MESSAGEPASSING_PROTOTYPE(unsigned long long);

ARCCORE_GENERATE_MESSAGEPASSING_PROTOTYPE(float);
ARCCORE_GENERATE_MESSAGEPASSING_PROTOTYPE(double);
ARCCORE_GENERATE_MESSAGEPASSING_PROTOTYPE(long double);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace MessagePassing
} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
