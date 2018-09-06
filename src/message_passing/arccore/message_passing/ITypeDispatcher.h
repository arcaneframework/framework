/*---------------------------------------------------------------------------*/
/* ITypeDispatcher.h                                           (C) 2000-2018 */
/*                                                                           */
/* Gestion des messages pour un type de données.                             */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_MESSAGEPASSING_ITYPEDISPATCHER_H
#define ARCCORE_MESSAGEPASSING_ITYPEDISPATCHER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/messagepassing/MessagePassingGlobal.h"
#include "arccore/collections/CollectionsGlobal.h"
#include "arccore/base/BaseTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore
{
namespace MessagePassing
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Gestion des messages parallèles pour le type \a Type.
 */
template<class Type>
class ITypeDispatcher
{
 public:
  virtual ~ITypeDispatcher() {}
  virtual void finalize() =0;
 public:
  virtual void broadcast(ArrayView<Type> send_buf,Int32 rank) =0;
  virtual void allGather(ConstArrayView<Type> send_buf,ArrayView<Type> recv_buf) =0;
  virtual void allGatherVariable(ConstArrayView<Type> send_buf,Array<Type>& recv_buf) =0;
  virtual void gather(ConstArrayView<Type> send_buf,ArrayView<Type> recv_buf,Int32 rank) =0;
  virtual void gatherVariable(ConstArrayView<Type> send_buf,Array<Type>& recv_buf,Int32 rank) =0;
  virtual void scatterVariable(ConstArrayView<Type> send_buf,ArrayView<Type> recv_buf,Int32 root) =0;
  virtual void allToAll(ConstArrayView<Type> send_buf,ArrayView<Type> recv_buf,Integer count) =0;
  virtual void allToAllVariable(ConstArrayView<Type> send_buf,Int32ConstArrayView send_count,
                                Int32ConstArrayView send_index,ArrayView<Type> recv_buf,
                                Int32ConstArrayView recv_count,Int32ConstArrayView recv_index) =0;
  virtual Request send(ConstArrayView<Type> send_buffer,Int32 rank,bool is_blocked) =0;
  virtual Request receive(ArrayView<Type> recv_buffer,Int32 rank,bool is_blocked) =0;
  virtual Type allReduce(eReduceType op,Type send_buf) =0;
  virtual void allReduce(eReduceType op,ArrayView<Type> send_buf) =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace MessagePassing
} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

