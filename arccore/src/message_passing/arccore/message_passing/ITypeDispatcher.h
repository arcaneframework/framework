// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ITypeDispatcher.h                                           (C) 2000-2025 */
/*                                                                           */
/* Gestion des messages pour un type de données.                             */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_MESSAGEPASSING_ITYPEDISPATCHER_H
#define ARCCORE_MESSAGEPASSING_ITYPEDISPATCHER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing/MessagePassingGlobal.h"
#include "arccore/message_passing/Request.h"
#include "arccore/collections/CollectionsGlobal.h"
#include "arccore/base/BaseTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MessagePassing
{
extern "C++" ARCCORE_MESSAGEPASSING_EXPORT void
_internalThrowNotImplementedTypeDispatcher ARCCORE_NORETURN ();

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Gestion des messages parallèles pour le type \a Type.
 */
template <class Type>
class ITypeDispatcher
{
 public:

  virtual ~ITypeDispatcher() = default;
  virtual void finalize() = 0;

 public:

  virtual void broadcast(Span<Type> send_buf, Int32 rank) = 0;
  virtual void allGather(Span<const Type> send_buf, Span<Type> recv_buf) = 0;
  virtual void allGatherVariable(Span<const Type> send_buf, Array<Type>& recv_buf) = 0;
  virtual void gather(Span<const Type> send_buf, Span<Type> recv_buf, Int32 rank) = 0;
  virtual void gatherVariable(Span<const Type> send_buf, Array<Type>& recv_buf, Int32 rank) = 0;
  virtual void scatterVariable(Span<const Type> send_buf, Span<Type> recv_buf, Int32 root) = 0;
  virtual void allToAll(Span<const Type> send_buf, Span<Type> recv_buf, Int32 count) = 0;
  virtual void allToAllVariable(Span<const Type> send_buf, ConstArrayView<Int32> send_count,
                                ConstArrayView<Int32> send_index, Span<Type> recv_buf,
                                ConstArrayView<Int32> recv_count, ConstArrayView<Int32> recv_index) = 0;
  virtual Request send(Span<const Type> send_buffer, Int32 rank, bool is_blocked) = 0;
  virtual Request send(Span<const Type> send_buffer, const PointToPointMessageInfo& message) = 0;
  virtual Request receive(Span<Type> recv_buffer, Int32 rank, bool is_blocked) = 0;
  virtual Request receive(Span<Type> recv_buffer, const PointToPointMessageInfo& message) = 0;
  virtual Type allReduce(eReduceType op, Type send_buf) = 0;
  virtual void allReduce(eReduceType op, Span<Type> send_buf) = 0;
  virtual Request nonBlockingAllReduce(eReduceType op, Span<const Type> send_buf, Span<Type> recv_buf) = 0;
  virtual Request nonBlockingAllGather(Span<const Type> send_buf, Span<Type> recv_buf) = 0;
  virtual Request nonBlockingBroadcast(Span<Type> send_buf, Int32 rank) = 0;
  virtual Request nonBlockingGather(Span<const Type> send_buf, Span<Type> recv_buf, Int32 rank) = 0;
  virtual Request nonBlockingAllToAll(Span<const Type> send_buf, Span<Type> recv_buf, Int32 count) = 0;
  virtual Request nonBlockingAllToAllVariable(Span<const Type> send_buf, ConstArrayView<Int32> send_count,
                                              ConstArrayView<Int32> send_index, Span<Type> recv_buf,
                                              ConstArrayView<Int32> recv_count, ConstArrayView<Int32> recv_index) = 0;
  virtual Request gather(GatherMessageInfo<Type>&)
  {
    _internalThrowNotImplementedTypeDispatcher();
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore::MessagePassing

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
