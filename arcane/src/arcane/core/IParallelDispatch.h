// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IParallelDispatch.h                                         (C) 2000-2025 */
/*                                                                           */
/* Gestion des messages parallèles pour un type de valeur.                   */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IPARALLELDISPATCH_H
#define ARCANE_CORE_IPARALLELDISPATCH_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"
#include "arcane/core/Parallel.h"
#include "arccore/message_passing/ITypeDispatcher.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Gestion des messages parallèles pour le type \a Type.
 */
template <class Type>
class IParallelDispatchT
{
 public:

  using Request = Parallel::Request;
  using eReduceType = Parallel::eReduceType;
  using PointToPointMessageInfo = Parallel::PointToPointMessageInfo;

 public:

  virtual ~IParallelDispatchT() = default;

 public:

  virtual void finalize() = 0;

 public:

  virtual void broadcast(ArrayView<Type> send_buf, Int32 rank) = 0;
  virtual void allGather(ConstArrayView<Type> send_buf, ArrayView<Type> recv_buf) = 0;
  virtual void allGatherVariable(ConstArrayView<Type> send_buf, Array<Type>& recv_buf) = 0;
  virtual void gather(ConstArrayView<Type> send_buf, ArrayView<Type> recv_buf, Int32 rank) = 0;
  virtual void gatherVariable(ConstArrayView<Type> send_buf, Array<Type>& recv_buf, Int32 rank) = 0;
  virtual void scatterVariable(ConstArrayView<Type> send_buf, ArrayView<Type> recv_buf, Int32 root) = 0;
  virtual void allToAll(ConstArrayView<Type> send_buf, ArrayView<Type> recv_buf, Integer count) = 0;
  virtual void allToAllVariable(ConstArrayView<Type> send_buf, Int32ConstArrayView send_count,
                                Int32ConstArrayView send_index, ArrayView<Type> recv_buf,
                                Int32ConstArrayView recv_count, Int32ConstArrayView recv_index) = 0;
  virtual Type allReduce(eReduceType op, Type send_buf) = 0;
  virtual void allReduce(eReduceType op, ArrayView<Type> send_buf) = 0;

 public:

  virtual Request send(ConstArrayView<Type> send_buffer, Int32 rank, bool is_blocked) = 0;
  virtual Request send(Span<const Type> recv_buffer, const PointToPointMessageInfo& message) = 0;
  virtual Request recv(ArrayView<Type> recv_buffer, Int32 rank, bool is_blocked) = 0;
  virtual Request receive(Span<Type> recv_buffer, const PointToPointMessageInfo& message) = 0;
  virtual void send(ConstArrayView<Type> send_buffer, Int32 rank) = 0;
  virtual void recv(ArrayView<Type> recv_buffer, Int32 rank) = 0;
  virtual void sendRecv(ConstArrayView<Type> send_buffer, ArrayView<Type> recv_buffer, Int32 proc) = 0;
  virtual Type scan(eReduceType op, Type send_buf) = 0;
  virtual void scan(eReduceType op, ArrayView<Type> send_buf) = 0;
  virtual void computeMinMaxSum(Type val, Type& min_val, Type& max_val, Type& sum_val,
                                Int32& min_rank,
                                Int32& max_rank) = 0;
  virtual void computeMinMaxSum(ConstArrayView<Type> values,
                                ArrayView<Type> min_values,
                                ArrayView<Type> max_values,
                                ArrayView<Type> sum_values,
                                ArrayView<Int32> min_ranks,
                                ArrayView<Int32> max_ranks) = 0;

 public:

  virtual ITypeDispatcher<Type>* toArccoreDispatcher() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

