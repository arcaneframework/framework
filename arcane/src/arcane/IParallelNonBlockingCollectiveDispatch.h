// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IParallelNonBlockingCollectiveDispatch.h                    (C) 2000-2015 */
/*                                                                           */
/* Interface des collectives non blocantes pour un type donné.               */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IPARALLELNONBLOCKINGCOLLECTIVEDISPATCH_H
#define ARCANE_IPARALLELNONBLOCKINGCOLLECTIVEDISPATCH_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/ArcaneTypes.h"
#include "arcane/Parallel.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Interface des collectives non blocantes pour le type \a Type.
 */
template<class Type>
class IParallelNonBlockingCollectiveDispatchT
{
 public:
  typedef Parallel::Request Request;
  typedef Parallel::eReduceType eReduceType;
 public:
  virtual ~IParallelNonBlockingCollectiveDispatchT() {}
  virtual void finalize() =0;
 public:
  virtual Request broadcast(ArrayView<Type> send_buf,Integer sub_domain) =0;
  virtual Request allGather(ConstArrayView<Type> send_buf,ArrayView<Type> recv_buf) =0;
  virtual Request allGatherVariable(ConstArrayView<Type> send_buf,Array<Type>& recv_buf) =0;
  virtual Request gather(ConstArrayView<Type> send_buf,ArrayView<Type> recv_buf,Integer rank) =0;
  virtual Request gatherVariable(ConstArrayView<Type> send_buf,Array<Type>& recv_buf,Integer rank) =0;
  virtual Request scatterVariable(ConstArrayView<Type> send_buf,ArrayView<Type> recv_buf,Integer root) =0;
  virtual Request allToAll(ConstArrayView<Type> send_buf,ArrayView<Type> recv_buf,Integer count) =0;
  virtual Request allToAllVariable(ConstArrayView<Type> send_buf,Int32ConstArrayView send_count,
                                   Int32ConstArrayView send_index,ArrayView<Type> recv_buf,
                                   Int32ConstArrayView recv_count,Int32ConstArrayView recv_index) =0;
  virtual Request allReduce(eReduceType op,ConstArrayView<Type> send_buf,ArrayView<Type> recv_buf) =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

