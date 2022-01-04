// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MpiParallelNonBlockingCollectiveDispatch.h                  (C) 2000-2018 */
/*                                                                           */
/* Implémentation MPI des collectives non bloquantes pour un type donné.     */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_PARALLEL_MPI_MPIPARALLELNONBLOCKINGCOLLECTIVEDISPATCH_H
#define ARCANE_PARALLEL_MPI_MPIPARALLELNONBLOCKINGCOLLECTIVEDISPATCH_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"

#include "arcane/IParallelNonBlockingCollectiveDispatch.h"

#include "arcane/parallel/mpi/ArcaneMpi.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class MpiParallelMng;
class IParallelNonBlockingCollective;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Implémentation MPI des collectives non bloquantes pour le type \a Type.
 */
template<class Type>
class MpiParallelNonBlockingCollectiveDispatchT
: public TraceAccessor
, public IParallelNonBlockingCollectiveDispatchT<Type>
{
 public:

  typedef Parallel::Request Request;
  typedef Parallel::eReduceType eReduceType;

 public:

  ARCANE_MPI_EXPORT MpiParallelNonBlockingCollectiveDispatchT(ITraceMng* tm, IParallelNonBlockingCollective* parallel_mng, MpiAdapter* adapter);
  virtual ARCANE_MPI_EXPORT ~MpiParallelNonBlockingCollectiveDispatchT();
  virtual ARCANE_MPI_EXPORT void finalize();

 public:

  virtual ARCANE_MPI_EXPORT Request broadcast(ArrayView<Type> send_buf,Integer sub_domain);
  virtual ARCANE_MPI_EXPORT Request allGather(ConstArrayView<Type> send_buf, ArrayView<Type> recv_buf);
  virtual ARCANE_MPI_EXPORT Request allGatherVariable(ConstArrayView<Type> send_buf, Array<Type>& recv_buf);
  virtual ARCANE_MPI_EXPORT Request gather(ConstArrayView<Type> send_buf, ArrayView<Type> recv_buf, Integer rank);
  virtual ARCANE_MPI_EXPORT Request gatherVariable(ConstArrayView<Type> send_buf, Array<Type>& recv_buf, Integer rank);
  virtual ARCANE_MPI_EXPORT Request scatterVariable(ConstArrayView<Type> send_buf, ArrayView<Type> recv_buf, Integer root);
  virtual ARCANE_MPI_EXPORT Request allToAll(ConstArrayView<Type> send_buf, ArrayView<Type> recv_buf, Integer count);
  virtual ARCANE_MPI_EXPORT Request allToAllVariable(ConstArrayView<Type> send_buf, Int32ConstArrayView send_count,
                                                     Int32ConstArrayView send_index,ArrayView<Type> recv_buf,
                                                     Int32ConstArrayView recv_count,Int32ConstArrayView recv_index);
  virtual ARCANE_MPI_EXPORT Request allReduce(eReduceType op, ConstArrayView<Type> send_buf, ArrayView<Type> recv_buf);


 private:

  IParallelMng* m_parallel_mng;
  MpiAdapter* m_adapter;
  MpiDatatype* m_datatype;

 private:
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

