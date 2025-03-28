// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MpiParallelNonBlockingCollectiveDispatch.cc                 (C) 2000-2025 */
/*                                                                           */
/* Implémentation MPI des collectives non bloquantes pour un type donné.     */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Array.h"
#include "arcane/utils/NotImplementedException.h"
#include "arcane/utils/Real2.h"
#include "arcane/utils/Real3.h"
#include "arcane/utils/Real2x2.h"
#include "arcane/utils/Real3x3.h"
#include "arcane/utils/HPReal.h"
#include "arcane/utils/FatalErrorException.h"

#include "arcane/core/IParallelNonBlockingCollective.h"
#include "arcane/core/ParallelMngDispatcher.h"

#include "arcane/parallel/mpi/MpiParallelNonBlockingCollectiveDispatch.h"
#include "arcane/parallel/mpi/MpiDatatype.h"
#include "arcane/parallel/mpi/MpiParallelDispatch.h"

#include "arccore/message_passing_mpi/internal/MpiAdapter.h"
#include "arccore/message_passing_mpi/internal/MpiLock.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Type> MpiParallelNonBlockingCollectiveDispatchT<Type>::
MpiParallelNonBlockingCollectiveDispatchT(ITraceMng* tm, IParallelNonBlockingCollective* collective_mng,
                                          MpiAdapter* adapter)
: TraceAccessor(tm)
, m_parallel_mng(collective_mng->parallelMng())
, m_adapter(adapter)
, m_datatype(nullptr)
{
  // Récupérer le datatype via le dispatcher MpiParallelDispatch
  // TODO: créer un type pour contenir tous les MpiDatatype.
  auto pmd = dynamic_cast<ParallelMngDispatcher*>(m_parallel_mng);
  if (!pmd)
    ARCANE_FATAL("Bad parallelMng()");
  Type* xtype = nullptr;
  auto dispatcher = pmd->dispatcher(xtype);
  auto true_dispatcher = dynamic_cast<MpiParallelDispatchT<Type>*>(dispatcher);
  if (!true_dispatcher)
    ARCANE_FATAL("Bad dispatcher. should have type MpiParallelDispatcher");
  m_datatype = true_dispatcher->datatype();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Type> MpiParallelNonBlockingCollectiveDispatchT<Type>::
~MpiParallelNonBlockingCollectiveDispatchT()
{
  // NOTE: m_datatype est géré par MpiParallelDispatch et ne doit pas être
  // détruit ici.
  finalize();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Type> void MpiParallelNonBlockingCollectiveDispatchT<Type>::
finalize()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Type> Parallel::Request MpiParallelNonBlockingCollectiveDispatchT<Type>::
broadcast(ArrayView<Type> send_buf, Integer sub_domain)
{
  MPI_Datatype type = m_datatype->datatype();
  return m_adapter->nonBlockingBroadcast(send_buf.data(), send_buf.size(), sub_domain, type);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Type> Parallel::Request MpiParallelNonBlockingCollectiveDispatchT<Type>::
allGather(ConstArrayView<Type> send_buf, ArrayView<Type> recv_buf)
{
  MPI_Datatype type = m_datatype->datatype();
  return m_adapter->nonBlockingAllGather(send_buf.data(), recv_buf.data(), send_buf.size(), type);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Type> Parallel::Request MpiParallelNonBlockingCollectiveDispatchT<Type>::
gather(ConstArrayView<Type> send_buf, ArrayView<Type> recv_buf, Integer rank)
{
  MPI_Datatype type = m_datatype->datatype();
  return m_adapter->nonBlockingGather(send_buf.data(), recv_buf.data(), send_buf.size(), rank, type);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Type> Parallel::Request MpiParallelNonBlockingCollectiveDispatchT<Type>::
allGatherVariable(ConstArrayView<Type> send_buf, Array<Type>& recv_buf)
{
  ARCANE_UNUSED(send_buf);
  ARCANE_UNUSED(recv_buf);
  throw NotImplementedException(A_FUNCINFO);
#if 0
  _gatherVariable2(send_buf,recv_buf,-1);
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Type> Parallel::Request MpiParallelNonBlockingCollectiveDispatchT<Type>::
gatherVariable(ConstArrayView<Type> send_buf, Array<Type>& recv_buf, Integer rank)
{
  ARCANE_UNUSED(send_buf);
  ARCANE_UNUSED(recv_buf);
  ARCANE_UNUSED(rank);
  throw NotImplementedException(A_FUNCINFO);
#if 0
  _gatherVariable2(send_buf,recv_buf,rank);
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Type> Parallel::Request MpiParallelNonBlockingCollectiveDispatchT<Type>::
scatterVariable(ConstArrayView<Type> send_buf, ArrayView<Type> recv_buf, Integer root)
{
  ARCANE_UNUSED(send_buf);
  ARCANE_UNUSED(recv_buf);
  ARCANE_UNUSED(root);
  throw NotImplementedException(A_FUNCINFO);
#if 0
  MPI_Datatype type = m_adapter->datatype(Type());

  Integer comm_size = static_cast<Integer>(m_adapter->commSize());
  UniqueArray<int> recv_counts(comm_size);
  UniqueArray<int> recv_indexes(comm_size);

  Integer nb_elem = recv_buf.size();
  int my_buf_count = static_cast<int>(nb_elem);
  ConstArrayView<int> count_r(1,&my_buf_count);

  // Récupère le nombre d'éléments de chaque processeur
  m_parallel_mng->allGather(count_r,recv_counts);

  // Remplit le tableau des index
  int index = 0;
  for( Integer i=0, is=comm_size; i<is; ++i ){
    recv_indexes[i] = index;
    index += recv_counts[i];
  }

  m_adapter->scatterVariable(send_buf.begin(),recv_counts.begin(),recv_indexes.begin(),
                             recv_buf.begin(),nb_elem,root,type);
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Type> Parallel::Request MpiParallelNonBlockingCollectiveDispatchT<Type>::
allToAll(ConstArrayView<Type> send_buf, ArrayView<Type> recv_buf, Integer count)
{
  MPI_Datatype type = m_datatype->datatype();
  return m_adapter->nonBlockingAllToAll(send_buf.data(), recv_buf.data(), count, type);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Type> Parallel::Request MpiParallelNonBlockingCollectiveDispatchT<Type>::
allToAllVariable(ConstArrayView<Type> send_buf,
                 Int32ConstArrayView send_count,
                 Int32ConstArrayView send_index,
                 ArrayView<Type> recv_buf,
                 Int32ConstArrayView recv_count,
                 Int32ConstArrayView recv_index)
{
  MPI_Datatype type = m_datatype->datatype();

  return m_adapter->nonBlockingAllToAllVariable(send_buf.data(), send_count.data(),
                                                send_index.data(), recv_buf.data(),
                                                recv_count.data(),
                                                recv_index.data(), type);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <class Type> Parallel::Request MpiParallelNonBlockingCollectiveDispatchT<Type>::
allReduce(eReduceType op, ConstArrayView<Type> send_buf, ArrayView<Type> recv_buf)
{
  MPI_Datatype type = m_datatype->datatype();
  Integer s = send_buf.size();
  MPI_Op operation = m_datatype->reduceOperator(op);

  Request request;
  {
    MpiLock::Section mls(m_adapter->mpiLock());
    request = m_adapter->nonBlockingAllReduce(send_buf.data(), recv_buf.data(),
                                              s, type, operation);
  }
  return request;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template class MpiParallelNonBlockingCollectiveDispatchT<char>;
template class MpiParallelNonBlockingCollectiveDispatchT<signed char>;
template class MpiParallelNonBlockingCollectiveDispatchT<unsigned char>;
template class MpiParallelNonBlockingCollectiveDispatchT<short>;
template class MpiParallelNonBlockingCollectiveDispatchT<unsigned short>;
template class MpiParallelNonBlockingCollectiveDispatchT<int>;
template class MpiParallelNonBlockingCollectiveDispatchT<unsigned int>;
template class MpiParallelNonBlockingCollectiveDispatchT<long>;
template class MpiParallelNonBlockingCollectiveDispatchT<unsigned long>;
template class MpiParallelNonBlockingCollectiveDispatchT<long long>;
template class MpiParallelNonBlockingCollectiveDispatchT<unsigned long long>;
template class MpiParallelNonBlockingCollectiveDispatchT<float>;
template class MpiParallelNonBlockingCollectiveDispatchT<double>;
template class MpiParallelNonBlockingCollectiveDispatchT<long double>;
template class MpiParallelNonBlockingCollectiveDispatchT<Real2>;
template class MpiParallelNonBlockingCollectiveDispatchT<Real3>;
template class MpiParallelNonBlockingCollectiveDispatchT<Real2x2>;
template class MpiParallelNonBlockingCollectiveDispatchT<Real3x3>;
template class MpiParallelNonBlockingCollectiveDispatchT<HPReal>;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
