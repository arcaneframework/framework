// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MpiParallelDispatch.h                                       (C) 2000-2024 */
/*                                                                           */
/* Implémentation des messages avec MPI.                                     */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_PARALLEL_MPI_MPIPARALLELDISPATCH_H
#define ARCANE_PARALLEL_MPI_MPIPARALLELDISPATCH_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"

#include "arcane/core/IParallelDispatch.h"

#include "arcane/parallel/mpi/ArcaneMpi.h"
#include "arcane/parallel/mpi/MpiDatatypeList.h"

#include "arccore/message_passing_mpi/internal/MpiTypeDispatcher.h"
#include "arccore/message_passing/PointToPointMessageInfo.h"
#include "arccore/base/ReferenceCounterImpl.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class MpiParallelMng;
namespace MP = ::Arccore::MessagePassing;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface des messages pour le type \a Type
 */
template<class Type>
class MpiParallelDispatchT
: public TraceAccessor
, public ReferenceCounterImpl
, public IParallelDispatchT<Type>
{
  ARCCORE_INTERNAL_DEFINE_REFERENCE_COUNTED_INCLASS_METHODS();
 private:
  class ARCANE_MPI_EXPORT MinMaxSumInfo
  {
   public:
    Integer m_min_rank;
    Integer m_max_rank;
    Type m_min_value;
    Type m_max_value;
    Type m_sum_value;
  };
  
 public:
  typedef Parallel::Request Request;
  typedef Parallel::eReduceType eReduceType;
  using PointToPointMessageInfo = Parallel::PointToPointMessageInfo;
 public:
  ARCANE_MPI_EXPORT MpiParallelDispatchT(ITraceMng* tm, IMessagePassingMng* parallel_mng, MpiAdapter* adapter,MpiDatatype* datatype);
 public:
  ARCANE_MPI_EXPORT ~MpiParallelDispatchT() override;
  ARCANE_MPI_EXPORT void finalize() override;
 public:
  void broadcast(ArrayView<Type> send_buf,Int32 rank) override
  { m_mp_dispatcher->broadcast(send_buf,rank); }
  void allGather(ConstArrayView<Type> send_buf, ArrayView<Type> recv_buf) override
  { m_mp_dispatcher->allGather(send_buf,recv_buf); }
  void allGatherVariable(ConstArrayView<Type> send_buf, Array<Type>& recv_buf) override
  { m_mp_dispatcher->allGatherVariable(send_buf,recv_buf); }
  void gather(ConstArrayView<Type> send_buf, ArrayView<Type> recv_buf, Int32 rank) override
  { m_mp_dispatcher->gather(send_buf,recv_buf,rank); }
  void gatherVariable(ConstArrayView<Type> send_buf, Array<Type>& recv_buf, Int32 rank) override
  { m_mp_dispatcher->gatherVariable(send_buf,recv_buf,rank); }
  void scatterVariable(ConstArrayView<Type> send_buf, ArrayView<Type> recv_buf, Int32 root) override
  { m_mp_dispatcher->scatterVariable(send_buf,recv_buf,root); }
  void allToAll(ConstArrayView<Type> send_buf, ArrayView<Type> recv_buf, Integer count) override
  { m_mp_dispatcher->allToAll(send_buf,recv_buf,count); }
  void allToAllVariable(ConstArrayView<Type> send_buf, Int32ConstArrayView send_count,
                        Int32ConstArrayView send_index,ArrayView<Type> recv_buf,
                        Int32ConstArrayView recv_count,Int32ConstArrayView recv_index) override
  { m_mp_dispatcher->allToAllVariable(send_buf,send_count,send_index,recv_buf,recv_count,recv_index); }
  Request send(ConstArrayView<Type> send_buffer, Int32 rank, bool is_blocked) override
  { return m_mp_dispatcher->send(send_buffer,rank,is_blocked); }
  Request send(Span<const Type> recv_buffer, const PointToPointMessageInfo& message) override
  { return m_mp_dispatcher->send(recv_buffer,message); }
  Request recv(ArrayView<Type> recv_buffer, Int32 rank, bool is_blocked) override
  { return m_mp_dispatcher->receive(recv_buffer,rank,is_blocked); }
  Request receive(Span<Type> recv_buffer, const PointToPointMessageInfo& message) override
  { return m_mp_dispatcher->receive(recv_buffer,message); }
  void send(ConstArrayView<Type> send_buffer, Int32 rank) override
  { m_mp_dispatcher->send(send_buffer,rank,true); }
  void recv(ArrayView<Type> recv_buffer, Int32 rank) override
  { m_mp_dispatcher->receive(recv_buffer,rank,true); }
  Type allReduce(eReduceType op, Type send_buf) override
  { return  m_mp_dispatcher->allReduce(op,send_buf); }
  void allReduce(eReduceType op, ArrayView<Type> send_buf) override
  { m_mp_dispatcher->allReduce(op,send_buf); }

 public:
  ARCANE_MPI_EXPORT void sendRecv(ConstArrayView<Type> send_buffer, ArrayView<Type> recv_buffer, Int32 rank) override;
  ARCANE_MPI_EXPORT Type scan(eReduceType op, Type send_buf) override;
  ARCANE_MPI_EXPORT void scan(eReduceType op, ArrayView<Type> send_buf) override;
  ARCANE_MPI_EXPORT void computeMinMaxSum(Type val, Type& min_val, Type& max_val, Type& sum_val,
                                          Int32& min_rank,
                                          Int32& max_rank) override;
  ARCANE_MPI_EXPORT void computeMinMaxSum(ConstArrayView<Type> values,
                                          ArrayView<Type> min_values,
                                          ArrayView<Type> max_values,
                                          ArrayView<Type> sum_values,
                                          ArrayView<Int32> min_ranks,
                                          ArrayView<Int32> max_ranks) override;

 public:

  ITypeDispatcher<Type>* toArccoreDispatcher() override;
  MpiDatatype* datatype() const;

  public:

  virtual ARCANE_MPI_EXPORT void computeMinMaxSumNoInit(Type& min_val, Type& max_val, Type& sum_val,
                                                        Int32& min_rank,Int32& max_rank);
 private:
  MP::Mpi::MpiTypeDispatcher<Type>* m_mp_dispatcher;

 private:
  MPI_Datatype m_min_max_sum_datatype;
  MPI_Op m_min_max_sum_operator;

 private:
  void _initialize();
  MPI_Datatype _mpiDatatype();
  MpiAdapter* _adapter();
  MPI_Op _mpiReduceOperator(eReduceType rt);
  static void ARCANE_MPIOP_CALL _MinMaxSumOperator(void* a,void* b,int* len,MPI_Datatype* type);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename Type> inline MpiParallelDispatchT<Type>*
createBuiltInDispatcher(ITraceMng* tm,IMessagePassingMng* pm,MpiAdapter* adapter,MpiDatatypeList* dtlist)
{
  MpiDatatype* dt = dtlist->datatype(Type());
  return new MpiParallelDispatchT<Type>(tm,pm,adapter,dt);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

