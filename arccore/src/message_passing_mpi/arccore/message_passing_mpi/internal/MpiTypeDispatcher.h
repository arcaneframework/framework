// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MpiTypeDispatcher.h                                         (C) 2000-2025 */
/*                                                                           */
/* Gestion des messages pour un type de données.                             */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_MESSAGEPASSINGMPI_INTERNAL_MPITYPEDISPATCHER_H
#define ARCCORE_MESSAGEPASSINGMPI_INTERNAL_MPITYPEDISPATCHER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing_mpi/MessagePassingMpiGlobal.h"
#include "arccore/message_passing/ITypeDispatcher.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MessagePassing::Mpi
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename Type>
class MpiTypeDispatcher
: public ITypeDispatcher<Type>
{
 public:

  MpiTypeDispatcher(IMessagePassingMng* parallel_mng, MpiAdapter* adapter, MpiDatatype* datatype);
  ~MpiTypeDispatcher();

 public:

  void finalize() override {}
  void broadcast(Span<Type> send_buf, Int32 rank) override;
  void allGather(Span<const Type> send_buf,Span<Type> recv_buf) override;
  void allGatherVariable(Span<const Type> send_buf,Array<Type>& recv_buf) override;
  void gather(Span<const Type> send_buf,Span<Type> recv_buf,Int32 rank) override;
  void gatherVariable(Span<const Type> send_buf,Array<Type>& recv_buf,Int32 rank) override;
  void scatterVariable(Span<const Type> send_buf,Span<Type> recv_buf,Int32 root) override;
  void allToAll(Span<const Type> send_buf,Span<Type> recv_buf,Int32 count) override;
  void allToAllVariable(Span<const Type> send_buf,ConstArrayView<Int32> send_count,
                        ConstArrayView<Int32> send_index,Span<Type> recv_buf,
                        ConstArrayView<Int32> recv_count,ConstArrayView<Int32> recv_index) override;
  Request send(Span<const Type> send_buffer,Int32 rank,bool is_blocked) override;
  Request send(Span<const Type> send_buffer,const PointToPointMessageInfo& message) override;
  Request receive(Span<Type> recv_buffer,Int32 rank,bool is_blocked) override;
  Request receive(Span<Type> recv_buffer,const PointToPointMessageInfo& message) override;
  Type allReduce(eReduceType op,Type send_buf) override;
  void allReduce(eReduceType op,Span<Type> send_buf) override;
  Request nonBlockingAllReduce(eReduceType op, Span<const Type> send_buf, Span<Type> recv_buf) override;
  Request nonBlockingAllGather(Span<const Type> send_buf, Span<Type> recv_buf) override;
  Request nonBlockingBroadcast(Span<Type> send_buf, Int32 rank) override;
  Request nonBlockingGather(Span<const Type> send_buf, Span<Type> recv_buf, Int32 rank) override;
  Request nonBlockingAllToAll(Span<const Type> send_buf, Span<Type> recv_buf, Int32 count) override;
  Request nonBlockingAllToAllVariable(Span<const Type> send_buf, ConstArrayView<Int32> send_count,
                                      ConstArrayView<Int32> send_index, Span<Type> recv_buf,
                                      ConstArrayView<Int32> recv_count, ConstArrayView<Int32> recv_index) override;
  Request gather(GatherMessageInfo<Type>& gather_info) override;
  void gatherVariable(Span<const Type> send_buf,Span<Type> recv_buf,Span<const Int32> displacements,
                      Span<const Int32> counts,Int32 rank);

 public:

  MpiDatatype* datatype() const { return m_datatype; }
  IMessagePassingMng* messagePassingMng() const { return m_parallel_mng; }
  MpiAdapter* adapter() const { return m_adapter; }
  void setDestroyDatatype(bool v) { m_is_destroy_datatype = v; }
  bool isDestroyDatatype() const { return m_is_destroy_datatype; }

 private:

  IMessagePassingMng* m_parallel_mng;
  MpiAdapter* m_adapter;
  MpiDatatype* m_datatype;
  bool m_is_destroy_datatype = false;

 private:

  void _gatherVariable2(Span<const Type> send_buf, Array<Type>& recv_buf, Integer rank);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore::MessagePassing::Mpi

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

