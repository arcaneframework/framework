// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
/*---------------------------------------------------------------------------*/
/* MpiTypeDispatcher.h                                         (C) 2000-2020 */
/*                                                                           */
/* Gestion des messages pour un type de données.                             */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_MESSAGEPASSINGMPI_MPITYPEDISPATCHER_H
#define ARCCORE_MESSAGEPASSINGMPI_MPITYPEDISPATCHER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing_mpi/MessagePassingMpiGlobal.h"
#include "arccore/message_passing/ITypeDispatcher.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore::MessagePassing::Mpi
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename Type>
class MpiTypeDispatcher
: public ITypeDispatcher<Type>
{
 public:
  MpiTypeDispatcher(IMessagePassingMng* parallel_mng,MpiAdapter* adapter,MpiDatatype* datatype);
  ~MpiTypeDispatcher();
 public:
  void finalize() override {}
  void broadcast(Span<Type> send_buf,Int32 rank) override;
  void allGather(Span<const Type> send_buf,Span<Type> recv_buf) override;
  void allGatherVariable(Span<const Type> send_buf,Array<Type>& recv_buf) override;
  void gather(Span<const Type> send_buf,Span<Type> recv_buf,Int32 rank) override;
  void gatherVariable(Span<const Type> send_buf,Array<Type>& recv_buf,Int32 rank) override;
  void scatterVariable(Span<const Type> send_buf,Span<Type> recv_buf,Int32 root) override;
  void allToAll(Span<const Type> send_buf,Span<Type> recv_buf,Int32 count) override;
  void allToAllVariable(Span<const Type> send_buf,Int32ConstArrayView send_count,
                        Int32ConstArrayView send_index,Span<Type> recv_buf,
                        Int32ConstArrayView recv_count,Int32ConstArrayView recv_index) override;
  Request send(Span<const Type> send_buffer,Int32 rank,bool is_blocked) override;
  Request send(Span<const Type> send_buffer,PointToPointMessageInfo message) override;
  Request receive(Span<Type> recv_buffer,Int32 rank,bool is_blocked) override;
  Request receive(Span<Type> recv_buffer,PointToPointMessageInfo message) override;
  Type allReduce(eReduceType op,Type send_buf) override;
  void allReduce(eReduceType op,Span<Type> send_buf) override;
 public:
  MpiDatatype* datatype() const { return m_datatype; }
  IMessagePassingMng* messagePassingMng() const { return m_parallel_mng; }
  MpiAdapter* adapter() const { return m_adapter; }
 private:
  IMessagePassingMng* m_parallel_mng;
  MpiAdapter* m_adapter;
  MpiDatatype* m_datatype;
 private:
  void _gatherVariable2(Span<const Type> send_buf,Array<Type>& recv_buf,Integer rank);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore::MessagePassing::Mpi

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

