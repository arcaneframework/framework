/*---------------------------------------------------------------------------*/
/* MpiTypeDispatcher.h                                         (C) 2000-2018 */
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

namespace Arccore
{
namespace MessagePassing
{
namespace Mpi
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
  void broadcast(ArrayView<Type> send_buf,Int32 rank) override;
  void allGather(ConstArrayView<Type> send_buf,ArrayView<Type> recv_buf) override;
  void allGatherVariable(ConstArrayView<Type> send_buf,Array<Type>& recv_buf) override;
  void gather(ConstArrayView<Type> send_buf,ArrayView<Type> recv_buf,Int32 rank) override;
  void gatherVariable(ConstArrayView<Type> send_buf,Array<Type>& recv_buf,Int32 rank) override;
  void scatterVariable(ConstArrayView<Type> send_buf,ArrayView<Type> recv_buf,Int32 root) override;
  void allToAll(ConstArrayView<Type> send_buf,ArrayView<Type> recv_buf,Integer count) override;
  void allToAllVariable(ConstArrayView<Type> send_buf,Int32ConstArrayView send_count,
                       Int32ConstArrayView send_index,ArrayView<Type> recv_buf,
                       Int32ConstArrayView recv_count,Int32ConstArrayView recv_index) override;
  Request send(ConstArrayView<Type> send_buffer,Int32 rank,bool is_blocked) override;
  Request receive(ArrayView<Type> recv_buffer,Int32 rank,bool is_blocked) override;
  Type allReduce(eReduceType op,Type send_buf) override;
  void allReduce(eReduceType op,ArrayView<Type> send_buf) override;
 public:
  MpiDatatype* datatype() const { return m_datatype; }
  IMessagePassingMng* messagePassingMng() const { return m_parallel_mng; }
  MpiAdapter* adapter() const { return m_adapter; }
 private:
  IMessagePassingMng* m_parallel_mng;
  MpiAdapter* m_adapter;
  MpiDatatype* m_datatype;
 private:
  void _gatherVariable2(ConstArrayView<Type> send_buf,Array<Type>& recv_buf,Integer rank);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Mpi
} // End namespace MessagePassing
} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

