//
// Created by chevalier on 14/01/2020.
//

#ifndef ARCCORE_MESSAGEPASSINGMPI_MPICONTROLDISPATCHER_H
#define ARCCORE_MESSAGEPASSINGMPI_MPICONTROLDISPATCHER_H

#include <arccore/message_passing_mpi/MessagePassingMpiGlobal.h>
#include <arccore/message_passing/IControlDispatcher.h>

namespace Arccore {
namespace MessagePassing {
namespace Mpi {

class MpiControlDispatcher : public IControlDispatcher
{
 public:
  MpiControlDispatcher(IMessagePassingMng* parallel_mng, MpiAdapter* adapter);
  ~MpiControlDispatcher() = default;
 public:
  void waitAllRequests(ArrayView<Request> requests) override;

  void waitSomeRequests(ArrayView<Request> requests, ArrayView<bool> indexes, bool is_non_blocking) override;

  IMessagePassingMng* commSplit(bool keep) override;

 private:
  IMessagePassingMng* m_parallel_mng;
  MpiAdapter* m_adapter;
};




}
}
}


#endif //ARCCORE_MESSAGEPASSINGMPI_MPICONTROLDISPATCHER_H
