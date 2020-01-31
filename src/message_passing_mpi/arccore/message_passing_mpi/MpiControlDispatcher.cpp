//
// Created by chevalier on 14/01/2020.
//

#include "MpiControlDispatcher.h"

#include "MpiAdapter.h"
#include "MpiMessagePassingMng.h"


namespace Arccore::MessagePassing::Mpi
{
MpiControlDispatcher::MpiControlDispatcher(IMessagePassingMng* parallel_mng, MpiAdapter* adapter)
: m_parallel_mng(parallel_mng)
, m_adapter(adapter)
{
}

void MpiControlDispatcher::waitAllRequests(ArrayView<Request> requests)
{
  m_adapter->waitAllRequests(requests);
}

void MpiControlDispatcher::waitSomeRequests(ArrayView<Request> requests,
                                            ArrayView<bool> indexes,
                                            bool is_non_blocking)
{
  m_adapter->waitSomeRequests(requests, indexes, is_non_blocking);
}

IMessagePassingMng* MpiControlDispatcher::commSplit(bool keep)
{
  return m_adapter->commSplit(keep);
}

} // namespace Arccore::MessagePassing::Mpi
