// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
/*---------------------------------------------------------------------------*/
/* MpiControlDispatcher.cc                                     (C) 2000-2020 */
/*                                                                           */
/* Manage Control/Utility parallel messages for MPI.                         */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing_mpi/MpiControlDispatcher.h"
#include "arccore/message_passing_mpi/MpiAdapter.h"
#include "arccore/message_passing_mpi/MpiMessagePassingMng.h"
#include "arccore/message_passing/Request.h"
#include "arccore/base/NotImplementedException.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore::MessagePassing::Mpi
{
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MpiControlDispatcher::
MpiControlDispatcher(IMessagePassingMng* parallel_mng, MpiAdapter* adapter)
: m_parallel_mng(parallel_mng)
, m_adapter(adapter)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiControlDispatcher::
waitAllRequests(ArrayView<Request> requests)
{
  m_adapter->waitAllRequests(requests);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiControlDispatcher::
waitSomeRequests(ArrayView<Request> requests,
                 ArrayView<bool> indexes,
                 bool is_non_blocking)
{
  m_adapter->waitSomeRequests(requests, indexes, is_non_blocking);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IMessagePassingMng* MpiControlDispatcher::
commSplit(bool keep)
{
  return m_adapter->commSplit(keep);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiControlDispatcher::
barrier()
{
  m_adapter->barrier();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MessageId MpiControlDispatcher::
probe(PointToPointMessageInfo message)
{
  ARCCORE_UNUSED(message);
  ARCCORE_THROW(NotImplementedException,"");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arccore::MessagePassing::Mpi

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
