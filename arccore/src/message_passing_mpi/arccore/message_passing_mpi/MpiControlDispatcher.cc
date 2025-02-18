// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MpiControlDispatcher.cc                                     (C) 2000-2025 */
/*                                                                           */
/* Manage Control/Utility parallel messages for MPI.                         */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing_mpi/MpiAdapter.h"
#include "arccore/message_passing_mpi/MpiMessagePassingMng.h"
#include "arccore/message_passing_mpi/internal/MpiControlDispatcher.h"
#include "arccore/message_passing/Request.h"
#include "arccore/base/NotImplementedException.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MessagePassing::Mpi
{
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MpiControlDispatcher::
MpiControlDispatcher(MpiAdapter* adapter)
: m_adapter(adapter)
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

Request MpiControlDispatcher::
nonBlockingBarrier()
{
  return m_adapter->nonBlockingBarrier();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MessageId MpiControlDispatcher::
probe(const PointToPointMessageInfo& message)
{
  return m_adapter->probeMessage(message);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MessageSourceInfo MpiControlDispatcher::
legacyProbe(const PointToPointMessageInfo& message)
{
  return m_adapter->legacyProbeMessage(message);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IProfiler* MpiControlDispatcher::
profiler() const
{
  return m_adapter->profiler();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MpiControlDispatcher::
setProfiler(IProfiler* p)
{
  m_adapter->setProfiler(p);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arccore::MessagePassing::Mpi

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
