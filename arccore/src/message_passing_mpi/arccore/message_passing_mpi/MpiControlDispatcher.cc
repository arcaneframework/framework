// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2020 IFPEN-CEA
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
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

MessageId MpiControlDispatcher::
probe(const PointToPointMessageInfo& message)
{
  return m_adapter->probeMessage(message);
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
