// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MessagePassingMng.cc                                        (C) 2000-2024 */
/*                                                                           */
/* Gestionnaire des échanges de messages.                                    */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing/MessagePassingMng.h"
#include "arccore/message_passing/Communicator.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore::MessagePassing
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MessagePassingMng::
MessagePassingMng(Int32 comm_rank, Int32 comm_size, IDispatchers* d)
: m_comm_rank(comm_rank)
, m_comm_size(comm_size)
, m_dispatchers(d)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MessagePassingMng::
~MessagePassingMng()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IDispatchers* MessagePassingMng::
dispatchers()
{
  return m_dispatchers;
}

ITimeMetricCollector* MessagePassingMng::
timeMetricCollector() const
{
  return m_time_metric_collector;
}

void MessagePassingMng::
setTimeMetricCollector(ITimeMetricCollector* c)
{
  m_time_metric_collector = c;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" void ARCCORE_MESSAGEPASSING_EXPORT
mpDelete(IMessagePassingMng* p)
{
  delete p;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Communicator IMessagePassingMng::
communicator() const
{
  return {};
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore::MessagePassing

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
