// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MpiMessagePassingMng.cc                                     (C) 2000-2025 */
/*                                                                           */
/* MPI implementation of the message exchange manager.                       */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing_mpi/MpiMessagePassingMng.h"
#include "arccore/message_passing/Communicator.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MessagePassing::Mpi
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MpiMessagePassingMng::
MpiMessagePassingMng(const BuildInfo& bi)
: MessagePassingMng(bi.commRank(), bi.commSize(), bi.dispatchers())
, m_mpi_communicator(bi.communicator())
{
  setCommunicator(Communicator{ m_mpi_communicator });
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::MessagePassing::Mpi

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
