// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MessagePassingUtils.cc                                      (C) 2000-2026 */
/*                                                                           */
/* Various utilities to handle message passing.                              */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/alina/MessagePassingUtils.h"

#include <mpi.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Alina
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

mpi_communicator::
mpi_communicator(MPI_Comm comm)
: comm(comm)
{
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);
  m_message_passing_mng = MessagePassing::Mpi::StandaloneMpiMessagePassingMng::createRef(comm);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

mpi_communicator::
mpi_communicator(IMessagePassingMng* mpm_comm)
{
  MessagePassing::Communicator c = mpm_comm->communicator();
  if (!c.isValid())
    ARCCORE_FATAL("Invalid 'IMessagePassingMng' communicator. Only MPI implementation is currently supported");
  comm = static_cast<MPI_Comm>(c);
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);
  m_message_passing_mng = makeRef(mpm_comm);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Alina

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
