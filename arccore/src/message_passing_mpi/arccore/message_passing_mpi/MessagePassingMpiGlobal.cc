// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MessagePassingMpiGlobal.h                                   (C) 2000-2026 */
/*                                                                           */
/* Global definitions for the 'MessagePassingMpi' component of 'Arccore'.    */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing_mpi/MessagePassingMpiGlobal.h"

#include "arccore/message_passing/IMessagePassingMng.h"
#include "arccore/message_passing/Communicator.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MPI_Comm MessagePassing::
toMpiCommunicator(IMessagePassingMng* mpm)
{
  MPI_Comm mpi_comm = MPI_COMM_NULL;
  if (mpm) {
    Communicator comm = mpm->communicator();
    if (comm.isValid())
      mpi_comm = static_cast<MPI_Comm>(comm);
  }
  return mpi_comm;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool MessagePassing::
fillMpiCommunicatorIfValid(IMessagePassingMng* mpm, MPI_Comm* mpi_comm)
{
  if (!mpm)
    return false;
  Communicator comm = mpm->communicator();
  if (!comm.isValid())
    return false;
  *mpi_comm = static_cast<MPI_Comm>(comm);
  return true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
