// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MessagePassingMpiEnum.cc                                    (C) 2000-2025 */
/*                                                                           */
/* Enumeration des différentes operations MPI.                               */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing_mpi/internal/MessagePassingMpiEnum.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MessagePassing::Mpi
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MpiInfo::
MpiInfo(eMpiName mpi_operation)
{
  switch (mpi_operation) {
  case eMpiName::Bcast:
    m_name = "MPI_Bcast";
    m_description = "Blocking Broadcast";
    break;
  case eMpiName::Gather:
    m_name = "MPI_Gather";
    m_description = "Blocking Gather";
    break;
  case eMpiName::Gatherv:
    m_name = "MPI_Gatherv";
    m_description = "Blocking Gather with variable message size";
    break;
  case eMpiName::Allgather:
    m_name = "MPI_Allgather";
    m_description = "Blocking allGather";
    break;
  case eMpiName::Allgatherv:
    m_name = "MPI_Allgatherv";
    m_description = "Blocking allGather with variable message size";
    break;
  case eMpiName::Scatterv:
    m_name = "MPI_Scatterv";
    m_description = "Blocking Scatter with variable message size";
    break;
  case eMpiName::Alltoall:
    m_name = "MPI_Alltoall";
    m_description = "Blocking Alltoall";
    break;
  case eMpiName::Alltoallv:
    m_name = "MPI_Alltoallv";
    m_description = "Blocking Alltoall with variable message size";
    break;
  case eMpiName::Barrier:
    m_name = "MPI_Barrier";
    m_description = "Blocking Barrier";
    break;
  case eMpiName::Reduce:
    m_name = "MPI_Reduce";
    m_description = "Blocking Reduce";
    break;
  case eMpiName::Allreduce:
    m_name = "MPI_Allreduce";
    m_description = "Blocking Allreduce";
    break;
  case eMpiName::Scan:
    m_name = "MPI_Scan";
    m_description = "Blocking Scan";
    break;
  case eMpiName::Sendrecv:
    m_name = "MPI_Sendrecv";
    m_description = "Blocking Sendrecv";
    break;
  case eMpiName::Isend:
    m_name = "MPI_Isend";
    m_description = "Non-blocking Send";
    break;
  case eMpiName::Send:
    m_name = "MPI_Send";
    m_description = "Blocking Send";
    break;
  case eMpiName::Irecv:
    m_name = "MPI_Irecv";
    m_description = "Non-blocking Recv";
    break;
  case eMpiName::Recv:
    m_name = "MPI_Recv";
    m_description = "Blocking Recv";
    break;
  case eMpiName::Test:
    m_name = "MPI_Test";
    m_description = "Test";
    break;
  case eMpiName::Probe:
    m_name = "MPI_Probe";
    m_description = "Probe";
    break;
  case eMpiName::Get_count:
    m_name = "MPI_Get_count";
    m_description = "Get count";
    break;
  case eMpiName::Wait:
    m_name = "MPI_Wait";
    m_description = "Wait";
    break;
  case eMpiName::Waitall:
    m_name = "MPI_Waitall";
    m_description = "Waitall";
    break;
  case eMpiName::Testsome:
    m_name = "MPI_Testsome";
    m_description = "Testsome";
    break;
  case eMpiName::Waitsome:
    m_name = "MPI_Waitsome";
    m_description = "Waitsome";
    break;
  default:
    m_name = "Unkown";
    m_description = "Unkown";
    break;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
const String& MpiInfo::
name() const
{
  return m_name;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const String& MpiInfo::
description() const
{
  return m_description;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::MessagePassing::Mpi

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
