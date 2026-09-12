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

AlinaCommunicator::
AlinaCommunicator(MPI_Comm comm)
: comm(comm)
{
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);
  m_message_passing_mng = MessagePassing::Mpi::StandaloneMpiMessagePassingMng::createRef(comm);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AlinaCommunicator::
AlinaCommunicator(IMessagePassingMng* mpm_comm)
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

void AlinaCommunicator::
check(bool cond, const String& message)
{
  int lc = (cond) ? 1 : 0;
  int gc = mpAllReduce(m_message_passing_mng.get(), MessagePassing::eReduceType::ReduceMin, cond);

  if (gc != 0)
    return;
  IMessagePassingMng* pm = m_message_passing_mng.get();
  UniqueArray<int> c(size);
  if (rank == 0)
    c.resize(size);
  ConstArrayView<int> in_view(1, &lc);
  mpGather(pm, in_view, c, 0);
  if (rank == 0) {
    std::cerr << "Failed assumption: " << message << std::endl;
    std::cerr << "Offending processes:";
    for (int i = 0; i < size; ++i)
      if (!c[i])
        std::cerr << " " << i;
    std::cerr << std::endl;
  }
  mpBarrier(pm);
  ARCCORE_FATAL("CheckError in MessagePassingUtils: {0}", message);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Alina

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
