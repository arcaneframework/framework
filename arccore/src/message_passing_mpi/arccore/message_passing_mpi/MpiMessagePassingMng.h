// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MpiMessagePassingMng.h                                      (C) 2000-2025 */
/*                                                                           */
/* Implémentation MPI du gestionnaire des échanges de messages.              */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_MESSAGEPASSINGMPI_MPIMESSAGEPASSINGMNG_H
#define ARCCORE_MESSAGEPASSINGMPI_MPIMESSAGEPASSINGMNG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing_mpi/MessagePassingMpiGlobal.h"
#include "arccore/message_passing/MessagePassingMng.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MessagePassing::Mpi
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Implémentation MPI du gestionnaire des échanges de messages.
 */
class ARCCORE_MESSAGEPASSINGMPI_EXPORT MpiMessagePassingMng
: public MessagePassingMng
{
 public:

  class BuildInfo
  {
   public:

    BuildInfo(Int32 comm_rank, Int32 comm_size, IDispatchers* dispatchers, MPI_Comm comm)
    : m_comm_rank(comm_rank)
    , m_comm_size(comm_size)
    , m_dispatchers(dispatchers)
    , m_communicator(comm)
    {}

   public:

    Int32 commRank() const { return m_comm_rank; }
    Int32 commSize() const { return m_comm_size; }
    IDispatchers* dispatchers() const { return m_dispatchers; }
    MPI_Comm communicator() const { return m_communicator; }

   private:

    Int32 m_comm_rank;
    Int32 m_comm_size;
    IDispatchers* m_dispatchers;
    MPI_Comm m_communicator;
  };

 public:

  explicit MpiMessagePassingMng(const BuildInfo& bi);

 public:

  const MPI_Comm* getMPIComm() const { return &m_mpi_communicator; }

 private:

  MPI_Comm m_mpi_communicator;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arccore::MessagePassing::Mpi

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
