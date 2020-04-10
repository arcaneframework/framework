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
/* MpiMessagePassingMng.h                                      (C) 2000-2020 */
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

namespace Arccore::MessagePassing::Mpi
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
    BuildInfo(Int32 comm_rank,Int32 comm_size,IDispatchers* dispatchers,MPI_Comm comm)
    : m_comm_rank(comm_rank), m_comm_size(comm_size),
      m_dispatchers(dispatchers), m_communicator(comm){}
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

  MpiMessagePassingMng(const BuildInfo& bi);
  ~MpiMessagePassingMng() override;

 public:

  const MPI_Comm* getMPIComm() const
  {
    return &m_communicator;
  }

 private:

  MPI_Comm m_communicator;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore::MessagePassing::Mpi

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
