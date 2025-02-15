// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MpiRequestList.h                                            (C) 2000-2025 */
/*                                                                           */
/* Liste de requêtes MPI.                                                    */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_MESSAGEPASSINGMPI_MPIREQUESTLIST_H
#define ARCCORE_MESSAGEPASSINGMPI_MPIREQUESTLIST_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing/RequestListBase.h"
#include "arccore/message_passing_mpi/MessagePassingMpiGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MessagePassing::Mpi
{
/*!
 * \brief Liste de requêtes MPI.
 */
class ARCCORE_MESSAGEPASSINGMPI_EXPORT MpiRequestList
: public internal::RequestListBase
{
 public:

  MpiRequestList(MpiAdapter* adapter) : m_adapter(adapter){}

 public:

  void _wait(eWaitType wait_type) override;

 private:

  MpiAdapter* m_adapter;
  UniqueArray<MPI_Status> m_requests_status;

 private:

  void _doWaitSome(bool is_non_blocking);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore::MessagePassing

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

