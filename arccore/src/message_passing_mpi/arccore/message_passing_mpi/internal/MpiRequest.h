// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MpiRequest.h                                                (C) 2000-2025 */
/*                                                                           */
/* Specialization of 'Request' for MPI.                                      */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_MESSAGEPASSINGMPI_INTERNAL_MPIREQUEST_H
#define ARCCORE_MESSAGEPASSINGMPI_INTERNAL_MPIREQUEST_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing_mpi/MessagePassingMpiGlobal.h"

#include "arccore/message_passing/Request.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MessagePassing::Mpi
{
class MpiAdapter;

/*!
 * \brief MPI specialization of a 'Request'.
 *
 * This class ensures that an MPI request is properly constructed
 * from an MPI_Request.
 */
class ARCCORE_MESSAGEPASSINGMPI_EXPORT MpiRequest
: public Request
{
 public:

  MpiRequest() = default;
  MpiRequest(int ret_value, MpiAdapter* creator, MPI_Request mpi_request)
  : Request(ret_value, creator, mpi_request)
  {}
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::MessagePassing::Mpi

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
