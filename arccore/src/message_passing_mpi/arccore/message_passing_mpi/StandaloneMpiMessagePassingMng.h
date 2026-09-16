// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* StandaloneMpiMessagePassingMng.h                            (C) 2000-2026 */
/*                                                                           */
/* Standalone version of MpiMessagePassingMng.                               */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_MESSAGEPASSINGMPI_STANDALONEMPIMESSAGEPASSINGMNG_H
#define ARCCORE_MESSAGEPASSINGMPI_STANDALONEMPIMESSAGEPASSINGMNG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing_mpi/MpiMessagePassingMng.h"

#include "arccore/base/RefDeclarations.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MessagePassing::Mpi
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Standalone version of IMessagePassingMng associated to a MPI Communicator
 *
 * Creation is done via the static method createRef() or deprectated
 * method create().
 *
 * From the returned IMessagePassingMng, it is possible to retreive
 * the associated MPI_Comm using helper functions fillMpiCommunicatorIfValid()
 * or toMpiCommunicator().
 */
class ARCCORE_MESSAGEPASSINGMPI_EXPORT StandaloneMpiMessagePassingMng
{
 public:

  /*!
   * \brief Creates a manager associated with the communicator \a comm.
   *
   * \deprecated Use createRef() instead because it has automatic memory
   * management.
   */
  ARCCORE_DEPRECATED_REASON("Y2026: Use createRef() instead")
  static MpiMessagePassingMng* create(MPI_Comm comm, bool clean_comm = false);

  /*!
   * \brief Creates a manager associated with the communicator \a comm.
   *
   * If \a clean_comm is true, MPI_Comm_free() is called on \a comm
   * when the instance is destroyed.
   */
  static Ref<IMessagePassingMng> createRef(MPI_Comm comm, bool clean_comm = false);

  /*!
   * \brief Creates a manager associated with the MPI_WORLD_COMM communicator.
   */
  static Ref<IMessagePassingMng> createWorldRef();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::MessagePassing::Mpi

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
