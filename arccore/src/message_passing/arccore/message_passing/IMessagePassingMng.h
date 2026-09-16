// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMessagePassingMng.h                                        (C) 2000-2026 */
/*                                                                           */
/* Interface of the message passing manager.                                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_MESSAGEPASSING_IMESSAGEPASSINGMNG_H
#define ARCCORE_MESSAGEPASSING_IMESSAGEPASSINGMNG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing/MessagePassingGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MessagePassing
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Destroys the message passing instance \a p.
 *
 * \warning This should only be used when using deprecated functions
 * which do not return instances of Ref<IMessagePassingMng> and instead
 * only return a IMessagePassingMng*.
 */
extern "C++" void ARCCORE_MESSAGEPASSING_EXPORT
mpDelete(IMessagePassingMng* p);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface of the message passing manager.
 *
 * This class uses a semantic by reference and instances are automatically
 * destroyed when needed.
 *
 * Most of the operations on this class use free functions (like mpSend() or mpReceive())
 * which are defined in file 'Messages.h'.
 *
 * %Arcane provides several implementation for message passing, using MPI,
 * shared memory message passing or hybrid (MPI + shared memory) message passing.
 * When using MPI, it is possible to create an instance of this class using
 * the class StandaloneMpiMessagePassingMng.
 */
class ARCCORE_MESSAGEPASSING_EXPORT IMessagePassingMng
{
  friend void ARCCORE_MESSAGEPASSING_EXPORT mpDelete(IMessagePassingMng*);
  ARCCORE_DECLARE_REFERENCE_COUNTED_INCLASS_METHODS();

 public:

  virtual ~IMessagePassingMng() = default;

 public:

  //! Rank of this instance in the communicator
  virtual Int32 commRank() const = 0;

  //! Number of instances in the communicator
  virtual Int32 commSize() const = 0;

  //! Interface for collecting execution times (can be null)
  virtual ITimeMetricCollector* timeMetricCollector() const = 0;

  /*!
   * \brief MPI communicator associated with this instance.
   *
   * The communicator is only valid if the instance is associated with an
   * MPI implementation. You can convert a communicator to a MPI communicator
   * using helper methods toMpiCommunicator() and fillMpiCommunicatorIfValid()
   * or like that:
   *
   * \code
   * IMessagePassingMng* mpm = ...;
   * Communicator comm = mpm->communicator();
   * if (comm.isValid())
   *   MPI_Comm mpi_comm = static_cast<MPI_Comm>(comm).
   * \endcode
   */
  virtual Communicator communicator() const;

 public:

  virtual IDispatchers* dispatchers() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::MessagePassing

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
