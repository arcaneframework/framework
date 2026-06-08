// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMessagePassingMng.h                                        (C) 2000-2025 */
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
 * \brief Destroys the instance \a p.
 *
 * The instance \a p must not be used after this call
 */
extern "C++" void ARCCORE_MESSAGEPASSING_EXPORT
mpDelete(IMessagePassingMng* p);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Interface of the message passing manager.
 *
 * This manager does not do much itself and merely
 * delegates operations via the IDispatchers interface.
 *
 * Instances of these classes must be destroyed via the method
 * mpDelete().
 */
class ARCCORE_MESSAGEPASSING_EXPORT IMessagePassingMng
{
  friend void ARCCORE_MESSAGEPASSING_EXPORT mpDelete(IMessagePassingMng*);
  ARCCORE_DECLARE_REFERENCE_COUNTED_INCLASS_METHODS();

 public:

  // TODO: Rendre obsolète fin 2022: [[deprecated("Use mpDelete() instead")]]
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
   * MPI implementation.
   */
  virtual Communicator communicator() const;

 public:

  virtual IDispatchers* dispatchers() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore::MessagePassing

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
