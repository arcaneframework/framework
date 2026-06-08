// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IRequestList.h                                              (C) 2000-2025 */
/*                                                                           */
/* Interface for a message request list.                                     */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_MESSAGEPASSING_REQUESTLIST_H
#define ARCCORE_MESSAGEPASSING_REQUESTLIST_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing/MessagePassingGlobal.h"
#include "arccore/base/BaseTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MessagePassing
{
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Message request list.
 */
class ARCCORE_MESSAGEPASSING_EXPORT IRequestList
{
 public:

  virtual ~IRequestList() = default;

 public:

  //! Adds the request \a r to the list of requests
  virtual void add(Request r) =0;

  //! Adds the list of requests \a rlist to the list of requests
  virtual void add(Span<Request> rlist) =0;

  //! The index-th request in the list
  virtual Request request(Int32 index) const =0;

  //! Number of requests
  virtual Int32 size() const =0;

  //! Removes all requests from the list
  virtual void clear() =0;

  /*!
   * \brief Waits for or tests the completion of one or more requests.
   *
   * It returns the number of newly completed requests.
   * It is then possible to test if a request is finished via the
   * isRequestDone() method or to retrieve the indices of the
   * completed requests via doneRequestIndexes().
   *
   * \note Requests completed after a call to wait() remain
   * in the list of requests. The method
   * removeDoneRequests() must be called if you wish to remove them.
   */
  virtual Int32 wait(eWaitType wait_type) =0;

  //! Indicates if the request is finished since the last call to wait()
  virtual bool isRequestDone(Int32 index) const =0;

  /*!
   * \brief Removes completed requests from the list.
   *
   * All requests for which isRequestDone() is true are
   * removed from the list of requests.
   * After calling this method, it is considered that there are no more
   * completed requests. Consequently, doneRequestIndexes() will be empty
   * and isRequestDone() will always return \a false.
   */
  virtual void removeDoneRequests() =0;

  /*!
   * \brief Indices in the request array of requests completed during
   * the last call to wait().
   */
  virtual ConstArrayView<Int32> doneRequestIndexes() const =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore::MessagePassing

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
