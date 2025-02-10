// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IControlDispatcher.h                                        (C) 2000-2025 */
/*                                                                           */
/* Manage Control/Utility parallel messages.                                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_MESSAGEPASSING_ICONTROLDISPATCHER_H
#define ARCCORE_MESSAGEPASSING_ICONTROLDISPATCHER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing/MessagePassingGlobal.h"
#include "arccore/collections/CollectionsGlobal.h"
#include "arccore/base/BaseTypes.h"
#include "arccore/base/Ref.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MessagePassing
{
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Manage control streams for parallel messages.
 */
class ARCCORE_MESSAGEPASSING_EXPORT IControlDispatcher
{
 public:

  virtual ~IControlDispatcher() = default;

 public:

  virtual void waitAllRequests(ArrayView<Request> requests) = 0;

  virtual void waitSomeRequests(ArrayView<Request> requests,
                                ArrayView<bool> indexes, bool is_non_blocking) = 0;

  virtual IMessagePassingMng* commSplit(bool keep) = 0;

  virtual void barrier() = 0;

  virtual Request nonBlockingBarrier() = 0;

  virtual MessageId probe(const PointToPointMessageInfo& message) = 0;

  // NOTE novembre 2022
  // Pour l'instant pas encore virtual pure pour rester compatible avec le code
  // existant. L'implémentation lève une exception NotSupportedException
  virtual MessageSourceInfo legacyProbe(const PointToPointMessageInfo& message);

  //! Création d'une liste de requêtes associé à ce gestionnaire
  virtual Ref<IRequestList> createRequestListRef() = 0;

 public:

  virtual IProfiler* profiler() const = 0;
  virtual void setProfiler(IProfiler* p) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::MessagePassing

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
