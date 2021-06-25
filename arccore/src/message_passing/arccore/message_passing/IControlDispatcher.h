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
/* IControlDispatcher.h                                        (C) 2000-2020 */
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

namespace Arccore::MessagePassing
{
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Manage control streams for parallel messages.
 */
class IControlDispatcher
{
 public:
  virtual ~IControlDispatcher() = default;

 public:
  virtual void waitAllRequests(ArrayView<Request> requests) =0;

  virtual void waitSomeRequests(ArrayView<Request> requests,
                                ArrayView<bool> indexes, bool is_non_blocking) =0;

  virtual IMessagePassingMng* commSplit(bool keep) =0;

  virtual void barrier() =0;

  virtual MessageId probe(const PointToPointMessageInfo& message) =0;

  //! Création d'une liste de requêtes associé à ce gestionnaire
  virtual Ref<IRequestList> createRequestListRef() =0;

 public:

  virtual IProfiler* profiler() const =0;
  virtual void setProfiler(IProfiler* p) =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore::MessagePassing

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
