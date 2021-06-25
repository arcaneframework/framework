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
/* IMessagePassingMng.h                                        (C) 2000-2020 */
/*                                                                           */
/* Interface du gestionnaire des échanges de messages.                       */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_MESSAGEPASSING_IMESSAGEPASSINGMNG_H
#define ARCCORE_MESSAGEPASSING_IMESSAGEPASSINGMNG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing/MessagePassingGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore::MessagePassing
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface du gestionnaire des échanges de messages.
 *
 * Ce gestionnaire ne fait pas grand chose en lui même et se contente de
 * déléguer les opérations via l'interface IDispatchers.
 * de
 */
class ARCCORE_MESSAGEPASSING_EXPORT IMessagePassingMng
{
 public:

  virtual ~IMessagePassingMng() = default;

 public:

  //! Rang de cette instance dans le communicateur
  virtual Int32 commRank() const =0;

  //! Nombre d'instance dans le communicateur
  virtual Int32 commSize() const =0;

  //! Interface pour collecter les temps d'exécution (peut être nul)
  virtual ITimeMetricCollector* timeMetricCollector() const =0;

 public:

  virtual IDispatchers* dispatchers() =0;

 public:
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore::MessagePassing

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
