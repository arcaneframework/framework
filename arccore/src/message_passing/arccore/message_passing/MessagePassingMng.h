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
/* MessagePassingMng.h                                         (C) 2000-2020 */
/*                                                                           */
/* Gestionnaire des échanges de messages.                                    */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_MESSAGEPASSING_MESSAGEPASSINGMNG_H
#define ARCCORE_MESSAGEPASSING_MESSAGEPASSINGMNG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing/IMessagePassingMng.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore::MessagePassing
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface du gestionnaire des échanges de messages.
 */
class ARCCORE_MESSAGEPASSING_EXPORT MessagePassingMng
: public IMessagePassingMng
{
 public:

  MessagePassingMng(Int32 comm_rank,Int32 comm_size,IDispatchers* d);
  ~MessagePassingMng() override;

 public:

  Int32 commRank() const override { return m_comm_rank; }
  Int32 commSize() const override { return m_comm_size; }
  IDispatchers* dispatchers() override;
  ITimeMetricCollector* timeMetricCollector() const override;
  void setTimeMetricCollector(ITimeMetricCollector* c);

 private:

  Int32 m_comm_rank;
  Int32 m_comm_size;
  IDispatchers* m_dispatchers = nullptr;
  ITimeMetricCollector* m_time_metric_collector = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore::MessagePassing

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
