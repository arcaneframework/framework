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
/* MpiSerializeMessageList.h                                   (C) 2000-2020 */
/*                                                                           */
/* Implémentation de ISerializeMessageList pour MPI.                         */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_MESSAGEPASSINGMPI_MPISERIALIZEMESSAGELIST_H
#define ARCCORE_MESSAGEPASSINGMPI_MPISERIALIZEMESSAGELIST_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing_mpi/MessagePassingMpiGlobal.h"
#include "arccore/message_passing/ISerializeMessageList.h"
#include "arccore/message_passing/Request.h"
#include "arccore/trace/TraceGlobal.h"
#include "arccore/trace/TimeMetric.h"
#include "arccore/base/BaseTypes.h"
#include "arccore/serialize/SerializeGlobal.h"
#include "arccore/collections/Array.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore::MessagePassing::Mpi
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class MyMpiParallelMng;
class MpiSerializeMessage;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCCORE_MESSAGEPASSINGMPI_EXPORT MpiSerializeMessageRequest
{
 public:
  MpiSerializeMessageRequest()
  : m_mpi_message(0), m_request() {}
  MpiSerializeMessageRequest(MpiSerializeMessage* mpi_message,Request request)
  : m_mpi_message(mpi_message), m_request(request) {}
 public:
  MpiSerializeMessage* m_mpi_message;
  Request m_request;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Implémentation MPI de la gestion des 'ISerializeMessage'.
 */
class ARCCORE_MESSAGEPASSINGMPI_EXPORT MpiSerializeMessageList
: public ISerializeMessageList
{
 private:

  class _SortMessages;

 public:

  MpiSerializeMessageList(MpiSerializeDispatcher* dispatcher);

 public:

  void addMessage(ISerializeMessage* msg) override;
  void processPendingMessages() override;
  Integer waitMessages(eWaitType wait_type) override;
  Ref<ISerializeMessage>
  createAndAddMessage(MessageRank destination,ePointToPointMessageType type) override;

  Request _processOneMessageGlobalBuffer(MpiSerializeMessage* msm,MessageRank source,MessageTag mpi_tag);
  Request _processOneMessage(MpiSerializeMessage* msm,MessageRank source,MessageTag mpi_tag);

 private:

  Integer _waitMessages(eWaitType wait_type);
  Integer _waitMessages2(eWaitType wait_type);

 private:

  MpiSerializeDispatcher* m_dispatcher = nullptr;
  MpiAdapter* m_adapter = nullptr;
  ITraceMng* m_trace = nullptr;
  UniqueArray<MpiSerializeMessage*> m_messages_to_process;
  UniqueArray<MpiSerializeMessageRequest> m_messages_request;
  TimeMetricAction m_message_passing_phase;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End Namespace  Arccore::MessagePassing::Mpi

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

