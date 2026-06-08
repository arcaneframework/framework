// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SerializeMessageList.h                                      (C) 2000-2025 */
/*                                                                           */
/* Serialization message list.                                               */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_MESSAGEPASSING_INTERNAL_SERIALIZEMESSAGEMESSAGELIST_H
#define ARCCORE_MESSAGEPASSING_INTERNAL_SERIALIZEMESSAGEMESSAGELIST_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing/ISerializeMessageList.h"

#include "arccore/message_passing/PointToPointMessageInfo.h"
#include "arccore/base/Ref.h"
#include "arccore/trace/internal/TimeMetric.h"
#include "arccore/collections/Array.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MessagePassing::internal
{
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Serialization message list.
 *
 * This class supports the case where an implementation does not support
 * messages having any rank as destination (for example
 * the hybrid mode).
 */
class ARCCORE_MESSAGEPASSING_EXPORT SerializeMessageList
: public ISerializeMessageList
{
  using SerializeMessageContainer = UniqueArray<BasicSerializeMessage*>;

  struct ProbeInfo
  {
   public:

    ProbeInfo() = default;
    ProbeInfo(BasicSerializeMessage* sm, const PointToPointMessageInfo& message_info)
    : m_serialize_message(sm)
    , m_message_info(message_info)
    {}

   public:

    BasicSerializeMessage* m_serialize_message = nullptr;
    PointToPointMessageInfo m_message_info;
    bool m_is_probe_done = false;
  };

 public:

  explicit SerializeMessageList(IMessagePassingMng* mpm);

 public:

  void addMessage(ISerializeMessage* msg) override;
  void processPendingMessages() override;
  Integer waitMessages(eWaitType wait_type) override;
  Ref<ISerializeMessage>
  createAndAddMessage(MessageRank destination, ePointToPointMessageType type) override;

  void setAllowAnyRankReceive(bool v) { m_allow_any_rank_receive = v; }

 private:

  IMessagePassingMng* m_message_passing_mng = nullptr;
  SerializeMessageContainer m_messages_to_process;
  Ref<IRequestList> m_request_list;
  SerializeMessageContainer m_messages_serialize;
  SerializeMessageContainer m_remaining_serialize_messages;
  UniqueArray<ProbeInfo> m_messages_to_probe;
  bool m_allow_any_rank_receive = true;
  TimeMetricAction m_message_passing_phase;

  Integer _waitMessages(eWaitType wait_type);
  void _addMessage(BasicSerializeMessage* sm, const PointToPointMessageInfo& message_info);
  PointToPointMessageInfo buildMessageInfo(ISerializeMessage* sm);
  void _doProbe();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::MessagePassing::internal

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
