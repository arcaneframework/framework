// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* BasicSerialize.h                                            (C) 2000-2025 */
/*                                                                           */
/* Message using a BasicSerializer.                                          */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_MESSAGEPASSING_BASICSERIALIZEMESSAGE_H
#define ARCCORE_MESSAGEPASSING_BASICSERIALIZEMESSAGE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/serialize/BasicSerializer.h"
#include "arccore/message_passing/ISerializeMessage.h"
#include "arccore/message_passing/MessageRank.h"
#include "arccore/message_passing/MessageTag.h"
#include "arccore/message_passing/MessageId.h"
#include "arccore/base/RefDeclarations.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MessagePassing::internal
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Serializing message using a BasicSerializer.
 *
 * This class is internal to %Arccore and should not be used
 * directly. If you want to create an instance of ISerializeMessage,
 * you must create a list of messages via
 * ISerializeMessageList::createMessage();
 *
 * A message consists of a series of bytes sent or received from a rank
 * (source()) to another (destination()). If isSend() is true,
 * source() sends bytes to destination(), otherwise source()
 * receives bytes from destination().
 * If it is a reception message, the serializer() is allocated
 * and filled automatically.
 */
class ARCCORE_MESSAGEPASSING_EXPORT BasicSerializeMessage
: public ISerializeMessage
{
 public:

  static const Int32 DEFAULT_SERIALIZE_TAG_VALUE = 101;

  //! Default tag for serialization messages
  static MessageTag defaultTag() { return MessageTag(DEFAULT_SERIALIZE_TAG_VALUE); }

 protected:
 public:

  ~BasicSerializeMessage() override;
  BasicSerializeMessage& operator=(const BasicSerializeMessage&) = delete;
  BasicSerializeMessage(const BasicSerializeMessage&) = delete;

 protected:

  BasicSerializeMessage(MessageRank orig_rank, MessageRank dest_rank,
                        ePointToPointMessageType mtype);
  BasicSerializeMessage(MessageRank orig_rank, MessageRank dest_rank,
                        MessageTag tag, ePointToPointMessageType mtype);
  BasicSerializeMessage(MessageRank orig_rank, MessageRank dest_rank,
                        ePointToPointMessageType type,
                        BasicSerializer* serializer);
  BasicSerializeMessage(MessageRank orig_rank, MessageRank dest_rank,
                        MessageTag tag, ePointToPointMessageType type,
                        BasicSerializer* serializer);

  BasicSerializeMessage(MessageRank orig_rank, MessageId message_id,
                        BasicSerializer* serializer);

 public:

  static Ref<ISerializeMessage>
  create(MessageRank source, MessageRank destination, ePointToPointMessageType type);
  static Ref<ISerializeMessage>
  create(MessageRank source, MessageRank destination, MessageTag tag,
         ePointToPointMessageType type);
  static Ref<ISerializeMessage>
  create(MessageRank source, MessageId message_id);

 public:

  bool isSend() const override { return m_is_send; }
  eMessageType messageType() const override { return m_old_message_type; }
  Int32 destRank() const override { return m_dest_rank.value(); }
  Int32 origRank() const override { return m_orig_rank.value(); }
  MessageRank destination() const override { return m_dest_rank; }
  MessageRank source() const override { return m_orig_rank; }
  ISerializer* serializer() override { return m_buffer; }
  bool finished() const override { return m_finished; }
  void setFinished(bool v) override { m_finished = v; }
  void setTag(Int32 tag) override { m_tag = MessageTag(tag); }
  Int32 tag() const override { return m_tag.value(); }
  void setInternalTag(MessageTag tag) override { m_tag = tag; }
  MessageTag internalTag() const override { return m_tag; }
  MessageId _internalMessageId() const override { return m_message_id; }
  void setStrategy(eStrategy strategy) override;
  eStrategy strategy() const override { return m_strategy; }
  bool isProcessed() const override { return m_is_processed; }

 public:

  BasicSerializer& buffer() { return *m_buffer; }
  BasicSerializer* trueSerializer() const { return m_buffer; }
  Int32 messageNumber() const { return m_message_number; }
  void setMessageNumber(Int32 v) { m_message_number = v; }
  void setIsProcessed(bool v) { m_is_processed = v; }
  ePointToPointMessageType _internalMessageType() const { return m_message_type; }

 protected:

  static ePointToPointMessageType _toP2PType(eMessageType mtype);
  static eMessageType _toMessageType(ePointToPointMessageType mtype);

 private:

  MessageRank m_orig_rank; //!< Rank of the request sender
  MessageRank m_dest_rank; //!< Rank of the message recipient
  MessageTag m_tag = defaultTag();
  eMessageType m_old_message_type; //!< Message type (obsolete)
  ePointToPointMessageType m_message_type; //!< Message type
  eStrategy m_strategy = eStrategy::Default;
  bool m_is_send; //!< \c true if sending, \c false if receiving
  BasicSerializer* m_buffer = nullptr; //!< Buffer containing the info
  bool m_finished = false; //!< \c true if message finished
  MessageId m_message_id; //!< Associated MessageId (can be null)
  Int32 m_message_number = 0; //! Message number when using multiple messages
  bool m_is_processed = false;

 private:

  void _init();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::MessagePassing::internal

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
