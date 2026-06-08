// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* PointToPointMessageInfo.h                                   (C) 2000-2025 */
/*                                                                           */
/* Information for point-to-point messages.                                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_MESSAGEPASSING_POINTTOPOINTINFO_H
#define ARCCORE_MESSAGEPASSING_POINTTOPOINTINFO_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing/MessageId.h"
#include "arccore/message_passing/MessageTag.h"
#include "arccore/message_passing/MessageRank.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MessagePassing
{

/*!
 * \brief Information for sending/receiving a point-to-point message.
 *
 * There are two ways to construct an instance of this class:
 *
 * 1. By providing a pair (destination rank, tag). The tag is optional
 *    and if not specified, its value will be that of MessageTag::defaultTag().
 * 2. Via a MessageId obtained during a call to mpProbe(). In this latter
 *    case, the instance can only be used for reception via mpReceive().
 *
 * It is possible to specify whether the message will be blocking during
 * construction or via the call to setBlocking(). By default, a message
 * is created in blocking mode.
 *
 * The sender (emiterRank()) of the message is the sender and the
 * destination (destinationRank()) is the receiver. For a send message (mpSend()),
 * destinationRank() is therefore the rank of the one who will receive the message. For a
 * receive message (mpReceive()), destinationRank() is the rank of the one from whom
 * we wish to receive the message or A_NULL_RANK if we wish to receive from
 * anyone.
 *
 * \note The sender is generally positioned by the IMessagePassingMng implementation
 * because it corresponds to the rank of the one posting the message.
 * The user never needs to specify it.
 */
class ARCCORE_MESSAGEPASSING_EXPORT PointToPointMessageInfo
{
 public:

  enum class Type
  {
    T_RankTag,
    T_MessageId,
    T_Null
  };

 public:

  //! Null message.
  PointToPointMessageInfo() {}

  //! Blocking message with default tag and destination rank \a rank
  explicit PointToPointMessageInfo(MessageRank dest_rank)
  : m_destination_rank(dest_rank)
  , m_type(Type::T_RankTag)
  {}

  //! Message with default tag, destination \a dest_rank, and blocking
  //mode \a blocking_type
  PointToPointMessageInfo(MessageRank dest_rank, eBlockingType blocking_type)
  : m_destination_rank(dest_rank)
  , m_is_blocking(blocking_type == Blocking)
  , m_type(Type::T_RankTag)
  {}

  //! Blocking message with tag \a tag and destination \a rank
  PointToPointMessageInfo(MessageRank dest_rank, MessageTag tag)
  : m_destination_rank(dest_rank)
  , m_tag(tag)
  , m_type(Type::T_RankTag)
  {}

  //! Message with tag \a tag, destination \a dest_rank, and blocking
  //mode \a blocking_type
  PointToPointMessageInfo(MessageRank dest_rank, MessageTag tag, eBlockingType blocking_type)
  : m_destination_rank(dest_rank)
  , m_tag(tag)
  , m_is_blocking(blocking_type == Blocking)
  , m_type(Type::T_RankTag)
  {}

  //! Blocking message associated with \a message_id
  explicit PointToPointMessageInfo(MessageId message_id)
  : m_message_id(message_id)
  , m_type(Type::T_MessageId)
  {
    _setInfosFromMessageId();
  }

  //! Message associated with \a message_id with blocking mode \a blocking_type
  PointToPointMessageInfo(MessageId message_id, eBlockingType blocking_type)
  : m_message_id(message_id)
  , m_is_blocking(blocking_type == Blocking)
  , m_type(Type::T_MessageId)
  {
    _setInfosFromMessageId();
  }

 public:

  /*!
   * \brief Message with default tag and source \a emiter_rank,
   * destination \a dest_rank, and blocking mode \a blocking_type
   */
  ARCCORE_DEPRECATED_REASON("Y2022: This constructor is internal to Arcane and deprecated")
  PointToPointMessageInfo(MessageRank emiter_rank, MessageRank dest_rank, eBlockingType blocking_type)
  : m_emiter_rank(emiter_rank)
  , m_destination_rank(dest_rank)
  , m_is_blocking(blocking_type == Blocking)
  , m_type(Type::T_RankTag)
  {}

  //! Blocking message with tag \a tag, source \a emiter_rank, and destination \a rank
  ARCCORE_DEPRECATED_REASON("Y2022: This constructor is internal to Arcane and deprecated")
  PointToPointMessageInfo(MessageRank emiter_rank, MessageRank dest_rank, MessageTag tag)
  : m_emiter_rank(emiter_rank)
  , m_destination_rank(dest_rank)
  , m_tag(tag)
  , m_type(Type::T_RankTag)
  {}

  //! Blocking message with default tag, source \a emiter_rank, and destination \a dest_rank
  ARCCORE_DEPRECATED_REASON("Y2022: This constructor is internal to Arcane and deprecated")
  PointToPointMessageInfo(MessageRank emiter_rank, MessageRank dest_rank)
  : m_emiter_rank(emiter_rank)
  , m_destination_rank(dest_rank)
  , m_type(Type::T_RankTag)
  {}

 public:

  /*!
   * \internal
   * \brief Message with tag \a tag, source \a emiter_rank,
   * destination \a dest_rank, and blocking mode \a blocking_type.
   */
  PointToPointMessageInfo(MessageRank emiter_rank, MessageRank dest_rank, MessageTag tag, eBlockingType blocking_type)
  : m_emiter_rank(emiter_rank)
  , m_destination_rank(dest_rank)
  , m_tag(tag)
  , m_is_blocking(blocking_type == Blocking)
  , m_type(Type::T_RankTag)
  {}

 public:

  PointToPointMessageInfo& setBlocking(bool is_blocking)
  {
    m_is_blocking = is_blocking;
    return (*this);
  }
  //! Indicates if the message is blocking.
  bool isBlocking() const { return m_is_blocking; }
  //! True if the instance was created with a MessageId. In this case messageId() is valid
  bool isMessageId() const { return m_type == Type::T_MessageId; }
  //! True if the instance was created with a pair (rank,tag). In this case rank() and tag() are valid.
  bool isRankTag() const { return m_type == Type::T_RankTag; }
  //! Message identifier
  MessageId messageId() const { return m_message_id; }
  //! Positions the message identifier and changes the message type
  void setMessageId(const MessageId& message_id)
  {
    m_type = Type::T_MessageId;
    m_message_id = message_id;
    _setInfosFromMessageId();
  }
  //! Positions the message destination rank and tag and changes the message type
  void setRankTag(MessageRank rank, MessageTag tag)
  {
    m_type = Type::T_RankTag;
    // Attention to properly call the methods to update
    // the associated values of `m_message_id`
    setDestinationRank(rank);
    setTag(tag);
  }
  //! Message destination rank
  MessageRank destinationRank() const { return m_destination_rank; }
  //! Positions the message destination rank
  void setDestinationRank(MessageRank rank)
  {
    m_destination_rank = rank;
    MessageId::SourceInfo si = m_message_id.sourceInfo();
    si.setRank(rank);
    m_message_id.setSourceInfo(si);
  }

  //! Message sender rank
  MessageRank emiterRank() const { return m_emiter_rank; }
  //! Positions the message sender rank
  void setEmiterRank(MessageRank rank) { m_emiter_rank = rank; }

  //! Message tag
  MessageTag tag() const { return m_tag; }
  //! Positions the message tag
  void setTag(MessageTag tag)
  {
    m_tag = tag;
    MessageId::SourceInfo si = m_message_id.sourceInfo();
    si.setTag(tag);
    m_message_id.setSourceInfo(si);
  }
  //! Prints the message
  void print(std::ostream& o) const;

  friend std::ostream& operator<<(std::ostream& o, const PointToPointMessageInfo& pmessage)
  {
    pmessage.print(o);
    return o;
  }

 public:

  // Indicates if the message is valid (i.e.: it was initialized with a valid message)
  bool isValid() const
  {
    if (m_type == Type::T_Null)
      return false;
    if (m_type == Type::T_MessageId)
      return m_message_id.isValid();
    if (m_type == Type::T_RankTag)
      return true;
    return false;
  }

 public:

  //! Message origin rank
  ARCCORE_DEPRECATED_REASON("Y2022: Use emiterRank() instead")
  MessageRank sourceRank() const { return m_emiter_rank; }

  //! Positions the message origin rank
  ARCCORE_DEPRECATED_REASON("Y2022: Use setEmiterRank() instead")
  void setSourceRank(MessageRank rank) { m_emiter_rank = rank; }

 private:

  MessageRank m_emiter_rank;
  MessageRank m_destination_rank;
  MessageTag m_tag = MessageTag::defaultTag();
  MessageId m_message_id;
  bool m_is_blocking = true;
  Type m_type = Type::T_Null;

  void _setInfosFromMessageId()
  {
    m_destination_rank = m_message_id.sourceInfo().rank();
    m_tag = m_message_id.sourceInfo().tag();
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore::MessagePassing

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
