// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MessageSourceInfo.h                                         (C) 2000-2025 */
/*                                                                           */
/* Information about the source of a message.                                */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_MESSAGEPASSING_MESSAGESOURCEINFO_H
#define ARCCORE_MESSAGEPASSING_MESSAGESOURCEINFO_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing/MessageTag.h"
#include "arccore/message_passing/MessageRank.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MessagePassing
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Information about the source of a message.
 *
 * This information is used to retrieve message information following a call
 * to mpProbe() or mpLegacyProbe().
 * The returned instance can be used to perform a reception via mpReceive().
 */
class ARCCORE_MESSAGEPASSING_EXPORT MessageSourceInfo
{
 public:

  //! Creates a source corresponding to no message (isValid()==false)
  MessageSourceInfo() = default;

  /*!
   * \brief Creates a source corresponding to rank \a rank and tag \a tag.
   *
   * If \a rank.isNull() or tag.isNull(), then isValid() will be \a false.
   */
  MessageSourceInfo(MessageRank rank, MessageTag tag, Int64 size)
  : m_rank(rank)
  , m_tag(tag)
  , m_size(size)
  {}

 public:

  //! Source rank
  MessageRank rank() const { return m_rank; }

  //! Sets the source rank
  void setRank(MessageRank rank) { m_rank = rank; }

  //! Message tag
  MessageTag tag() const { return m_tag; }

  //! Sets the message tag
  void setTag(MessageTag tag) { m_tag = tag; }

  //! Message size
  Int64 size() const { return m_size; }

  //! Sets the message size
  void setSize(Int64 size) { m_size = size; }

  //! Indicates if the source is valid
  bool isValid() const { return !m_rank.isNull() && !m_tag.isNull(); }

 private:

  MessageRank m_rank;
  MessageTag m_tag;
  Int64 m_size = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::MessagePassing

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
