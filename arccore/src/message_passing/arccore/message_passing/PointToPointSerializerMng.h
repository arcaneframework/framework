// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* PointToPointSerializerMng.h                                 (C) 2000-2025 */
/*                                                                           */
/* Point-to-point communications using 'ISerializer'.                        */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_MESSAGEPASSING_POINTOTPOINTSERIALIZERMNG_H
#define ARCCORE_MESSAGEPASSING_POINTOTPOINTSERIALIZERMNG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing/ISerializeMessage.h"
#include "arccore/base/RefDeclarations.h"

#include <functional>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MessagePassing
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ISerializeMessage;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Point-to-point communications using 'ISerializer'.
 */
class ARCCORE_MESSAGEPASSING_EXPORT PointToPointSerializerMng
{
  class Impl;

 public:

  PointToPointSerializerMng(IMessagePassingMng* mpm);
  ~PointToPointSerializerMng();

 public:
  PointToPointSerializerMng(const PointToPointSerializerMng&) = delete;
  PointToPointSerializerMng& operator=(const PointToPointSerializerMng&) = delete;
 public:

  //! Associated message manager
  IMessagePassingMng* messagePassingMng() const;

  /*!
   * \brief Sends the messages from the list that have not yet been processed.
   *
   * It is generally not necessary to call this
   * method because it is done automatically when calling waitMessages().
   */
  void processPendingMessages();

  /*!
   * \brief Waits for the messages to finish execution.
   * 
   * The wait type is specified by \a wt.
   *
   * \return the number of messages completely executed or (-1) if they
   * all have been.
   */
  Integer waitMessages(eWaitType wt,std::function<void(ISerializeMessage*)> functor);

  //! Indicates if there are remaining messages that have not yet finished.
  bool hasMessages() const;

  /*!
   * \brief Creates a receiving serialization message
   *
   * \a sender_rank is the rank of the sender of the corresponding message.
   * It is possible to specify a null rank to indicate that one
   * wishes to receive from anyone.
   */
  Ref<ISerializeMessage> addReceiveMessage(MessageRank sender_rank);

  /*!
   * \brief Creates a receiving serialization message
   *
   * \a sender_rank is the rank of the sender of the corresponding message.
   * It is possible to specify a null rank to indicate that one
   * wishes to receive from anyone.
   */
  Ref<ISerializeMessage> addReceiveMessage(MessageId message_id);

  /*!
   * \brief Creates a sending serialization message.
   */
  Ref<ISerializeMessage> addSendMessage(MessageRank receiver_rank);

  /*!
   * \brief Default tag used for messages.
   *
   * This method can only be called if there are no messages
   * currently in progress (hasMessages()==false). All ranks of messagePassingMng()
   * must use the same tag.
   */
  void setDefaultTag(MessageTag default_tag);

  /*!
   * \brief Strategy used for messages.
   *
   * This method can only be called if there are no messages
   * currently in progress (hasMessages()==false). All ranks of messagePassingMng()
   * must use the same strategy.
   */
  void setStrategy(ISerializeMessage::eStrategy strategy);

 private:

  Impl* m_p;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore::MessagePassing

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
