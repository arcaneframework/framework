// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ISerializeMessageList.h                                     (C) 2000-2025 */
/*                                                                           */
/* Interface for a serialization message list.                               */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_MESSAGEPASSING_ISERIALIZEMESSAGEMESSAGELIST_H
#define ARCCORE_MESSAGEPASSING_ISERIALIZEMESSAGEMESSAGELIST_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing/MessagePassingGlobal.h"
#include "arccore/base/RefDeclarations.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MessagePassing
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Interface for a serialization message list.
 *
 * Instances of this class are generally created via the
 * method mpCreateSerializeMessageListRef().
 *
 * \code
 * using namespace Arccore;
 * IMessagePassingMng* mpm = ...;
 * Ref<ISerializeMessagList> serialize_list(mpCreateSerializeMessageListRef(mpm));
 * // Adds two messages to receive from ranks 1 and 3.
 * Ref<ISerializeMessage> message1(serialize_list->createAndAddMessage(MessageRank(1),MsgReceive));
 * Ref<ISerializeMessage> message2(serialize_list->createAndAddMessage(MessageRank(3),MsgReceive));
 *
 * // Waits until messages are finished
 * // In WaitSome or TestSome mode, it is possible to know if
 * // a message is finished by calling the ISerializeMessage::finished() method
 * serialize_list->wait(WaitAll);
 *
 * // Retrieves the serialization data of the first message
 * ISerializer* sb1 = message1.serializer();
 * sb1->setMode(ISerializer::ModeGet);
 * UniqueArray<Int32> int32_array;
 * sb1->getArray(int32_array);
 * ...
 * \endcode
 */
class ARCCORE_MESSAGEPASSING_EXPORT ISerializeMessageList
{
 public:

  virtual ~ISerializeMessageList() {} //!< Frees resources.

 public:

  /*!
   * \brief Adds a message to the list.
   *
   * The message is not posted until processPendingMessages()
   * has been called. The user retains ownership of the message, which
   * must not be destroyed until it is finished.
   */
  virtual void addMessage(ISerializeMessage* msg) = 0;

  /*!
   * \brief Sends the messages in the list that have not yet been sent.
   *
   * This method sends the messages added via addMessage() that have not
   * yet been sent. It is generally not necessary to call this
   * method because it is done automatically when calling waitMessages().
   */
  virtual void processPendingMessages() = 0;

  /*!
   * \brief Waits until the messages have finished execution.
   * 
   * The waiting type is specified by \a wt.
   *
   * It is then possible to test if a message is finished
   * via the ISerializeMessage::isFinished() method. This class does not keep
   * any reference to finished messages, which can therefore be
   * destroyed by the user as soon as they are no longer needed.
   *
   * \return the number of completely executed messages or (-1) if
   * they have all been executed.
   */
  virtual Integer waitMessages(eWaitType wt) = 0;

  /*!
   * \brief Creates and adds a serialization message.
   *
   * The message can be a send or receive message.
   *
   * If the message is a receive message (MsgReceive), it is possible to
   * specify a null rank to indicate that one wishes to receive from anyone.
   *
   * This method calls addMessage() to automatically add the
   * created message to the list of messages. The instance does not keep a reference
   * to the created message, which must not be destroyed until it is
   * finished.
   */
  virtual Ref<ISerializeMessage>
  createAndAddMessage(MessageRank destination,
                      ePointToPointMessageType type) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::MessagePassing

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
