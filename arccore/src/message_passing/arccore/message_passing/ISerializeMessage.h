// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ISerializeMessage.h                                         (C) 2000-2025 */
/*                                                                           */
/* Interface for a serialization message between subdomains.                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_MESSAGEPASSING_ISERIALIZEMESSAGE_H
#define ARCCORE_MESSAGEPASSING_ISERIALIZEMESSAGE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing/MessagePassingGlobal.h"
#include "arccore/serialize/SerializeGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MessagePassing
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Interface for a serialization message between IMessagePassingMng.
 *
 * A serialization message consists of a series of bytes sent
 * from a source rank() to a destination rank().
 * If isSend() is true, source() sends to destination(),
 * otherwise it is the reverse.
 * If it is a receive message, the serializer() is allocated
 * and filled automatically. For parallelism to work correctly,
 * a send message must correspond to a receive message
 * (sent by the destination rank).
 *
 * The message can be non-blocking. A message can be destroyed
 * when its finished() property is true.
 */
class ARCCORE_MESSAGEPASSING_EXPORT ISerializeMessage
{
 public:

  enum eMessageType
  {
    MT_Send,
    MT_Recv,
    MT_Broadcast
  };
 //! Sending/receiving strategy
  enum class eStrategy
  {
    //! Default strategy.
    Default,
    /*!
     * \brief Strategy using a single message if possible.
     *
     * This assumes using the mpProbe() function
     * to know the message size
     * before posting the reception.
     */
    OneMessage
  };

  virtual ~ISerializeMessage() = default; //!< Releases resources.

 public:

  //! \a true if it should send, \a false if it should receive
  virtual bool isSend() const =0;

  //! Message type
  virtual eMessageType messageType() const =0;

  /*!
   * \brief Destination rank (if isSend() is true) or sender.
   *
   * In the case of a reception, it is possible to specify any
   * rank by specifying A_NULL_RANK.
   */
  ARCCORE_DEPRECATED_2020("Use destination() instead")
  virtual Int32 destRank() const =0;

  /*!
   * \brief Destination rank (if isSend() is true) or sender.
   *
   * In the case of a reception, the rank can be null to indicate
   * that you wish to receive from anyone.
   * rank by specifying A_NULL_RANK.
   */
  virtual MessageRank destination() const =0;

  /*!
   * \brief Message sender rank
   * See also destRank() for interpretation based on the value of isSend()
   */
  ARCCORE_DEPRECATED_2020("Use source() instead")
  virtual Int32 origRank() const =0;

  /*!
   * \brief Message sender rank
   *
   * See also destination() for interpretation based on the value of isSend()
   */
  virtual MessageRank source() const =0;

  //! Serializer
  virtual ISerializer* serializer() =0;

  //! \a true if the message is finished
  virtual bool finished() const =0;

  /*!
   * \internal
   * \brief Sets the 'finished' state of the message.
   */
  virtual void setFinished(bool v) =0;

  /*!
   * \internal
   */
  ARCCORE_DEPRECATED_2020("Use setInternalTag() instead")
  virtual void setTag(Int32 tag) =0;

  /*!
   * \internal
   * \brief Sets an internal tag for the message.
   *
   * This tag is useful if multiple messages need to be sent/received
   * to the same origin/destination pair.
   *
   * This method is internal to %Arccore.
   */
  virtual void setInternalTag(MessageTag tag) =0;

  /*!
   * \internal
   * \brief Internal tag of the message.
   */
  ARCCORE_DEPRECATED_2020("Use internalTag() instead")
  virtual Int32 tag() const =0;

  /*!
   * \internal
   * \brief Internal tag of the message.
   */
  virtual MessageTag internalTag() const =0;

  /*!
   * \internal
   * \brief Message identifier
   */
  virtual MessageId _internalMessageId() const =0;

  /*!
   * \brief Sets the sending/receiving strategy.
   *
   * The strategy used must be the same for the sent message
   * and the receive message, otherwise the behavior is undefined.
   */
  virtual void setStrategy(eStrategy strategy) =0;

  //! Strategy used for sends/receives
  virtual eStrategy strategy() const =0;

  /*!
   * \brief Indicates if the message has already been processed.
   *
   * If the message has already been processed, it is not possible to change certain
   * characteristics (such as the strategy or the tag)
   */
  virtual bool isProcessed() const =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore::MessagePassing

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
