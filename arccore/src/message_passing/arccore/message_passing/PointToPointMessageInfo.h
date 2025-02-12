// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* PointToPointMessageInfo.h                                   (C) 2000-2025 */
/*                                                                           */
/* Informations pour les messages point à point.                             */
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
 * \brief Informations pour envoyer/recevoir un message point à point.
 *
 * Il existe deux manières de construire une instance de cette classe:
 *
 * 1. en donnant un couple (rang destinataire,tag). Le tag est optionnel
 *    et s'il n'est pas spécifié, sa valeur sera celle de MessageTag::defaultTag().
 * 2. via un MessageId obtenu lors d'un appel à mpProbe(). Dans ce dernier
 *    cas, l'instance ne peut alors être utilisée qu'en réception via mpReceive().
 *
 * Il est possible de spécifier si le message sera bloquant lors de la
 * construction ou via l'appel à setBlocking(). Par défaut un message
 * est créé en mode bloquant.
 *
 * L'émetteur (emiterRank()) du message est l'émetteur et la
 * destination (destinationRank() le récepteur. Pour un message d'envoi (mpSend()),
 * destinationRank() est donc le rang de celui qui va recevoir le message. Pour un
 * message de réception (mpReceive()), destinationRank() est le rang de celui dont
 * on souhaite recevoir le message ou A_NULL_RANK si on souhaite recevoir de
 * n'importe qui.
 *
 * \note L'émetteur est en général positionnée par l'implémentation de
 * IMessagePassingMng car il correspond au rang de celui qui poste le message.
 * L'utilisateur n'a donc jamais besoin de le spécifier.
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

  //! Message nul.
  PointToPointMessageInfo() {}

  //! Message bloquant avec tag par défaut et ayant pour destination \a rank
  explicit PointToPointMessageInfo(MessageRank dest_rank)
  : m_destination_rank(dest_rank)
  , m_type(Type::T_RankTag)
  {}

  //! Message avec tag par défaut, ayant destination \a dest_rank et mode bloquant \a blocking_type
  PointToPointMessageInfo(MessageRank dest_rank, eBlockingType blocking_type)
  : m_destination_rank(dest_rank)
  , m_is_blocking(blocking_type == Blocking)
  , m_type(Type::T_RankTag)
  {}

  //! Message bloquant avec tag \a tag et ayant pour destination \a rank
  PointToPointMessageInfo(MessageRank dest_rank, MessageTag tag)
  : m_destination_rank(dest_rank)
  , m_tag(tag)
  , m_type(Type::T_RankTag)
  {}

  //! Message avec tag \a tag, ayant pour destination \a dest_rank et mode bloquant \a blocking_type
  PointToPointMessageInfo(MessageRank dest_rank, MessageTag tag, eBlockingType blocking_type)
  : m_destination_rank(dest_rank)
  , m_tag(tag)
  , m_is_blocking(blocking_type == Blocking)
  , m_type(Type::T_RankTag)
  {}

  //! Message bloquant associé à \a message_id
  explicit PointToPointMessageInfo(MessageId message_id)
  : m_message_id(message_id)
  , m_type(Type::T_MessageId)
  {
    _setInfosFromMessageId();
  }

  //! Message associé à \a message_id avec le mode bloquant \a blocking_type
  PointToPointMessageInfo(MessageId message_id, eBlockingType blocking_type)
  : m_message_id(message_id)
  , m_is_blocking(blocking_type == Blocking)
  , m_type(Type::T_MessageId)
  {
    _setInfosFromMessageId();
  }

 public:

  /*!
   * \brief Message avec tag par défaut et ayant pour source \a emiter_rank,
   * destination \a dest_rank et mode bloquant \a blocking_type
   */
  ARCCORE_DEPRECATED_REASON("Y2022: This constructor is internal to Arcane and deprecated")
  PointToPointMessageInfo(MessageRank emiter_rank, MessageRank dest_rank, eBlockingType blocking_type)
  : m_emiter_rank(emiter_rank)
  , m_destination_rank(dest_rank)
  , m_is_blocking(blocking_type == Blocking)
  , m_type(Type::T_RankTag)
  {}

  //! Message bloquant avec tag \a tag, ayant pour source \a emiter_rank, et ayant pour destination \a rank
  ARCCORE_DEPRECATED_REASON("Y2022: This constructor is internal to Arcane and deprecated")
  PointToPointMessageInfo(MessageRank emiter_rank, MessageRank dest_rank, MessageTag tag)
  : m_emiter_rank(emiter_rank)
  , m_destination_rank(dest_rank)
  , m_tag(tag)
  , m_type(Type::T_RankTag)
  {}

  //! Message bloquant avec tag par défaut et ayant pour source \a emiter_rank et destination \a dest_rank
  ARCCORE_DEPRECATED_REASON("Y2022: This constructor is internal to Arcane and deprecated")
  PointToPointMessageInfo(MessageRank emiter_rank, MessageRank dest_rank)
  : m_emiter_rank(emiter_rank)
  , m_destination_rank(dest_rank)
  , m_type(Type::T_RankTag)
  {}

 public:

  /*!
   * \internal
   * \brief Message avec tag \a tag, ayant pour source \a emiter_rank,
   * pour destination \a dest_rank et mode bloquant \a blocking_type.
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
  //! Indique si le message est bloquant.
  bool isBlocking() const { return m_is_blocking; }
  //! Vrai si l'instance a été créée avec un MessageId. Dans ce cas messageId() est valide
  bool isMessageId() const { return m_type == Type::T_MessageId; }
  //! Vrai si l'instance a été créée avec un couple (rank,tag). Dans ce cas rank() et tag() sont valides.
  bool isRankTag() const { return m_type == Type::T_RankTag; }
  //! Identifiant du message
  MessageId messageId() const { return m_message_id; }
  //! Positionne l'identifiant du message et change le type du message
  void setMessageId(const MessageId& message_id)
  {
    m_type = Type::T_MessageId;
    m_message_id = message_id;
    _setInfosFromMessageId();
  }
  //! Positionne le rang destination et le tag du message et change le type du message
  void setRankTag(MessageRank rank, MessageTag tag)
  {
    m_type = Type::T_RankTag;
    // Attention à bien appeler les méthodes pour mettre à jour
    // les valeurs associées de `m_message_id`
    setDestinationRank(rank);
    setTag(tag);
  }
  //! Rang de la destination du message
  MessageRank destinationRank() const { return m_destination_rank; }
  //! Positionne le rang de la destination du message
  void setDestinationRank(MessageRank rank)
  {
    m_destination_rank = rank;
    MessageId::SourceInfo si = m_message_id.sourceInfo();
    si.setRank(rank);
    m_message_id.setSourceInfo(si);
  }

  //! Rang de l'émetteur du message
  MessageRank emiterRank() const { return m_emiter_rank; }
  //! Positionne le rang de l'émetteur du message
  void setEmiterRank(MessageRank rank) { m_emiter_rank = rank; }

  //! Tag du message
  MessageTag tag() const { return m_tag; }
  //! Positionne le tag du message
  void setTag(MessageTag tag)
  {
    m_tag = tag;
    MessageId::SourceInfo si = m_message_id.sourceInfo();
    si.setTag(tag);
    m_message_id.setSourceInfo(si);
  }
  //! Affiche le message
  void print(std::ostream& o) const;

  friend std::ostream& operator<<(std::ostream& o, const PointToPointMessageInfo& pmessage)
  {
    pmessage.print(o);
    return o;
  }

 public:

  // Indique si le message est valide (i.e: il a été initialisé avec un message valide)
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

  //! Rang d'origine du message
  ARCCORE_DEPRECATED_REASON("Y2022: Use emiterRank() instead")
  MessageRank sourceRank() const { return m_emiter_rank; }

  //! Positionne le rang d'origine du message
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

