// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* PointToPointMessageInfo.h                                   (C) 2000-2020 */
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

namespace Arccore::MessagePassing
{
/*!
 * \brief Informations pour envoyer/recevoir un message point à point.
 *
 * Il existe deux manières de construire une instance de cette classe:
 *
 * 1. en donnant un couple (rang destinataire,tag). Le tag est optionnel
 *    et s'il n'est pas spécifié, sa valeur sera celle de MessageTag::defaultTag().
 *    De même, la source est optionnelle.
 * 2. via un MessageId obtenu lors d'un appel à mpProbe(). Dans ce dernier
 *    cas, l'instance ne peut être utilisée qu'en réception via mpReceive().
 *
 * Il est possible de spécifier si le message sera bloquant lors de la
 * construction ou via l'appel à setBlocking(). Par défaut un message
 * est créé en mode bloquant.
 *
 * La source (sourceRank()) du message est l'emetteur et la
 * destination (destinationRank() le récepteur. La source est en général
 * positionnée par l'implémentation de IMessagePassingMng.
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
  PointToPointMessageInfo(){}

  //! Message bloquant avec tag par défaut et ayant pour destination \a rank
  explicit PointToPointMessageInfo(MessageRank dest_rank)
  : m_dest_rank(dest_rank), m_type(Type::T_RankTag){}

  //! Message bloquant avec tag par défaut et ayant pour source \a source_rank et destination \a dest_rank
  PointToPointMessageInfo(MessageRank source_rank,MessageRank dest_rank)
  : m_source_rank(source_rank), m_dest_rank(dest_rank), m_type(Type::T_RankTag){}

  //! Message avec tag par défaut, ayant destination \a dest_rank et mode bloquant \a blocking_type
  PointToPointMessageInfo(MessageRank dest_rank,eBlockingType blocking_type)
  : m_dest_rank(dest_rank), m_is_blocking(blocking_type==Blocking), m_type(Type::T_RankTag){}

  /*!
   * \brief Message avec tag par défaut et ayant pour source \a source_rank,
   * destination \a dest_rank et mode bloquant \a blocking_type
   */
  PointToPointMessageInfo(MessageRank source_rank,MessageRank dest_rank,eBlockingType blocking_type)
  : m_source_rank(source_rank), m_dest_rank(dest_rank)
  , m_is_blocking(blocking_type==Blocking), m_type(Type::T_RankTag){}

  //! Message bloquant avec tag \a tag et ayant pour destination \a rank
  PointToPointMessageInfo(MessageRank dest_rank,MessageTag tag)
  : m_dest_rank(dest_rank), m_tag(tag), m_type(Type::T_RankTag){}

  //! Message bloquant avec tag \a tag, ayant pour source \a source_rank, et ayant pour destination \a rank
  PointToPointMessageInfo(MessageRank source_rank,MessageRank dest_rank,MessageTag tag)
  : m_source_rank(source_rank), m_dest_rank(dest_rank), m_tag(tag), m_type(Type::T_RankTag){}

  //! Message avec tag \a tag, ayant pour destination \a dest_rank et mode bloquant \a blocking_type
  PointToPointMessageInfo(MessageRank dest_rank,MessageTag tag,eBlockingType blocking_type)
  : m_dest_rank(dest_rank), m_tag(tag), m_is_blocking(blocking_type==Blocking), m_type(Type::T_RankTag){}

  /*!
   * \brief Message avec tag \a tag, ayant pour source \a source_rank,
   * pour destination \a dest_rank et mode bloquant \a blocking_type.
   */
  PointToPointMessageInfo(MessageRank source_rank,MessageRank dest_rank,MessageTag tag,eBlockingType blocking_type)
  : m_source_rank(source_rank), m_dest_rank(dest_rank), m_tag(tag)
  , m_is_blocking(blocking_type==Blocking), m_type(Type::T_RankTag){}

  //! Message bloquant associé à \a message_id
  explicit PointToPointMessageInfo(MessageId message_id)
  : m_message_id(message_id), m_type(Type::T_MessageId)
  {
    _setInfosFromMessageId();
  }

  //! Message associé à \a message_id avec le mode bloquant \a blocking_type
  PointToPointMessageInfo(MessageId message_id,eBlockingType blocking_type)
  : m_message_id(message_id), m_is_blocking(blocking_type==Blocking), m_type(Type::T_MessageId)
  {
    _setInfosFromMessageId();
  }

 public:
  PointToPointMessageInfo& setBlocking(bool is_blocking)
  {
    m_is_blocking = is_blocking;
    return (*this);
  }
  //! Indique si le message est bloquant.
  bool isBlocking() const { return m_is_blocking; }
  //! Vrai si l'instance a été créée avec un MessageId. Dans ce cas messageId() est valide
  bool isMessageId() const { return m_type==Type::T_MessageId; }
  //! Vrai si l'instance a été créée avec un couple (rank,tag). Dans ce cas rank() et tag() sont valides.
  bool isRankTag() const { return m_type==Type::T_RankTag; }
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
  void setRankTag(MessageRank rank,MessageTag tag)
  {
    m_type = Type::T_RankTag;
    // Attention à bien appeler les méthodes pour mettre à jour
    // les valeurs associées de `m_message_id`
    setDestinationRank(rank);
    setTag(tag);
  }
  //! Rang de la destination du message
  MessageRank destinationRank() const { return m_dest_rank; }
  //! Positionne le rang de la destination du message
  void setDestinationRank(MessageRank rank)
  {
    m_dest_rank = rank;
    MessageId::SourceInfo si = m_message_id.sourceInfo();
    si.setRank(rank);
    m_message_id.setSourceInfo(si);
  }
  //! Rang d'origine du message
  MessageRank sourceRank() const { return m_source_rank; }
  //! Positionne le rang d'origine du message
  void setSourceRank(MessageRank rank) { m_source_rank = rank; }
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

 public:

  // Indique si le message est valide (i.e: il a été initialisé avec un message valide)
  bool isValid() const
  {
    if (m_type==Type::T_Null)
      return false;
    if (m_type==Type::T_MessageId)
      return m_message_id.isValid();
    if (m_type==Type::T_RankTag)
      return true;
    return false;
  }

 private:

  MessageRank m_source_rank;
  MessageRank m_dest_rank;
  MessageTag m_tag = MessageTag::defaultTag();
  MessageId m_message_id;
  bool m_is_blocking = true;
  Type m_type = Type::T_Null;

  void _setInfosFromMessageId()
  {
    m_dest_rank = m_message_id.sourceInfo().rank();
    m_tag = m_message_id.sourceInfo().tag();
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline std::ostream&
operator<<(std::ostream& o,const PointToPointMessageInfo& pmessage)
{
  pmessage.print(o);
  return o;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore::MessagePassing

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

