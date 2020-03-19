// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
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

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore::MessagePassing
{
/*!
 * \brief Informations pour envoyer/recevoir un message point à point.
 *
 * Il existe deux manière de construire une instance de cette classe:
 * - en donnant un rang destinataire, avec éventuellement un tag
 * - via un MessageId obtenu lors d'un appel à mpProbe(). Dans ce dernier
 * cas, l'instance ne peut être utilisée qu'en réception via mpReceive().
 *
 * Il est possible de spécifier si le message sera bloquant lors de la
 * construction ou via l'appel à setBlocking().
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

  PointToPointMessageInfo(){}
  explicit PointToPointMessageInfo(Int32 rank)
  : m_rank(rank), m_type(Type::T_RankTag){}
  PointToPointMessageInfo(Int32 rank,eBlockingType blocking_type)
  : m_rank(rank), m_is_blocking(blocking_type==Blocking), m_type(Type::T_RankTag){}
  PointToPointMessageInfo(Int32 rank,MessageTag tag)
  : m_rank(rank), m_tag(tag), m_type(Type::T_RankTag){}
  PointToPointMessageInfo(Int32 rank,MessageTag tag,eBlockingType blocking_type)
  : m_rank(rank), m_tag(tag), m_is_blocking(blocking_type==Blocking), m_type(Type::T_RankTag){}
  explicit PointToPointMessageInfo(MessageId message_id)
  : m_message_id(message_id), m_type(Type::T_MessageId){}
  PointToPointMessageInfo(MessageId message_id,eBlockingType blocking_type)
  : m_message_id(message_id), m_is_blocking(blocking_type==Blocking), m_type(Type::T_MessageId){}

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
  //! Rang de la destination/origine du message
  Int32 rank() const { return m_rank; }
  //! Tag du message
  MessageTag tag() const { return m_tag; }
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

  Int32 m_rank = A_NULL_RANK;
  MessageTag m_tag = MessageTag::defaultTag();
  MessageId m_message_id;
  bool m_is_blocking = true;
  Type m_type = Type::T_Null;
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

