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

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore::MessagePassing
{
/*!
 * \brief PointToPointMessage.
 *
 * Informations pour les messages point à point.
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

  explicit PointToPointMessageInfo(Int32 rank)
  : m_rank(rank), m_tag(100), m_type(Type::T_RankTag){}
  PointToPointMessageInfo(Int32 rank,Int32 tag)
  : m_rank(rank), m_tag(tag), m_type(Type::T_RankTag){}
  PointToPointMessageInfo(MessageId message_id)
  : m_message_id(message_id), m_type(Type::T_MessageId){}

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
  Int32 rank() const { return m_rank; }
  Int32 tag() const { return m_tag; }
  void print(std::ostream& o) const;

 public:

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
  Int32 m_tag = 100;
  MessageId m_message_id;
  Type m_type = Type::T_Null;
  bool m_is_blocking = false;
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

