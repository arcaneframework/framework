// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MessageId.h                                                 (C) 2000-2025 */
/*                                                                           */
/* Identifiant d'un message point à point.                                   */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_MESSAGEPASSING_MESSAGEID_H
#define ARCCORE_MESSAGEPASSING_MESSAGEID_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing/MessageTag.h"
#include "arccore/message_passing/MessageRank.h"
#include "arccore/message_passing/MessageSourceInfo.h"

#include <cstddef>
#include <iosfwd>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MessagePassing
{
/*!
 * \brief MessageId.
 *
 * Ces informations sont utilisées pour récupérer les informations d'un
 * message suite à un appel à mpProbe(). L'instance retournée peut-être
 * utilisée pour faire une réception via mpReceive().
 *
 * Une fois l'appel à mpProbe() effectué, il est possible de récupérer les
 * informations sur la source du message via sourceInfo().
 *
 * Avec MPI, cette classe encapsule le type MPI_Message.
 */
class ARCCORE_MESSAGEPASSING_EXPORT MessageId
{
  union _Message
  {
    int i;
    long l;
    std::size_t st;
    void* v;
    const void* cv;
  };

  enum Type
  {
    T_Int,
    T_Long,
    T_SizeT,
    T_Ptr,
    T_Null
  };

 public:

  using SourceInfo = MessageSourceInfo;

  MessageId() : m_message(null_message){}

  MessageId(MessageSourceInfo source_info,void* amessage)
  : m_source_info(source_info)
  {
    m_type = T_Ptr;
    m_message.v = amessage;
  }

  MessageId(MessageSourceInfo source_info,const void* amessage)
  : m_source_info(source_info)
  {
    m_type = T_Ptr;
    m_message.cv = amessage;
  }

  MessageId(MessageSourceInfo source_info,int amessage)
  : m_source_info(source_info)
  {
    m_type = T_Int;
    m_message.i = amessage;
  }

  MessageId(MessageSourceInfo source_info,long amessage)
  : m_source_info(source_info)
  {
    m_type = T_Long;
    m_message.l = amessage;
  }

  MessageId(MessageSourceInfo source_info,std::size_t amessage)
  : m_source_info(source_info)
  {
    m_type = T_SizeT;
    m_message.st = amessage;
  }

  MessageId(const MessageId& rhs)
  : m_source_info(rhs.m_source_info), m_type(rhs.m_type)
  {
    m_message.cv = rhs.m_message.cv;
  }

  const MessageId& operator=(const MessageId& rhs)
  {
    m_source_info = rhs.m_source_info;
    m_type = rhs.m_type;
    m_message.cv = rhs.m_message.cv;
    return (*this);
  }

 public:

  template<typename T>
  explicit operator const T*() const { return (const T*)m_message.cv; }
  template<typename T>
  explicit operator T*() const { return (T*)m_message.v; }
  explicit operator int() const { return m_message.i; }
  explicit operator long() const { return m_message.l; }
  explicit operator size_t() const { return m_message.st; }

 public:

  //int returnValue() const { return m_source_info; }
  bool isValid() const
  {
    if (m_type==T_Null)
      return false;
    if (m_type==T_Int)
      return m_message.i!=null_message.i;
    if (m_type==T_Long)
      return m_message.l!=null_message.l;
    if (m_type==T_SizeT)
      return m_message.st!=null_message.st;
    return m_message.cv!=null_message.cv;
  }
  void* messageAsVoidPtr() const { return m_message.v; }

  static void setNullMessage(MessageId r) { null_message = r.m_message; }

  void reset()
  {
    m_message = null_message;
  }

  void print(std::ostream& o) const;

  //! Informations sur la source du message;
  MessageSourceInfo sourceInfo() const { return m_source_info; }

  //! Positionne les informations sur la source du message;
  void setSourceInfo(MessageSourceInfo si) { m_source_info = si; }

 private:

  MessageSourceInfo m_source_info;
  int m_type = T_Null;
  _Message m_message;
  static _Message null_message;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline std::ostream&
operator<<(std::ostream& o,const MessageId& pmessage)
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

