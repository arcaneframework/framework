// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
/*---------------------------------------------------------------------------*/
/* MPMessage.h                                                 (C) 2000-2020 */
/*                                                                           */
/* Message.                                                                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_MESSAGEPASSING_MESSAGE_H
#define ARCCORE_MESSAGEPASSING_MESSAGE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing/MessagePassingGlobal.h"

#include <cstddef>
#include <iosfwd>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore::MessagePassing
{
/*!
 * \brief Message.
 *
 * Ces informations sont utilisées pour récupérer les informations suite à un
 * appel à mpMessageProbe().
 */
class ARCCORE_MESSAGEPASSING_EXPORT MPMessage
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

  MPMessage()
  : m_return_value(0)
  {
    m_type = T_Null;
    m_message = null_message;
  }

  MPMessage(int return_value,void* amessage)
  : m_return_value(return_value)
  {
    m_type = T_Ptr;
    m_message.v = amessage;
  }

  MPMessage(int return_value,const void* amessage)
  : m_return_value(return_value)
  {
    m_type = T_Ptr;
    m_message.cv = amessage;
  }

  MPMessage(int return_value,int amessage)
  : m_return_value(return_value)
  {
    m_type = T_Int;
    m_message.i = amessage;
  }

  MPMessage(int return_value,long amessage)
  : m_return_value(return_value)
  {
    m_type = T_Long;
    m_message.l = amessage;
  }

  MPMessage(int return_value,std::size_t amessage)
  : m_return_value(return_value)
  {
    m_type = T_SizeT;
    m_message.st = amessage;
  }

  MPMessage(const MPMessage& rhs)
  : m_return_value(rhs.m_return_value), m_type(rhs.m_type)
  {
    m_message.cv = rhs.m_message.cv;
  }

  const MPMessage& operator=(const MPMessage& rhs)
  {
    m_return_value = rhs.m_return_value;
    m_type = rhs.m_type;
    m_message.cv = rhs.m_message.cv;
    return (*this);
  }

 public:

  template<typename T>
  operator const T*() const { return (const T*)m_message.cv; }
  template<typename T>
  operator T*() const { return (T*)m_message.v; }
  operator int() const { return m_message.i; }
  operator long() const { return m_message.l; }
  operator size_t() const { return m_message.st; }

 public:

  //int returnValue() const { return m_return_value; }
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

  static void setNullMessage(MPMessage r) { null_message = r.m_message; }

  void reset() {
    m_message = null_message;
  }

  void print(std::ostream& o) const;

 private:

  int m_return_value;
  int m_type;
  _Message m_message;

  static _Message null_message;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline std::ostream&
operator<<(std::ostream& o,const MPMessage& pmessage)
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

