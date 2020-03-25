// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
/*---------------------------------------------------------------------------*/
/* Request.h                                                   (C) 2000-2020 */
/*                                                                           */
/* Requête d'un message.                                                     */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_MESSAGEPASSING_REQUEST_H
#define ARCCORE_MESSAGEPASSING_REQUEST_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing/MessagePassingGlobal.h"
#include "arccore/base/Ref.h"

#include <cstddef>
#include <iosfwd>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore::MessagePassing
{
class Request;

class ARCCORE_MESSAGEPASSING_EXPORT ISubRequest
{
 public:
  virtual ~ISubRequest() = default;
 public:
  virtual Request executeOnCompletion() =0;
};

/*!
 * \brief Requête d'un message.
 *
 * Ces informations sont utilisées pour les messages non bloquants.
 *
 * \note Le type exact de la requête dépend de l'implémentation. On encapsule
 * cela dans une union.
 */
class ARCCORE_MESSAGEPASSING_EXPORT Request
{
  union _Request
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

  Request()
  {
    m_type = T_Null;
    m_request = null_request;
  }

  Request(int return_value,void* arequest)
  : m_return_value(return_value)
  {
    m_type = T_Ptr;
    m_request.v = arequest;
  }

  Request(int return_value,const void* arequest)
  : m_return_value(return_value)
  {
    m_type = T_Ptr;
    m_request.cv = arequest;
  }

  Request(int return_value,int arequest)
  : m_return_value(return_value)
  {
    m_type = T_Int;
    m_request.i = arequest;
  }

  Request(int return_value,long arequest)
  : m_return_value(return_value)
  {
    m_type = T_Long;
    m_request.l = arequest;
  }

  Request(int return_value,std::size_t arequest)
  : m_return_value(return_value)
  {
    m_type = T_SizeT;
    m_request.st = arequest;
  }

  Request& operator=(const Request& rhs) = default;

 public:

  template<typename T>
  operator const T*() const { return (const T*)m_request.cv; }
  template<typename T>
  operator T*() const { return (T*)m_request.v; }
  operator int() const { return m_request.i; }
  operator long() const { return m_request.l; }
  operator size_t() const { return m_request.st; }

 public:

  int returnValue() const { return m_return_value; }
  bool isValid() const
  {
    if (m_type==T_Null)
      return false;
    if (m_type==T_Int)
      return m_request.i!=null_request.i;
    if (m_type==T_Long)
      return m_request.l!=null_request.l;
    if (m_type==T_SizeT)
      return m_request.st!=null_request.st;
    return m_request.cv!=null_request.cv;
  }
  void* requestAsVoidPtr() const { return m_request.v; }

  static void setNullRequest(Request r) { null_request = r.m_request; }

  void reset()
  {
    m_request = null_request;
    m_sub_request.reset();
  }
  Ref<ISubRequest> subRequest() const { return m_sub_request; }
  bool hasSubRequest() const { return !m_sub_request.isNull(); }
  void setSubRequest(Ref<ISubRequest> s) { m_sub_request = s; }
  void print(std::ostream& o) const;

 private:

  int m_return_value = 0;
  int m_type = T_Null;
  _Request m_request;
  Ref<ISubRequest> m_sub_request;
  static _Request null_request;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline std::ostream&
operator<<(std::ostream& o,const Request& prequest)
{
  prequest.print(o);
  return o;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore::MessagePassing

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

