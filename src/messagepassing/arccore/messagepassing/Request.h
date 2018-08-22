/*---------------------------------------------------------------------------*/
/* Request.h                                                   (C) 2000-2018 */
/*                                                                           */
/* Requête d'un message.                                                     */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_MESSAGEPASSING_REQUEST_H
#define ARCCORE_MESSAGEPASSING_REQUEST_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/messagepassing/MessagePassingGlobal.h"

#include <cstddef>
#include <iosfwd>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore
{

namespace MessagePassing
{

/*!
 * \brief Requête d'un message.
 *
 * Ces informations sont utilisées pour les messages non bloquants.
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
  : m_return_value(0)
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

  Request(const Request& rhs)
  : m_return_value(rhs.m_return_value), m_type(rhs.m_type)
  {
    m_request.cv = rhs.m_request.cv;
  }

  const Request& operator=(const Request& rhs)
  {
    m_return_value = rhs.m_return_value;
    m_type = rhs.m_type;
    m_request.cv = rhs.m_request.cv;
    return (*this);
  }

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
  //ARCCORE_DEPRECATED long request() const { return m_request.l; }
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

  void reset() {
    m_request = null_request;
  }

  void print(std::ostream& o) const;

 private:

  int m_return_value;
  int m_type;
  _Request m_request;

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

} // End namespace MessagePassing

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

