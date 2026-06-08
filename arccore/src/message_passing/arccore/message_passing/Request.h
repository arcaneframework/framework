// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Request.h                                                   (C) 2000-2025 */
/*                                                                           */
/* Message request.                                                          */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_MESSAGEPASSING_REQUEST_H
#define ARCCORE_MESSAGEPASSING_REQUEST_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/message_passing/MessagePassingGlobal.h"
#include "arccore/message_passing/MessageRank.h"
#include "arccore/message_passing/MessageTag.h"
#include "arccore/base/Ref.h"

#include <cstddef>
#include <iosfwd>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MessagePassing
{

/*!
 * \internal
 * \brief Sub-request of a request
 */
class ARCCORE_MESSAGEPASSING_EXPORT ISubRequest
{
 public:

  virtual ~ISubRequest() = default;

 public:

  //! Callback called when the associated request is finished
  virtual Request executeOnCompletion(const SubRequestCompletionInfo&) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Interface for a request creator.
 */
class ARCCORE_MESSAGEPASSING_EXPORT IRequestCreator
{
 public:
  virtual ~IRequestCreator() = default;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Message request.
 *
 * This information is used for non-blocking messages. A
 * non-null request is associated with an \a IMessagePassingMng.
 *
 * This class allows generically storing a request without
 * knowing its exact type (for example, MPI_Request with the MPI standard).
 * For this, a union is used. To ensure that an instance of this
 * class is created with the correct parameters, it is preferable to use a
 * specialization (for example, the MpiRequest class).
 *
 * A request can be associated with a sub-request (ISubRequest) whose
 * method ISubRequest::executeOnCompletion() will be executed
 * when the request is satisfied. This allows for the automatic generation of other
 * requests.
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

 public:

  ARCCORE_DEPRECATED_2020("Use overload with IRequestCreator pointer")
  Request(int return_value,void* arequest)
  : m_return_value(return_value)
  {
    m_type = T_Ptr;
    m_request.v = arequest;
  }

  ARCCORE_DEPRECATED_2020("Use overload with IRequestCreator pointer")
  Request(int return_value,const void* arequest)
  : m_return_value(return_value)
  {
    m_type = T_Ptr;
    m_request.cv = arequest;
  }

  ARCCORE_DEPRECATED_2020("Use overload with IRequestCreator pointer")
  Request(int return_value,int arequest)
  : m_return_value(return_value)
  {
    m_type = T_Int;
    m_request.i = arequest;
  }

  ARCCORE_DEPRECATED_2020("Use overload with IRequestCreator pointer")
  Request(int return_value,long arequest)
  : m_return_value(return_value)
  {
    m_type = T_Long;
    m_request.l = arequest;
  }

  ARCCORE_DEPRECATED_2020("Use overload with IRequestCreator pointer")
  Request(int return_value,std::size_t arequest)
  : m_return_value(return_value)
  {
    m_type = T_SizeT;
    m_request.st = arequest;
  }

 public:

  Request(int return_value,IRequestCreator* creator,void* arequest)
  : m_return_value(return_value), m_creator(creator)
  {
    m_type = T_Ptr;
    m_request.v = arequest;
  }

  Request(int return_value,IRequestCreator* creator,const void* arequest)
  : m_return_value(return_value), m_creator(creator)
  {
    m_type = T_Ptr;
    m_request.cv = arequest;
  }

  Request(int return_value,IRequestCreator* creator,int arequest)
  : m_return_value(return_value), m_creator(creator)
  {
    m_type = T_Int;
    m_request.i = arequest;
  }

  Request(int return_value,IRequestCreator* creator,long arequest)
  : m_return_value(return_value), m_creator(creator)
  {
    m_type = T_Long;
    m_request.l = arequest;
  }

  Request(int return_value,IRequestCreator* creator,std::size_t arequest)
  : m_return_value(return_value), m_creator(creator)
  {
    m_type = T_SizeT;
    m_request.st = arequest;
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
  bool isValid() const
  {
    if (m_type==T_Null)
      return false;
    // If the request type is different from the null request type,
    // then the request is considered valid.
    if (m_type!=null_request_type)
      return true;
    if (m_type==T_Int)
      return m_request.i!=null_request.i;
    if (m_type==T_Long)
      return m_request.l!=null_request.l;
    if (m_type==T_SizeT)
      return m_request.st!=null_request.st;
    if (m_type==T_Ptr)
      return m_request.cv!=null_request.cv;
    return false;
  }
  void* requestAsVoidPtr() const { return m_request.v; }

  static void setNullRequest(Request r)
  {
    null_request = r.m_request;
    null_request_type = r.m_type;
  }

  void reset()
  {
    m_request = null_request;
    m_sub_request.reset();
    m_type = T_Null;
  }
  Ref<ISubRequest> subRequest() const { return m_sub_request; }
  bool hasSubRequest() const { return !m_sub_request.isNull(); }
  void setSubRequest(Ref<ISubRequest> s) { m_sub_request = s; }

  //! Request creator
  IRequestCreator* creator() const { return m_creator; }

  void print(std::ostream& o) const;

  friend inline std::ostream& operator<<(std::ostream& o,const Request& prequest)
  {
    prequest.print(o);
    return o;
  }

  //! \internal
  Int32 _type() const { return m_type; }

 private:

  int m_return_value = 0;
  int m_type = T_Null;
  _Request m_request;
  Ref<ISubRequest> m_sub_request;
  IRequestCreator* m_creator = nullptr;
  static _Request null_request;
  static int null_request_type;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore::MessagePassing

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
