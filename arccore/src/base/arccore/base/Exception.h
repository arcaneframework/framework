// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Exception.h                                                 (C) 2000-2025 */
/*                                                                           */
/* Base class for an exception.                                              */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_EXCEPTION_H
#define ARCCORE_BASE_EXCEPTION_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/String.h"
#include "arccore/base/StackTrace.h"
// We don't explicitly need this .h but it is simpler
// to have it to easily throw exceptions with traces
#include "arccore/base/TraceInfo.h"

#include <exception>
#include <atomic>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup Core
 * \brief Base class for an exception.
 *
 * Exceptions are managed by the C++ <tt>try</tt> and <tt>catch</tt>
 * code. All exceptions thrown in the code <strong>must</strong>
 * derive from this class.
 *
 * An exception can be collective. This means it will be thrown
 * by all processors. In this case, it is possible to display only a
 * single message and possibly stop the execution cleanly.
 */
class ARCCORE_BASE_EXPORT Exception
: public std::exception
{
 public:

  Exception& operator=(const Exception&) = delete; //PURE

 public:

  /*!
   * Constructs an exception of name \a name and
   * sent from the function \a where.
   */
  Exception(const String& name,const String& where);
  /*!
   * Constructs an exception of name \a name and
   * sent from the function \a awhere.
   */
  Exception(const String& name, const TraceInfo& where);
  /*!
   * Constructs an exception of name \a name,
   * sent from the function \a awhere and with the message \a message.
   */
  Exception(const String& name, const String& awhere, const String& message);
  /*!
   * Constructs an exception of name \a name,
   * sent from the function \a where and with the message \a message.
   */
  Exception(const String& name, const TraceInfo& trace, const String& message);
  /*!
   * Constructs an exception of name \a name and
   * sent from the function \a where.
   */
  Exception(const String& name,const String& where,const StackTrace& stack_trace);
  /*!
   * Constructs an exception of name \a name and
   * sent from the function \a where.
   */
  Exception(const String& name,const TraceInfo& where,const StackTrace& stack_trace);
  /*!
   * Constructs an exception of name \a name,
   * sent from the function \a where and with the message \a message.
   */
  Exception(const String& name,const String& where,
            const String& message,const StackTrace& stack_trace);
  /*!
   * Constructs an exception of name \a name,
   * sent from the function \a where and with the message \a message.
   */
  Exception(const String& name,const TraceInfo& trace,
            const String& message,const StackTrace& stack_trace);
  //! Copy constructor.
  Exception(const Exception&);

  //! Releases resources
  ~Exception() ARCCORE_NOEXCEPT override;

 public:
 
  virtual void write(std::ostream& o) const;

  //! True if it is a collective error (concerns all processors)
  bool isCollective() const { return m_is_collective; }

  //! Sets the collective state of the expression
  void setCollective(bool v) { m_is_collective = v; }

  //! Sets the additional information
  void setAdditionalInfo(const String& v) { m_additional_info = v; }

  //! Returns the additional information
  const String& additionalInfo() const { return m_additional_info; }

  //! Call stack at the moment of the exception (requires a stacktrace service)
  const StackTrace& stackTrace() const { return m_stack_trace; }

  //! Call stack at the moment of the exception (requires a stacktrace service)
  const String& stackTraceString() const { return m_stack_trace.toString(); }

  //! Indicates if there are pending exceptions
  static bool hasPendingException();

  //! Exception message
  const String& message() const { return m_message; }

  //! Location of the exception
  const String& where() const { return m_where; }

  //! Name of the exception
  const String& name() const { return m_name; }

  //! Output operator
  friend ARCCORE_BASE_EXPORT std::ostream&
  operator<<(std::ostream& o, const Exception& ex);

 public:

  //! \internal
  static void staticInit();

 protected:

  /*!
   * \brief Explains the cause of the exception in the stream \a o.
   *
   * This method allows adding additional information
   * to the exception message.
   */
  virtual void explain(std::ostream& o) const;

  //! Sets the exception message
  void setMessage(const String& msg)
  {
    m_message = msg;
  }

 private:

  String m_name;
  String m_where;
  StackTrace m_stack_trace;
  String m_message;
  String m_additional_info;
  bool m_is_collective = false;

  void _setStackTrace();
  void _setWhere(const TraceInfo& where);
  void _checkExplainAndPause();

 private:

  static std::atomic<Int32> m_nb_pending_exception;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
