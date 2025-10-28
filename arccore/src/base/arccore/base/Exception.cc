// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Exception.cc                                                (C) 2000-2025 */
/*                                                                           */
/* Classes gérant les exceptions.                                            */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/Exception.h"
#include "arccore/base/IStackTraceService.h"
#include "arccore/base/PlatformUtils.h"
#include "arccore/base/TraceInfo.h"

#include <iostream>

#ifndef ARCCORE_OS_WIN32
#include <unistd.h>
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// TODO: Vérifier si la valeur de construction est bien 0
std::atomic<Int32> Exception::m_nb_pending_exception;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{
  // Si vrai, affiche les informations de l'exception dans les appels aux
  // constructeurs. Cela permet d'avoir le message dans le cas où une exception
  // lève une autre exception (ce qui appelle directement std::terminate et on
  // ne peut pas la récupérer).
  bool global_explain_in_constructor = false;
  // Si vrai, se met en pause dans le constructeur pour attendre de brancher
  // un débugger
  bool global_pause_in_constructor = false;
} // namespace

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCCORE_BASE_EXPORT void
arccoreSetPauseOnException(bool v)
{
  global_pause_in_constructor = v;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCCORE_BASE_EXPORT void
arccoreCallExplainInExceptionConstructor(bool v)
{
  global_explain_in_constructor = v;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Exception::
Exception(const String& aname,const String& awhere)
: m_name(aname)
, m_where(awhere)
{
  ++m_nb_pending_exception;
  _setStackTrace();
  _checkExplainAndPause();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Exception::
Exception(const String& aname,const String& awhere,const String& amessage)
: m_name(aname)
, m_where(awhere)
, m_message(amessage)
{
  ++m_nb_pending_exception;
  _setStackTrace();
  _checkExplainAndPause();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Exception::
Exception(const String& aname,const String& awhere,
          const StackTrace& stack_trace)
: m_name(aname)
, m_where(awhere)
, m_stack_trace(stack_trace)
{
  ++m_nb_pending_exception;
  _checkExplainAndPause();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Exception::
Exception(const String& aname,const String& awhere,
          const String& amessage,const StackTrace& stack_trace)
: m_name(aname)
, m_where(awhere)
, m_stack_trace(stack_trace)
, m_message(amessage)
{
  ++m_nb_pending_exception;
  _checkExplainAndPause();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Exception::
Exception(const String& aname,const TraceInfo& awhere)
: m_name(aname)
{
  ++m_nb_pending_exception;
  _setWhere(awhere);
  _setStackTrace();
  _checkExplainAndPause();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Exception::
Exception(const String& aname,const TraceInfo& awhere,const String& amessage)
: m_name(aname)
, m_message(amessage)
, m_is_collective(false)
{
  ++m_nb_pending_exception;
  _setWhere(awhere);
  _setStackTrace();
  _checkExplainAndPause();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Exception::
Exception(const String& aname,const TraceInfo& awhere,
          const StackTrace& stack_trace)
: m_name(aname)
, m_stack_trace(stack_trace)
, m_is_collective(false)
{
  ++m_nb_pending_exception;
  _setWhere(awhere);
  _checkExplainAndPause();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Exception::
Exception(const String& aname,const TraceInfo& awhere,
          const String& amessage,const StackTrace& stack_trace)
: m_name(aname)
, m_stack_trace(stack_trace)
, m_message(amessage)
, m_is_collective(false)
{
  ++m_nb_pending_exception;
  _setWhere(awhere);
  _checkExplainAndPause();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Exception::
Exception(const Exception& from)
: m_name(from.m_name)
, m_where(from.m_where)
, m_stack_trace(from.m_stack_trace)
, m_message(from.m_message)
, m_is_collective(from.m_is_collective)
{
  ++m_nb_pending_exception;
  _checkExplainAndPause();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Exception::
~Exception() ARCCORE_NOEXCEPT
{
  --m_nb_pending_exception;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Exception::
_setStackTrace()
{
  IStackTraceService* stack_service = Platform::getStackTraceService();
  if (stack_service){
    m_stack_trace = stack_service->stackTrace();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Exception::
_setWhere(const TraceInfo& awhere)
{
  m_where = awhere.name();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Exception::
write(std::ostream& o) const
{
  o << "Exception!\n\nThrown in: '" << m_where
    << "'\nType: '" << m_name << "'\n\n";
  if (!m_message.null())
    o << "Message: " << m_message << '\n';
  this->explain(o);
  String st = m_stack_trace.toString();
  if (!st.null()){
    o << "\nCall stack:\n";
    o << st << '\n';
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool Exception::
hasPendingException()
{
  //TODO utiliser test atomic
  return m_nb_pending_exception.load()!=0;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Exception::
staticInit()
{
  m_nb_pending_exception = 0;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Exception::
explain(std::ostream&) const
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

std::ostream&
operator<<(std::ostream& o,const Exception& ex)
{
  ex.write(o);
  return o;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Exception::
_checkExplainAndPause()
{
  if (global_explain_in_constructor){
    std::cerr << "** Exception:" << (*this) << "\n";
  }

  if (global_pause_in_constructor){
    std::cerr << "** Exception: Debug mode activated. Execution paused.\n";
    std::cerr << "** Exception: To find the location of the exception, start the debugger\n";
    // Utilise format pour être sur que le message ne sera pas affiché en plusieurs
    // morceaux
    std::cerr << String::format("** Exception: using process number {0} on host '{1}'\n",
                                Platform::getProcessId(),Platform::getHostName());

#ifndef ARCCORE_OS_WIN32
    ::pause();
#endif
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

