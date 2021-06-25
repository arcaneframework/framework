// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2020 IFPEN-CEA
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Exception.cc                                                (C) 2000-2018 */
/*                                                                           */
/* Classes gérant les exceptions.                                            */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/Exception.h"
#include "arccore/base/IStackTraceService.h"
#include "arccore/base/PlatformUtils.h"
#include "arccore/base/TraceInfo.h"

#include <iostream>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// TODO: Vérifier si la valeur de construction est bien 0
std::atomic<Int32> Exception::m_nb_pending_exception;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// #define ARCCORE_DEBUG_EXCEPTION

Exception::
Exception(const String& aname,const String& awhere)
: m_name(aname)
, m_where(awhere)
, m_is_collective(false)
{
  ++m_nb_pending_exception;
  _setStackTrace();
#if defined(ARCCORE_DEBUG_EXCEPTION)
  explain(cerr);
  cerr << "** Exception: Debug mode activated. Execution paused.\n";
  cerr << "** Exception: To find the location of the exception, start the debugger\n";
  cerr << "** Exception: using process number " << Platform::getProcessId() << '\n';
  cerr << "** Exception: on host " << String(Platform::getHostName()) << ".\n";
#ifndef ARCCORE_OS_WIN32
  ::pause();
#endif
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Exception::
Exception(const String& aname,const String& awhere,const String& amessage)
: m_name(aname)
, m_where(awhere)
, m_message(amessage)
, m_is_collective(false)
{
  ++m_nb_pending_exception;
  _setStackTrace();

#if defined(ARCCORE_DEBUG_EXCEPTION)
  explain(cerr);
  cerr << "** Exception: Debug mode activated. Execution paused.\n";
  cerr << "** Exception: To find the location of the exception, start the debugger\n";
  cerr << "** Exception: using process number " << platform::getProcessId() << '\n';
  cerr << "** Exception: on host " << String(platform::getHostName()) << ".\n";
#ifndef ARCCORE_OS_WIN32
  ::pause();
#endif
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Exception::
Exception(const String& aname,const String& awhere,
          const StackTrace& stack_trace)
: m_name(aname)
, m_where(awhere)
, m_stack_trace(stack_trace)
, m_is_collective(false)
{
  ++m_nb_pending_exception;
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
, m_is_collective(false)
{
  ++m_nb_pending_exception;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Exception::
Exception(const String& aname,const TraceInfo& awhere)
: m_name(aname)
, m_is_collective(false)
{
  ++m_nb_pending_exception;
  _setWhere(awhere);
  _setStackTrace();
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
#if defined(ARCCORE_DEBUG_EXCEPTION)
  explain(cerr);
  cerr << "** Exception: Debug mode activated. Execution paused.\n";
  cerr << "** Exception: To find the location of the exception, start the debugger\n";
  cerr << "** Exception: using process number " << platform::getProcessId() << '\n';
  cerr << "** Exception: on host " << String(platform::getHostName()) << ".\n";
#ifndef ARCCORE_OS_WIN32
  ::pause();
#endif
#endif
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

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

