/*---------------------------------------------------------------------------*/
/* Exception.cc                                                (C) 2000-2018 */
/*                                                                           */
/* Classes gérant les exceptions.                                            */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/Exception.h"
#if ARCCORE_NEED_IMPLEMENTATION
#include "arccore/base/IStackTraceService.h"
#include "arccore/base/ISymbolizerService.h"
#include "arccore/base/PlatformUtils.h"
#endif
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
#if ARCCORE_NEED_IMPLEMENTATION

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
#if ARCCORE_NEED_IMPLEMENTATION

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
#if ARCCORE_NEED_IMPLEMENTATION

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
#if ARCCORE_NEED_IMPLEMENTATION
  IStackTraceService* stack_service = platform::getStackTraceService();
  // Evite les appels trop recursifs qui peuvent se produire si on lève des
  // exceptions dans l'appel à stackTrace().
  if (stack_service && m_nb_pending_exception.value()<=2){
    m_stack_trace = stack_service->stackTrace();
  }
#endif
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

