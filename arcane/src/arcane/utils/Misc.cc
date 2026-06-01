// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Misc.cc                                                     (C) 2000-2025 */
/*                                                                           */
/* Various functions                                                         */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Iostream.h"
#include "arcane/utils/Convert.h"
#include "arcane/utils/StdHeader.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/SignalException.h"
#include "arcane/utils/TimeoutException.h"
#include "arcane/utils/ArithmeticException.h"
#include "arcane/utils/IStackTraceService.h"
#include "arcane/utils/TraceInfo.h"
#include "arcane/utils/OStringStream.h"
#include "arcane/utils/IThreadImplementation.h"
// Not used here but necessary to load symbols in the DLL
#include "arcane/utils/NullThreadMng.h"
#include "arcane/utils/CriticalSection.h"

#include <limits.h>
#include <math.h>
#ifndef ARCANE_OS_WIN32
#include <unistd.h>
#endif
#include <time.h>
#include <stdarg.h>
#include <stdlib.h>
#ifndef ARCANE_OS_WIN32
#include <sys/time.h>
#endif

#define USE_SIGACTION 1

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C" {
typedef void (*fSignalFunc)(int);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifdef ARCANE_CHECK
static bool global_arcane_is_check = true;
#else
static bool global_arcane_is_check = false;
#endif

extern "C++" ARCANE_UTILS_EXPORT bool arcaneIsCheck()
{
  return global_arcane_is_check;
}

extern "C++" ARCANE_UTILS_EXPORT void arcaneSetCheck(bool v)
{
  global_arcane_is_check = v;
}

extern "C++" ARCANE_UTILS_EXPORT bool arcaneIsDebug()
{
#ifdef ARCANE_DEBUG
  return true;
#else
  return false;
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_UTILS_EXPORT bool arcaneHasThread()
{
  return true;
}

extern "C++" ARCANE_UTILS_EXPORT void arcaneSetHasThread([[maybe_unused]] bool v)
{
  if (!v)
    std::cout << "WARNING: disabling thread via arcaneSetHasThread() is no longer available.\n";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_UTILS_EXPORT
Int64
arcaneCurrentThread()
{
  IThreadImplementation* ti = platform::getThreadImplementationService();
  if (ti)
    return ti->currentThread();
  return 0;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_UTILS_EXPORT void
arcaneSetPauseOnError(bool v)
{
  arccoreSetPauseOnError(v);
}

extern "C++" ARCANE_UTILS_EXPORT void
arcaneDebugPause(const char* msg)
{
  arccoreDebugPause(msg);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_UTILS_EXPORT void
arcaneNullPointerError(const void* ptr)
{
  ARCANE_UNUSED(ptr);
  cerr << "** FATAL: Trying to use a null pointer.\n";
  arcaneDebugPause("arcaneNullPointerError");
  throw FatalErrorException(A_FUNCINFO);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{
  void _doNoReferenceError(const void* ptr)
  {
    cerr << "** FATAL: Null reference.\n";
    cerr << "** FATAL: Trying to use an item not referenced.\n";
    cerr << "** FATAL: Item is located at memory address " << ptr << ".\n";
    arcaneDebugPause("arcaneNoReferenceError");
  }
} // namespace

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_UTILS_EXPORT void
arcaneNoReferenceError(const void* ptr)
{
  _doNoReferenceError(ptr);
  throw FatalErrorException(A_FUNCINFO);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_UTILS_EXPORT void
arcaneNoReferenceErrorCallTerminate(const void* ptr)
{
  _doNoReferenceError(ptr);
  std::terminate();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_UTILS_EXPORT void
arcanePrintf(const char* format, ...)
{
  // \n is written at the same time to avoid parasitic intermediate writes
  char buffer[256];
  va_list ap;
  va_start(ap, format);
  vsnprintf(buffer, 256, format, ap);
  va_end(ap);
  cerr << buffer << "\n";
  cout << "*E* " << buffer << "\n";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_UTILS_EXPORT void
arcaneObsolete(const char* file, const char* func, unsigned long line, const char* text)
{
  cerr << file << ':' << func << ':' << line << '\n';
  cerr << "usage of this function is deprecated";
  if (text)
    cerr << ": " << text;
  cerr << '\n';
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/// Function called when an assertion fails.
typedef void (*fDoAssert)(const char*, const char*, const char*, size_t);
/// Function called to indicate whether debug information should be displayed
typedef bool (*fCheckDebug)(unsigned int);

static fDoAssert g_do_assert_func = 0;
static fCheckDebug g_check_debug_func = 0;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * Displaying a failed assertion.
 */
ARCANE_UTILS_EXPORT void
_doAssert(const char* text, const char* file, const char* func, size_t line)
{
  if (g_do_assert_func)
    (*g_do_assert_func)(text, file, func, line);
  else {
    std::ostringstream ostr;
    ostr << text << ':' << file << ':' << func << ':' << line << ": ";
    throw FatalErrorException("Assert", ostr.str());
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/**
 * Checks if a debug message should be displayed.
 */
extern "C++" ARCANE_UTILS_EXPORT bool
_checkDebug(unsigned int val)
{
  if (g_check_debug_func)
    return (*g_check_debug_func)(val);
  return false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{
  static fSignalFunc g_signal_func = 0;
}

fSignalFunc
setSignalFunc(fSignalFunc func)
{
  fSignalFunc old = g_signal_func;
  g_signal_func = func;
  return old;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifdef ARCANE_OS_LINUX
#define STD_SIGNAL_TYPE 1
#endif

#if STD_SIGNAL_TYPE == 1
#include <sys/signal.h>
#include <signal.h>
#include <unistd.h>

//extern "C" void (*sigset (int sig, void (*disp)(int)))(int);

extern "C" void
_MiscSigFunc(int val)
{
  if (g_signal_func)
    (*g_signal_func)(val);
}

extern "C" void
_MiscSigactionFunc(int val, siginfo_t*, void*)
{
  if (g_signal_func)
    (*g_signal_func)(val);
}

#endif // STD_SIGNAL_TYPE == 1

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
namespace
{
#ifndef USE_SIGACTION
  fSignalFunc default_signal_func_sigsegv = 0;
  fSignalFunc default_signal_func_sigfpe = 0;
  fSignalFunc default_signal_func_sigbus = 0;
  fSignalFunc default_signal_func_sigalrm = 0;
  fSignalFunc default_signal_func_sigvtalrm = 0;
#endif
  bool global_already_in_signal = false;
} // namespace

extern "C++" ARCANE_UTILS_EXPORT void
arcaneRedirectSignals(fSignalFunc sig_func)
{
  setSignalFunc(sig_func);
#if STD_SIGNAL_TYPE == 1

#ifdef USE_SIGACTION
  struct sigaction sa;
  sa.sa_flags = SA_SIGINFO | SA_NODEFER;
  sigemptyset(&sa.sa_mask);
  sa.sa_sigaction = _MiscSigactionFunc;

  sigaction(SIGSEGV, &sa, nullptr); // Segmentation fault.
  sigaction(SIGFPE, &sa, nullptr); // Floating Exception.
  sigaction(SIGBUS, &sa, nullptr); // Bus Error.
  sigaction(SIGALRM, &sa, nullptr); // Signal alarm (ITIMER_REAL)
  sigaction(SIGVTALRM, &sa, nullptr); // Signal alarm (ITIMER_VIRTUAL)
#else
  default_signal_func_sigsegv = sigset(SIGSEGV, _MiscSigFunc); // Segmentation fault.
  default_signal_func_sigfpe = sigset(SIGFPE, _MiscSigFunc); // Floating Exception.
  default_signal_func_sigbus = sigset(SIGBUS, _MiscSigFunc); // Bus Error.
  //default_signal_func_sigsys  = sigset(SIGSYS ,_MiscSigFunc);  // Bad argument in system call.
  //default_signal_func_sigpipe = sigset(SIGPIPE,_MiscSigFunc);  // Pipe error.
  default_signal_func_sigalrm = sigset(SIGALRM, _MiscSigFunc); // Signal alarm (ITIMER_REAL)
  default_signal_func_sigvtalrm = sigset(SIGVTALRM, _MiscSigFunc); // Signal alarm (ITIMER_VIRTUAL)
#endif // USE_SIGACTION
#endif // STD_SIGNAL_TYPE == 1
}

extern "C++" ARCANE_UTILS_EXPORT void
arcaneCallDefaultSignal(int val)
{
#if STD_SIGNAL_TYPE == 1

  //fSignalFunc func = 0;
  SignalException::eSignalType signal_type = SignalException::ST_Unknown;

  switch (val) {
  case SIGSEGV:
    signal_type = SignalException::ST_SegmentationFault;
    break;
  case SIGFPE:
    signal_type = SignalException::ST_FloatingException;
    break;
  case SIGBUS:
    signal_type = SignalException::ST_BusError;
    break;
  case SIGALRM:
  case SIGVTALRM:
    signal_type = SignalException::ST_Alarm;
    break;
  }

  // cerr << "** SIGVAL " << val << ' ' << SIG_DFL
  // << ' ' << SIG_IGN << " is_in_signal?=" << global_already_in_signal << '\n';

  // If a new signal occurs while already in a handler, or
  // if it is a memory error signal (SIGBUS or SIGSEGV), we abort.
  if (global_already_in_signal)
    ::abort();
  global_already_in_signal = true;

  arcaneDebugPause("arcaneCallDefaultSignal");

  // To avoid certain problems when recovering the stack
  // which might cause an exception, we first retrieve the call
  // stack and pass it to the exception constructor.
  StackTrace stack_trace;
  IStackTraceService* stack_service = platform::getStackTraceService();
  if (stack_service) {
    stack_trace = stack_service->stackTrace();
    cerr << " Signal exception Stack: " << stack_trace.toString() << '\n';
  }
  else
    cerr << " No stack trace service\n";

  // If requested, display the call stack via the debugger. This allows for
  // more information (but it takes longer to execute)
  bool do_debug_stack = false;
  if (auto v = Convert::Type<Int32>::tryParseFromEnvironment("ARCANE_DUMP_DEBUGGER_STACK_IN_SIGNAL", true))
    do_debug_stack = (v.value() != 0);
  if (do_debug_stack) {
    std::cerr << "GBDStack pid=" << platform::getProcessId() << " stack=" << platform::getGDBStack() << "\n";
  }
  else
    std::cerr << "SignalCaught: You can dump full stacktrace of the process if environment "
                 "variable ARCANE_DUMP_DEBUGGER_STACK_IN_SIGNAL is set to 1\n";

  if (signal_type == SignalException::ST_Alarm) {
    global_already_in_signal = false;
    throw TimeoutException("Arcane.Signal", stack_trace);
  }
  else if (signal_type == SignalException::ST_FloatingException) {
    global_already_in_signal = false;
    cerr << "** THROW ARITHMETIC EXCEPTION\n";
    // Re-enable floating exceptions for the next time
    platform::enableFloatingException(true);
    throw ArithmeticException(A_FUNCINFO, stack_trace);
  }
  throw SignalException("Arcane.Signal", stack_trace, signal_type, val);
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifdef ARCANE_DEBUG
extern "C" void
_PureVirtual()
{
  cerr << "** pure virtual method called\n";
  //pause();
}
#endif

class HexaPrint
{
 public:

  HexaPrint(Real v)
  : m_v(v)
  {}

 public:

  void write(std::ostream& o) const
  {
    char* v = (char*)&m_v;
    o << m_v << " HEXA(";
    o << (int)v[0] << '-';
    o << (int)v[1] << '-';
    o << (int)v[2] << '-';
    o << (int)v[3] << '-';
    o << (int)v[4] << '-';
    o << (int)v[5] << '-';
    o << (int)v[6] << '-';
    o << (int)v[7] << ")";
  }
  Real m_v;
};

ARCANE_UTILS_EXPORT std::ostream&
operator<<(std::ostream& o, const HexaPrint& hp)
{
  hp.write(o);
  return o;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
