﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Misc.cc                                                     (C) 2000-2016 */
/*                                                                           */
/* Diverses fonctions                                                        */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/utils/Iostream.h"
#include "arcane/utils/ValueConvert.h"
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
// Pas utilise ici mais necessaire pour charger les symbols dans la DLL
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

//#include <fenv.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C"
{
  typedef void (*fSignalFunc)(int);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<> ARCANE_UTILS_EXPORT bool builtInGetValue(double& v,const String& s)
{
  const char* ptr = s.localstr();
#ifdef WIN32
  if(s=="infinity" || s=="inf")
  {
	v = std::numeric_limits<double>::infinity();
	return false;
  }
#endif
  char* ptr2 = 0;
  v = ::strtod(ptr,&ptr2);
  return (ptr2!=(ptr+s.length()));
}
template<> ARCANE_UTILS_EXPORT bool builtInGetValue(float& v,const String& s)
{
  double z = 0.;
  bool r = builtInGetValue(z,s);
  v = (float)z;
  return r;
}
template<> ARCANE_UTILS_EXPORT bool builtInGetValue(long& v,const String& s)
{
  const char* ptr = s.localstr();
  char* ptr2 = 0;
  v = ::strtol(ptr,&ptr2,0);
  return (ptr2!=(ptr+s.length()));
}
template<> ARCANE_UTILS_EXPORT bool builtInGetValue(int& v,const String& s)
{
  long z = 0;
  bool r = builtInGetValue(z,s);
  v = (int)z;
  return r;
}
template<> ARCANE_UTILS_EXPORT bool builtInGetValue(short& v,const String& s)
{
  long z = 0;
  bool r = builtInGetValue(z,s);
  v = (short)z;
  return r;
}
template<> ARCANE_UTILS_EXPORT bool builtInGetValue(unsigned long& v,const String& s)
{
  const char* ptr = s.localstr();
  char* ptr2 = 0;
  v = ::strtoul(ptr,&ptr2,0);
  return (ptr2!=(ptr+s.length()));
}
template<> ARCANE_UTILS_EXPORT bool builtInGetValue(unsigned int& v,const String& s)
{
  unsigned long z = 0;
  bool r = builtInGetValue(z,s);
  v = (unsigned int)z;
  return r;
}
template<> ARCANE_UTILS_EXPORT bool builtInGetValue(unsigned short& v,const String& s)
{
  unsigned long z = 0;
  bool r = builtInGetValue(z,s);
  v = (unsigned short)z;
  return r;
}
template<> ARCANE_UTILS_EXPORT bool builtInGetValue(long long& v,const String& s)
{
  const char* ptr = s.localstr();
  char* ptr2 = 0;
  v = ::strtoll(ptr,&ptr2,0);
  return (ptr2!=(ptr+s.length()));
}
template<> ARCANE_UTILS_EXPORT bool builtInGetValue(unsigned long long& v,const String& s)
{
  const char* ptr = s.localstr();
  char* ptr2 = 0;
  v = ::strtoull(ptr,&ptr2,0);
  return (ptr2!=(ptr+s.length()));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifdef ARCANE_CHECK
static bool global_arcane_is_check = true;
#else
static bool global_arcane_is_check = false;
#endif

extern "C++" ARCANE_UTILS_EXPORT
bool arcaneIsCheck()
{
  return global_arcane_is_check;
}

extern "C++" ARCANE_UTILS_EXPORT
void arcaneSetCheck(bool v)
{
  global_arcane_is_check = v;
}

extern "C++" ARCANE_UTILS_EXPORT
bool arcaneIsDebug()
{
#ifdef ARCANE_DEBUG
  return true;
#else
  return false;
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifdef ARCANE_OS_LINUX
#if defined(_REENTRANT)
static bool global_arcane_has_thread = true;
#else
static bool global_arcane_has_thread = false;
#endif
#endif

#ifdef ARCANE_OS_WIN32
static bool global_arcane_has_thread = true;
#endif

extern "C++" ARCANE_UTILS_EXPORT
bool arcaneHasThread()
{
  return global_arcane_has_thread;
}

extern "C++" ARCANE_UTILS_EXPORT
void arcaneSetHasThread(bool v)
{
  if (!v){
    global_arcane_has_thread = v;
    return;
  }
  // Si non compile avec '-pthread', on ne pourra jamais utiliser
  // les threads.
  //TODO: regarder le cas Win32 voir la macro est la meme
#if defined(_REENTRANT) || defined(ARCANE_OS_WIN32)
  global_arcane_has_thread = v;
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_UTILS_EXPORT
Int64 arcaneCurrentThread()
{
  if (arcaneHasThread()){
    IThreadImplementation* ti = platform::getThreadImplementationService();
    if (ti)
      return ti->currentThread();
  }
  return 0;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_UTILS_EXPORT void
arcaneSetPauseOnError(bool v)
{
  Arccore::arccoreSetPauseOnError(v);
}

extern "C++" ARCANE_UTILS_EXPORT void
arcaneDebugPause(const char* msg)
{
  Arccore::arccoreDebugPause(msg);
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
/*!
 * \brief Message pour le cas où on essaie d'accèder à un objet plus référence.
 */
extern "C++" ARCANE_UTILS_EXPORT void
arcaneNoReferenceError(const void* ptr)
{
  cerr << "** FATAL: Null reference.\n";
  cerr << "** FATAL: Trying to use an item not referenced.\n";
  cerr << "** FATAL: Item is located at memory address " << ptr << ".\n";
  arcaneDebugPause("arcaneNoReferenceError");
  throw FatalErrorException(A_FUNCINFO);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_UTILS_EXPORT void
arcanePrintf(const char* format,...)
{
  // \n écrit en meme temps pour éviter des écritures intermédiares parasites
  char buffer[256];
  va_list ap;
  va_start(ap,format);
  vsnprintf(buffer,256,format,ap);
  va_end(ap);
  cerr << buffer << "\n";
  cout << "*E* " << buffer << "\n";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_UTILS_EXPORT void
arcaneObsolete(const char* file,const char* func,unsigned long line,const char* text)
{
  cerr << file << ':' << func << ':' << line << '\n';
  cerr << "usage of this function is deprecated";
  if (text)
    cerr << ": " << text;
  cerr << '\n';
}


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/// Fonction appelée lorsqu'une assertion échoue.
typedef void (*fDoAssert)(const char*,const char*,const char*,size_t);
/// Fonction appelée pour indiquer s'il faut afficher l'information de débug
typedef bool (*fCheckDebug)(unsigned int);

static fDoAssert g_do_assert_func = 0;
static fCheckDebug  g_check_debug_func = 0;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * Affichage d'une assertion ayant échouée.
 */
ARCANE_UTILS_EXPORT void
_doAssert(const char* text,const char* file,const char* func,size_t line)
{
  if (g_do_assert_func)
    (*g_do_assert_func)(text,file,func,line);
  else{
    std::ostringstream ostr;
    ostr << text << ':' << file << ':' << func << ':' << line << ": ";
    throw FatalErrorException("Assert",ostr.str());
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/**
 * Vérifie si un message de débug doit être affiché.
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
#define STD_SIGNAL_TYPE   1
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
#endif // STD_SIGNAL_TYPE == 1

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
static fSignalFunc default_signal_func_sigsegv = 0;
static fSignalFunc default_signal_func_sigfpe  = 0;
static fSignalFunc default_signal_func_sigbus  = 0;
//static fSignalFunc default_signal_func_sigsys  = 0;
//static fSignalFunc default_signal_func_sigpipe = 0;
static fSignalFunc default_signal_func_sigalrm = 0;
static fSignalFunc default_signal_func_sigvtalrm = 0;
static bool        global_already_in_signal = false;

extern "C++" ARCANE_UTILS_EXPORT void
arcaneRedirectSignals(fSignalFunc sig_func)
{
  setSignalFunc(sig_func);

#if STD_SIGNAL_TYPE == 1
  default_signal_func_sigsegv = sigset(SIGSEGV,_MiscSigFunc);  // Segmentation fault.
  default_signal_func_sigfpe  = sigset(SIGFPE ,_MiscSigFunc);  // Floating Exception.
  default_signal_func_sigbus  = sigset(SIGBUS ,_MiscSigFunc);  // Bus Error.
  //default_signal_func_sigsys  = sigset(SIGSYS ,_MiscSigFunc);  // Bad argument in system call.
  //default_signal_func_sigpipe = sigset(SIGPIPE,_MiscSigFunc);  // Pipe error.
  default_signal_func_sigalrm = sigset(SIGALRM,_MiscSigFunc);  // Signal alarm (ITIMER_REAL)
  default_signal_func_sigvtalrm = sigset(SIGVTALRM,_MiscSigFunc);  // Signal alarm (ITIMER_VIRTUAL)
#endif // STD_SIGNAL_TYPE == 1
}

extern "C++" ARCANE_UTILS_EXPORT void
arcaneCallDefaultSignal(int val)
{
#if STD_SIGNAL_TYPE == 1

  //fSignalFunc func = 0;
  SignalException::eSignalType signal_type = SignalException::ST_Unknown;

  switch(val){
  case SIGSEGV:
    //func = default_signal_func_sigsegv;
    signal_type = SignalException::ST_SegmentationFault;
    break;
  case SIGFPE:
    //func = default_signal_func_sigfpe;
    signal_type = SignalException::ST_FloatingException;
    break;
  case SIGBUS:
    //func = default_signal_func_sigbus;
    signal_type = SignalException::ST_BusError;
    break;
  case SIGALRM:
  case SIGVTALRM:
    // func = default_signal_func_sigalrm;
    signal_type = SignalException::ST_Alarm;
    break;
  }

  //cerr << "** SIGVAL " << func << ' ' << SIG_DFL << ' '
  //<< SIG_IGN << ' ' << global_already_in_signal << '\n';

  //if (val==SIGSEGV || val==SIGBUS)
  //::abort();

  // En cas de nouveau signal alors qu'on est déja dans un handler, ou
  // s'il s'agit d'un signal d'erreur memoire (SIGBUS ou SIGSEGV), on fait un abort.
  if (global_already_in_signal)
    ::abort();
  global_already_in_signal = true;
    
  arcaneDebugPause("arcaneCallDefaultSignal");

  // Pour éviter certains problèmes lors de la récupération de la pile
  // qui peuvent provoquer une exception, on récupère d'abord la pile
  // d'appel et on la passe au constructeur de l'exception.
  StackTrace stack_trace;
  IStackTraceService* stack_service = platform::getStackTraceService();
  if (stack_service){
    stack_trace = stack_service->stackTrace();
    cerr << " Signal exception Stack: " << stack_trace.toString() << '\n';
  }
  else
    cerr << " No stack trace service\n";
  
  if (signal_type==SignalException::ST_Alarm){
    global_already_in_signal = false;
    throw TimeoutException("Arcane.Signal",stack_trace);
  }
  else if (signal_type==SignalException::ST_FloatingException){
    global_already_in_signal = false;
    cerr << "** THROW ARITHMETIC EXCEPTION\n";
    // Réactive les exceptions flottantes pour le prochain coup
    platform::enableFloatingException(true);
    throw ArithmeticException(A_FUNCINFO,stack_trace);
  }
  throw SignalException("Arcane.Signal",stack_trace,signal_type,val);
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
  HexaPrint(Real v) : m_v(v) {}
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
operator<<(std::ostream& o,const HexaPrint& hp)
{
  hp.write(o);
  return o;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
