// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ArccoreGlobal.cc                                            (C) 2000-2025 */
/*                                                                           */
/* Déclarations générales de Arccore.                                        */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ArccoreGlobal.h"
#include "arccore/base/TraceInfo.h"
#include "arccore/base/PlatformUtils.h"
#include "arccore/base/String.h"
#include "arccore/base/IndexOutOfRangeException.h"
#include "arccore/base/FatalErrorException.h"
#include "arccore/base/Ref.h"

// Nécessaire pour les exports de symboles
#include "arccore/base/ReferenceCounterImpl.h"
#include "arccore/base/Float16.h"
#include "arccore/base/BFloat16.h"
#include "arccore/base/Float128.h"
#include "arccore/base/Int128.h"
#include "arccore/base/IRangeFunctor.h"
#include "arccore/base/CheckedConvert.h"
#include "arccore/base/ForLoopRunInfo.h"
#include "arccore/base/ForLoopRanges.h"
#include "arccore/base/ParallelLoopOptions.h"
#include "arccore/base/ISymbolizerService.h"
#include "arccore/base/internal/IDynamicLibraryLoader.h"

#include <iostream>
#include <cstring>
#include <sstream>
#include <cstdarg>

#ifndef ARCCORE_OS_WIN32
#include <unistd.h>
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \file ArccoreGlobal.h
 *
 * \brief Définitions et globaux de %Arccore
 */
/*!
 * \namespace Arccore
 *
 * \brief Espace de nom de %Arccore
 *
 * Toutes les classes et types utilisés dans \b Arccore sont dans ce
 * namespace.
 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{
#ifdef ARCCORE_CHECK
static bool global_arccore_is_check = true;
#else
static bool global_arccore_is_check = false;
#endif
}

extern "C++" ARCCORE_BASE_EXPORT
bool arccoreIsCheck()
{
  return global_arccore_is_check;
}

extern "C++" ARCCORE_BASE_EXPORT
void arccoreSetCheck(bool v)
{
  global_arccore_is_check = v;
}

extern "C++" ARCCORE_BASE_EXPORT
bool arccoreIsDebug()
{
#ifdef ARCCORE_DEBUG
  return true;
#else
  return false;
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{
bool global_pause_on_error = false;
}

extern "C++" ARCCORE_BASE_EXPORT void
arccoreSetPauseOnError(bool v)
{
  global_pause_on_error = v;
}

extern "C++" ARCCORE_BASE_EXPORT void
arccoreDebugPause(const char* msg)
{
  if (global_pause_on_error){
    std::ostringstream ostr;
    String host_name(Platform::getHostName());
    ostr << "** FATAL: Debug mode activated. Execution paused\n"
         << "** FATAL: message:" << msg << "\n"
         << "** FATAL: To find the location of the error, start\n"
         << "** FATAL: start the debugger using the process number\n"
         << "** FATAL: (pid=" << Platform::getProcessId() << ",host=" << host_name << ").\n";
    std::cerr << ostr.str();
#ifndef ARCCORE_OS_WIN32
    ::pause();
#endif
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCCORE_BASE_EXPORT void
arccoreRangeError(Int64 i,Int64 min_value_inclusive,Int64 max_value_exclusive)
{
  arccoreDebugPause("arccoreRangeError");
  throw IndexOutOfRangeException(A_FUNCINFO,String(),i,min_value_inclusive,max_value_exclusive);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCCORE_BASE_EXPORT void
arccoreRangeError(Int32 i,Int32 max_size)
{
  arccoreDebugPause("arccoreRangeError");
  throw IndexOutOfRangeException(A_FUNCINFO,String(),i,0,max_size);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCCORE_BASE_EXPORT void
arccoreRangeError(Int64 i,Int64 max_size)
{
  arccoreDebugPause("arccoreRangeError");
  throw IndexOutOfRangeException(A_FUNCINFO,String(),i,0,max_size);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCCORE_BASE_EXPORT void
arccoreNullPointerError()
{
  std::cerr << "** FATAL: null pointer.\n";
  std::cerr << "** FATAL: Trying to dereference a null pointer.\n";
  arccoreDebugPause("arcaneNullPointerPtr");
  throw FatalErrorException(A_FUNCINFO,"null pointer");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCCORE_BASE_EXPORT void
arccoreThrowNullPointerError(const char* ptr_name,const char* text)
{
  throw FatalErrorException(A_FUNCINFO,text ? text : ptr_name);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Cette fonction peut être appelée souvent et certaines fois
// dans des conditions d'exceptions. Pour cette raison, il ne
// faut pas qu'elle fasse d'allocations.
namespace
{
void _printFuncName(std::ostream& o,const char* name)
{
  const char* par_pos = std::strchr(name,'(');
  if (!par_pos){
    o << name;
    return;
  }

  // Recherche quelque chose du type namespace::class_name::func_name
  // et essaye de ne conserver que class_name::func_name
  ptrdiff_t len = par_pos - name;
  ptrdiff_t last_scope = 0;
  ptrdiff_t last_scope2 = 0;
  for( ptrdiff_t i=0; i<len; ++i ){
    if (name[i]==':' && name[i+1]==':'){
      last_scope2 = last_scope;
      last_scope = i;
    }
  }
  if (last_scope2!=0)
    last_scope2+=2;
  ptrdiff_t true_pos = last_scope2;
  ptrdiff_t true_len = len - true_pos;
  o.write(&name[true_pos],true_len);
  o << "()";
}
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCCORE_BASE_EXPORT std::ostream&
operator<<(std::ostream& o,const TraceInfo& t)
{
  if (t.printSignature())
    o << t.name() << ":" << t.line(); 
  else{
    _printFuncName(o,t.name());
  }
  return o;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{
/// Fonction appelée lorsqu'une assertion échoue.
typedef void (*fDoAssert)(const char*,const char*,const char*,size_t);
/// Fonction appelée pour indiquer s'il faut afficher l'information de débug
typedef bool (*fCheckDebug)(unsigned int);

fDoAssert g_do_assert_func = 0;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * Affichage d'une assertion ayant échouée.
 */
extern "C++" ARCCORE_BASE_EXPORT void
_doAssert(const char* text,const char* file,const char* func,int line)
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

extern "C++" ARCCORE_BASE_EXPORT void
arccorePrintf(const char* format,...)
{
  // \n écrit en meme temps pour éviter des écritures intermédiares parasites
  char buffer[4096];
  va_list ap;
  va_start(ap,format);
  vsnprintf(buffer,4095,format,ap);
  va_end(ap);
  std::cerr << buffer << "\n";
  std::cout << "*E* " << buffer << "\n";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
