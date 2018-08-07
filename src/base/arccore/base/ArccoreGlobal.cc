/*---------------------------------------------------------------------------*/
/* ArccoreGlobal.cc                                            (C) 2000-2018 */
/*                                                                           */
/* Déclarations générales de Arccore.                                        */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ArccoreGlobal.h"

#include <iostream>

#ifndef ARCCORE_OS_WIN32
#include <unistd.h>
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
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

namespace Arccore
{
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

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
#if ARCCORE_NEED_IMPLEMENTATION
    OStringStream ostr;
    String host_name(platform::getHostName());
    ostr() << "** FATAL: Debug mode activated. Execution paused\n"
           << "** FATAL: message:" << msg << "\n"
           << "** FATAL: To find the location of the error, start\n"
           << "** FATAL: start the debugger using the process number\n"
           << "** FATAL: (pid=" << platform::getProcessId() << ",host=" << host_name << ").\n";
    cerr << ostr.str();
#else
    std::cerr << "arccoreDebugPause Not Yet Implemented";
#endif
#ifndef ARCANE_OS_WIN32
    ::pause();
#endif
  }
}

extern "C++" ARCCORE_BASE_EXPORT void
arccoreRangeError(Int32 i,Int32 max_size)
{
  arccoreDebugPause("arccoreRangeError");
#if ARCCORE_NEED_IMPLEMENTATION
  throw IndexOutOfRangeException(A_FUNCINFO,String(),i,0,max_size);
#else
  throw std::exception();
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCCORE_BASE_EXPORT void
arccoreRangeError(Int64 i,Int64 max_size)
{
  arccoreDebugPause("arccoreRangeError");
#if ARCCORE_NEED_IMPLEMENTATION
  throw IndexOutOfRangeException(A_FUNCINFO,String(),i,0,max_size);
#else
  throw std::exception();
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
