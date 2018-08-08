/*---------------------------------------------------------------------------*/
/* ArccoreGlobal.cc                                            (C) 2000-2018 */
/*                                                                           */
/* Déclarations générales de Arccore.                                        */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ArccoreGlobal.h"
#include "arccore/base/TraceInfo.h"

#include <iostream>
#include <cstring>

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

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
