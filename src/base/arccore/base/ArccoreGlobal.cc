/*---------------------------------------------------------------------------*/
/* ArccoreGlobal.cc                                            (C) 2000-2018 */
/*                                                                           */
/* D�clarations g�n�rales de Arccore.                                        */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ArccoreGlobal.h"
#include "arccore/base/TraceInfo.h"
#include "arccore/base/PlatformUtils.h"
#include "arccore/base/String.h"
#include "arccore/base/IndexOutOfRangeException.h"
#include "arccore/base/FatalErrorException.h"

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
 * \namespace Arccore
 *
 * \brief Espace de nom de %Arccore
 *
 * Toutes les classes et types utilis�s dans \b Arccore sont dans ce
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
    std::ostringstream ostr;
    String host_name(Platform::getHostName());
    ostr << "** FATAL: Debug mode activated. Execution paused\n"
         << "** FATAL: message:" << msg << "\n"
         << "** FATAL: To find the location of the error, start\n"
         << "** FATAL: start the debugger using the process number\n"
         << "** FATAL: (pid=" << Platform::getProcessId() << ",host=" << host_name << ").\n";
    std::cerr << ostr.str();
#ifndef ARCANE_OS_WIN32
    ::pause();
#endif
  }
}

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

// Cette fonction peut �tre appel�e souvent et certaines fois
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

/// Fonction appel�e lorsqu'une assertion �choue.
typedef void (*fDoAssert)(const char*,const char*,const char*,size_t);
/// Fonction appel�e pour indiquer s'il faut afficher l'information de d�bug
typedef bool (*fCheckDebug)(unsigned int);

static fDoAssert g_do_assert_func = 0;
static fCheckDebug  g_check_debug_func = 0;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * Affichage d'une assertion ayant �chou�e.
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
  // \n �crit en meme temps pour �viter des �critures interm�diares parasites
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
