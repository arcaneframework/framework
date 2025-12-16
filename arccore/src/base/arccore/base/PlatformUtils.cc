// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* PlatformUtils.cc                                            (C) 2000-2025 */
/*                                                                           */
/* Fonctions utilitaires dépendant de la plateforme.                         */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/PlatformUtils.h"
#include "arccore/base/String.h"
#include "arccore/base/StackTrace.h"
#include "arccore/base/IStackTraceService.h"
#include "arccore/base/StringBuilder.h"

#include <chrono>
#include <iostream>
#include <fstream>
#include <cstring>

#ifdef ARCCORE_OS_WIN32
#include <sys/types.h>
#include <sys/timeb.h>
#include <sys/stat.h>
#include <direct.h>
#include <process.h>
#include <windows.h>
#include <shlobj.h>
#endif

#if defined(ARCCORE_OS_LINUX) or defined(ARCCORE_OS_MACOS)
#include <unistd.h>
#include <sys/resource.h>
#include <time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/time.h>
#endif

#if defined(ARCCORE_OS_MACOS)
#include <cstdlib>
#include <mach-o/dyld.h>
#include <crt_externs.h>
#endif

// SD: Useless ? Bug with MacOS
//#include <malloc.h>

#if !defined(ARCCORE_OS_CYGWIN) && !defined(ARCCORE_OS_WIN32)
#if defined(__i386__)
#define ARCCORE_HAS_I386_FPU_CONTROL_H
#include <fpu_control.h>
#endif
#endif

#ifndef ARCCORE_OS_WIN32
#include <pwd.h>
#include <sys/types.h>
#include <unistd.h>
#endif

// Support pour gérer les exceptions flottantes:
// - sous Linux avec la GlibC, cela se fait via les méthodes
// feenableexcept(), fedisableexcept() et fegetexcept()
// - sous Win32, cela se fait via la méthode _controlfp() mais pour
// l'instant ce n'est pas utilisé dans Arccore.
#if defined(ARCCORE_OS_LINUX) && defined(__USE_GNU)
#  include <fenv.h>
#  define ARCCORE_GLIBC_FENV
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

// Ces deux fonctions sont définies dans 'Exception.cc'
extern "C++" ARCCORE_BASE_EXPORT void
arccoreSetPauseOnException(bool v);
extern "C++" ARCCORE_BASE_EXPORT void
arccoreCallExplainInExceptionConstructor(bool v);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Platform
{
  IStackTraceService* global_stack_trace_service = nullptr;
  bool global_has_color_console = false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCCORE_BASE_EXPORT String Platform::
getCurrentDate()
{
  static const size_t max_len = 80;
  char str[max_len];
  time_t now_time;
  const struct tm* now_tm;
  ::time(&now_time);
  now_tm = ::localtime(&now_time);

  ::strftime(str,max_len,"%m/%d/%Y %X",now_tm);
  return std::string_view(str);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCCORE_BASE_EXPORT long Platform::
getCurrentTime()
{
  time_t now_time;
  ::time(&now_time);
  return (long)now_time;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCCORE_BASE_EXPORT String Platform::
getCurrentDateTime()
{
  static const size_t max_len = 80;
  char str[max_len];
  time_t now_time;
  const struct tm* now_tm;
  ::time(&now_time);
  now_tm = ::localtime(&now_time);

  // Formattage ISO 8601
  ::strftime(str,max_len,"%Y-%m-%dT%H:%M:%S",now_tm);
  return std::string_view(str);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCCORE_BASE_EXPORT String Platform::
getHostName()
{
  char buf[1024];
  const char* host_name = buf;
#ifdef ARCCORE_OS_WIN32
  size_t len = sizeof(buf)-1;
  DWORD slen = (DWORD)len;
  if (GetComputerName(buf,&slen)==0)
    host_name = "Unknown";
#else
  if (::gethostname(buf,sizeof(buf)-1)!=0)
    host_name = "Unknown";
#endif
  return std::string_view(host_name);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCCORE_BASE_EXPORT String Platform::
getUserName()
{
  // Récupère le nom de l'utilisateur
  String user_name = "noname";
#ifdef ARCCORE_OS_WIN32
  char buf[1024];
  const char* user_name_ptr = buf;
  size_t len = sizeof(buf)-1;
  DWORD slen = (DWORD)len;
  if (GetUserName(buf,&slen)==0)
    user_name_ptr = "Unknown";
  user_name = String(user_name_ptr,true);
#else
  struct passwd* pwd = ::getpwuid(::getuid());
  if (pwd)
    user_name = String(std::string_view(pwd->pw_name));
#endif
  return user_name;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCCORE_BASE_EXPORT String Platform::
getHomeDirectory()
{
  String home_dir("/");
#ifdef ARCCORE_OS_WIN32
  {
    TCHAR szPath[MAX_PATH];
    if(SUCCEEDED(SHGetFolderPath(NULL, 
                                 CSIDL_PERSONAL, 
                                 NULL, 0, szPath)))
      {
        home_dir = String(szPath,true);
      }
    else 
      {
        home_dir = "c:";
      }
  }
#else
  String user_home_env = Platform::getEnvironmentVariable("HOME");
  if (!user_home_env.null())
    home_dir = user_home_env;
#endif
  return home_dir;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCCORE_BASE_EXPORT String Platform::
getCurrentDirectory()
{
  char buf[4096];
  char* c = ::getcwd(buf,4095);
  if (!c){
    std::cerr << "ERROR: Arccore::getCurrentDirectory() can not get current directory\n";
    return std::string_view(".");
  }
  return std::string_view(buf);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCCORE_BASE_EXPORT int Platform::
getProcessId()
{
  return ::getpid();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCCORE_BASE_EXPORT long unsigned int Platform::
getFileLength(const String& filename)
{
  struct stat stat_info;
  int r = ::stat(filename.localstr(),&stat_info);
  if (r==-1)
    return 0UL;
  return stat_info.st_size;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCCORE_BASE_EXPORT String Platform::
getEnvironmentVariable(const String& name)
{
  char* s = ::getenv(name.localstr());
  if (!s)
    return String();
  return String(s);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Platform
{
extern "C++" ARCCORE_BASE_EXPORT bool
isDirectoryExist(const String& dir_name,bool& can_create)
{
  can_create = true;
  struct stat dirstat;
  const char* dirname = dir_name.localstr();
  int stat_val = ::stat(dirname,&dirstat);
  if (stat_val==0){
    if (dirstat.st_mode & S_IFDIR)
      return true;
    // Le fichier existe mais n'est pas un répertoire
    can_create = false;
    return false;
  }
  return false;
}
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCCORE_BASE_EXPORT bool Platform::
recursiveCreateDirectory(const String& dir_name)
{
  //cerr << "** REC CRE DIR " << dir_name << '\n';
  bool can_create = false;
  bool is_dir = isDirectoryExist(dir_name,can_create);
  if (is_dir)
    return false;
  if (!can_create)
    return true;
  Int64 pos = 0;
  const char* dir_name_str = dir_name.localstr();
  for( Int64 i=0, is=dir_name.length(); i<is; ++i )
    if (dir_name_str[i]=='/')
      pos = i;
  if (pos!=0){
    String parent_dir(std::string_view(dir_name_str,pos));
    bool is_bad = recursiveCreateDirectory(parent_dir);
    if (is_bad)
      return true;
  }
  //cerr << "** REAL DIR " << dir_name << '\n';
#ifdef ARCCORE_OS_WIN32
  int ret = mkdir(dir_name_str);
#else
  int ret = mkdir(dir_name_str,S_IRWXU|S_IRWXG);
#endif
  if (ret==(-1))
    return true;
  return false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCCORE_BASE_EXPORT bool Platform::
createDirectory(const String& dir_name)
{
  //cout << "** CREATE DIRECTORY " << dir_name << '\n';
  struct stat dirstat;
  const char* dirname = dir_name.localstr();
  int stat_val = ::stat(dirname,&dirstat);
  if (stat_val==0){
    if (dirstat.st_mode & S_IFDIR)
      return false;
    // Le fichier existe mais n'est pas un répertoire
    return true;
  }
#ifdef ARCCORE_OS_WIN32
  int ret = mkdir(dirname);
#else
  int ret = mkdir(dirname,S_IRWXU|S_IRWXG);
#endif
  if (ret==(-1))
    return true;
  return false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCCORE_BASE_EXPORT String Platform::
getFileDirName(const String& file_name)
{
  // Sous windows, regarde s'il y a des '/'. Dans ce cas on prend ce caractère comme
  // séparateur. Sinon, on prend '\\'.
  char separator = '/';
  const char* file_name_str = file_name.localstr();
#if defined(ARCCORE_OS_WIN32)
  {
    bool has_slash = false;
    for( Int64 i=0, n=file_name.length(); i<n; ++i )
      if (file_name_str[i]=='/'){
        has_slash = true;
        break;
      }
    if (!has_slash)
      separator = '\\';
  }
#endif
  Int64 pos = 0;
  for( Int64 i=0, n=file_name.length(); i<n; ++i )
    if (file_name_str[i]==separator)
      pos = i;

  if (pos==0)
    return String(".");

  return std::string_view(file_name_str,pos);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCCORE_BASE_EXPORT bool Platform::
removeFile(const String& file_name)
{
  const char* file_name_str = file_name.localstr();
	int r = ::unlink(file_name_str);
	if (r!=0)
		return true;
	return false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCCORE_BASE_EXPORT bool Platform::
isFileReadable(const String& file_name)
{
  std::ifstream ifile(file_name.localstr());
	if (!ifile)
		return false;
	return true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCCORE_BASE_EXPORT void Platform::
stdMemcpy(void* to,const void* from,::size_t len)
{
  std::memcpy(to,from,len);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCCORE_BASE_EXPORT bool Platform::
isDenormalized(Real /*v*/)
{
  //TODO: à implémenter
  return false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCCORE_BASE_EXPORT IStackTraceService* Platform::
getStackTraceService()
{
  return global_stack_trace_service;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCCORE_BASE_EXPORT IStackTraceService* Platform::
setStackTraceService(IStackTraceService* service)
{
  IStackTraceService* old_service = global_stack_trace_service;
  global_stack_trace_service = service;
  return old_service;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCCORE_BASE_EXPORT String Platform::
getStackTrace()
{
  IStackTraceService* stack_service = getStackTraceService();
  String s;
  if (stack_service){
    s = stack_service->stackTrace().toString();
  }
  return s;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCCORE_BASE_EXPORT void Platform::
safeStringCopy(char* output,Integer /*output_len*/,const char* input)
{
  //TODO utiliser correctement 'output_len'
  ::strcpy(output,input);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifdef ARCCORE_OS_LINUX
static long _getRSSMemoryLinux()
{
  FILE* f = ::fopen("/proc/self/stat","r");
  if (!f)
    return 0;

  // See proc(1), category 'stat'
  int z_pid;
  char z_comm[4096];
  char z_state;
  int z_ppid;
  int z_pgrp;
  int z_session;
  int z_tty_nr;
  int z_tpgid;
  unsigned long z_flags;
  unsigned long z_minflt;
  unsigned long z_cminflt;
  unsigned long z_majflt;
  unsigned long z_cmajflt;
  unsigned long z_utime;
  unsigned long z_stime;
  long z_cutime;
  long z_cstime;
  long z_priority;
  long z_nice;
  long z_zero;
  long z_itrealvalue;
  long z_starttime;
  unsigned long z_vsize;
  long z_rss;
  int r = fscanf(f,"%d %s %c %d %d %d %d %d"
                 "%lu %lu %lu %lu %lu %lu %lu"
                 "%ld %ld %ld %ld %ld %ld %ld %lu %ld",
                 &z_pid,z_comm,&z_state,&z_ppid,&z_pgrp,&z_session,&z_tty_nr,&z_tpgid,
                 &z_flags,&z_minflt,&z_cminflt,&z_majflt,&z_cmajflt,&z_utime,&z_stime,
                 &z_cutime,&z_cstime,&z_priority,&z_nice,&z_zero,&z_itrealvalue,
                 &z_starttime,&z_vsize,&z_rss);
  if (r!=24)
    z_rss = 0;

  ::fclose(f);
  return z_rss;
}
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCCORE_BASE_EXPORT double Platform::
getMemoryUsed()
{
  double mem_used = 0.0;

#ifdef ARCCORE_OS_LINUX
  long z_rss = _getRSSMemoryLinux();
  double d_z_rss = (double)(z_rss);
  double d_pagesize = (double)getpagesize();
  mem_used = d_z_rss * d_pagesize;
#endif

  return mem_used;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCCORE_BASE_EXPORT Int64 Platform::
getCPUTime()
{
#if defined(ARCCORE_OS_LINUX)
  struct timespec ts;
  if (clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts) == 0){
    return (ts.tv_sec*1000000) + (ts.tv_nsec / 1000);
  }
  return 1;
#else
  // TODO Supprimer l'usage de clock() pour Win32 et MacOS.
  static Int64 orig_clock = 0;
  static clock_t call_clock = 0;
  clock_t current_clock = ::clock();
  if (orig_clock==0 && call_clock==0){
    call_clock = current_clock;
  }
  else{
    if (current_clock<call_clock){
      // On a depasse la valeur max de clock
      //cout << " WARNING: CLOCK depasse INT_MAX: " << call_clock
      //     << " current=" << current_clock << '\n';
      Int64 v = ARCCORE_INT64_MAX;
      v *= 2;
      orig_clock += v;
    }
    call_clock = current_clock;
  }
  //cout << " INFO: CLOCK orig=" << orig_clock << " call=" << call_clock
  //   << " current=" << current_clock << '\n';
  return orig_clock + call_clock;
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCCORE_BASE_EXPORT Real Platform::
getRealTime()
{
  auto x = std::chrono::high_resolution_clock::now();
  // Converti la valeur en nanosecondes.
  auto y = std::chrono::time_point_cast<std::chrono::nanoseconds>(x);
  // Retourne le temps en secondes.
  return (Real)y.time_since_epoch().count() / 1.0e9;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCCORE_BASE_EXPORT String Platform::
timeToHourMinuteSecond(Real t)
{
  Int64 s = (Int64)(t);
  Int64 hour = s / 3600;
  Int64 remaining_hour = s - (3600 * hour);
  Int64 minute = remaining_hour / 60;
  Int64 second = remaining_hour - (60 * minute);

  StringBuilder sb;
  sb += hour;
  sb += "h";
  sb += minute;
  sb += "m";
  sb += second;
  sb += "s";

  return sb.toString();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Met le process en sommeil pendant \a nb_second secondes.
 */
extern "C++" ARCCORE_BASE_EXPORT void Platform::
sleep(Integer nb_second)
{
  if (nb_second>0){
#ifdef ARCCORE_OS_WIN32
    Sleep(nb_second*1000);
#endif
#ifdef ARCCORE_OS_UNIX
    ::sleep(nb_second);
#endif
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCCORE_BASE_EXPORT void Platform::
dumpStackTrace(std::ostream& ostr)
{
  IStackTraceService* stack_service = Platform::getStackTraceService();
  if (stack_service){
    StackTrace st = stack_service->stackTrace();
    String s = st.toString();
    ostr << s;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifdef ARCCORE_GLIBC_FENV
const int FloatExceptFlags = FE_DIVBYZERO|FE_INVALID;
#endif

extern "C++" ARCCORE_BASE_EXPORT bool Platform::
hasFloatingExceptionSupport()
{
#ifdef ARCCORE_GLIBC_FENV
  return true;
#else
  return false;
#endif
}

extern "C++" ARCCORE_BASE_EXPORT void Platform::
enableFloatingException(bool active)
{
#ifdef ARCCORE_GLIBC_FENV
 if (active){
    ::fesetenv(FE_DFL_ENV);
    ::feenableexcept(FloatExceptFlags);
  }
  else{
    ::fedisableexcept(FloatExceptFlags);
  }
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCCORE_BASE_EXPORT bool Platform::
isFloatingExceptionEnabled()
{
#ifdef ARCCORE_GLIBC_FENV
  int x = ::fegetexcept();
  return (x & (FloatExceptFlags))!=0;
#endif
  return false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCCORE_BASE_EXPORT void Platform::
raiseFloatingException()
{
#ifdef ARCCORE_GLIBC_FENV
  // Note: cette méthode n'a besoin que de fenv.h et devrait être portable
  // même sans la GLIBC.
  ::feraiseexcept(FloatExceptFlags);
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCCORE_BASE_EXPORT Int64 Platform::
getRealTimeNS()
{
  auto x = std::chrono::high_resolution_clock::now();
  // Converti la valeur en nanosecondes.
  auto y = std::chrono::time_point_cast<std::chrono::nanoseconds>(x);
  // Retourne le temps en nano-secondes.
  return static_cast<Int64>(y.time_since_epoch().count());
}


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" Int64 Platform::
getPageSize()
{
#if defined(ARCCORE_OS_WIN32)
  SYSTEM_INFO si;
  GetSystemInfo(&si);
  return si.dwPageSize;
#elif defined(ARCCORE_OS_LINUX)
  return ::sysconf(_SC_PAGESIZE);
#else
#warning "getPageSize() not implemented for your platform. Default is 4096"
  Int64 page_size = 4096;
  return page_size;
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool _getHasColorTerminal()
{
#ifdef ARCCORE_OS_UNIX
  String force_color = Platform::getEnvironmentVariable("ARCCORE_COLORTERM");
  if (!force_color.null())
    return true;

  if (!::isatty(::fileno(stdout)))
    return false;

  String term = Platform::getEnvironmentVariable("TERM");
  //std::cout << "CHECK ENV term=" << term << '\n';

  if (term=="xterm-color" || term=="xterm-256color")
    return true;

  if (term=="xterm" || term=="rxvt" || term=="rxvt-unicode"){
    String color_term = Platform::getEnvironmentVariable("COLORTERM");
    //std::cout << "CHECK ENV color_term=" << color_term << '\n';
    if (!color_term.null())
      return true;
  }
#endif
  return false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Platform::
platformInitialize(bool enable_fpe)
{
  // Pour l'instant, la seule initialisation spécifique dépend
  // des processeurs i386. Elle consiste à changer la valeur par
  // défaut des flags de la FPU pour générer une exception
  // lors d'une erreur arithmétique (comme les divisions par zéro).
  if (enable_fpe)
    enableFloatingException(true);

  getCPUTime();

  global_has_color_console = _getHasColorTerminal();

  if (getEnvironmentVariable("ARCCORE_PAUSE_ON_EXCEPTION")=="1")
    arccoreSetPauseOnException(true);
  if (getEnvironmentVariable("ARCCORE_PRINT_ON_EXCEPTION")=="1")
    arccoreCallExplainInExceptionConstructor(true);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Platform::
platformInitialize()
{
  platformInitialize(true);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCCORE_BASE_EXPORT void Platform::
platformTerminate()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" bool Platform::
getConsoleHasColor()
{
  return global_has_color_console;
}


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" String Platform::
getLoadedSharedLibraryFullPath(const String& dll_name)
{
  String full_path;
  if (dll_name.null())
    return full_path;
#if defined(ARCCORE_OS_LINUX)
  {
    std::ifstream ifile("/proc/self/maps");
    String v;
    String true_name = "lib" + dll_name + ".so";
    while (ifile.good()){
      ifile >> v;
      Span<const Byte> vb = v.bytes();
      if (vb.size()>0 && vb[0]=='/'){
        if (v.endsWith(true_name)){
          full_path = v;
          //std::cout << "V='" << v << "'\n";
          break;
        }
      }
    }
  }
#elif defined(ARCCORE_OS_WIN32)
  HMODULE hModule = GetModuleHandleA(dll_name.localstr());
  if (!hModule)
    return full_path;
  TCHAR dllPath[_MAX_PATH];
  GetModuleFileName(hModule, dllPath, _MAX_PATH);
  full_path = StringView(dllPath);
#elif defined(ARCCORE_OS_MACOS)
  {
    String true_name = "lib" + dll_name + ".dylib";
    uint32_t count = _dyld_image_count();
    for (uint32_t i = 0; i < count; i++) {
      const char* image_name = _dyld_get_image_name(i);
      if (image_name) {
        String image_path(image_name);
        if (image_path.endsWith(true_name)) {
          full_path = image_path;
          break;
        }
      }
    }
  }
#else
  throw NotSupportedException(A_FUNCINFO);
//#error "platform::getSymbolFullPath() not implemented for this platform"
#endif
  return full_path;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//TODO: faire pour windows.
extern "C++" String Platform::
getCompilerId()
{
  String compiler_name = "Unknown";
  Integer version_major = 0;
  Integer version_minor = 0;
#ifdef __clang__
  compiler_name = "Clang";
  version_major = __clang_major__;
  version_minor = __clang_minor__;
#else
#ifdef __INTEL_COMPILER
  compiler_name = "ICC";
  version_major = __INTEL_COMPILER / 100;
  version_minor = __INTEL_COMPILER % 100;
#else
#ifdef __GNUC__
  compiler_name = "GCC";
  version_major = __GNUC__;
  version_minor = __GNUC_MINOR__;
#else
#ifdef _MSC_VER
  compiler_name = "MSVC";
  version_major = _MSC_VER / 100;
  version_minor = _MSC_VER % 100;
#endif // _MSC_VER
#endif // __GNUC__
#endif // __INTEL_COMPILER
#endif // __clang__
  StringBuilder ostr;
  ostr += compiler_name;
  ostr += " ";
  ostr += version_major;
  ostr += ".";
  ostr += version_minor;
  return ostr.toString();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
