﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* PlatformUtils.cc                                            (C) 2000-2024 */
/*                                                                           */
/* Fonctions utilitaires dépendant de la plateforme.                         */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/String.h"
#include "arcane/utils/StdHeader.h"
#include "arcane/utils/StackTrace.h"
#include "arcane/utils/IStackTraceService.h"
#include "arcane/utils/IOnlineDebuggerService.h"
#include "arcane/utils/Iostream.h"
#include "arcane/utils/StringBuilder.h"
#include "arcane/utils/NotSupportedException.h"
#include "arcane/utils/Array.h"
#include "arcane/utils/StringList.h"
#include "arcane/utils/internal/MemoryRessourceMng.h"

#include <chrono>

#ifndef ARCANE_OS_WIN32
#define ARCANE_OS_UNIX
#endif

#ifdef ARCANE_OS_WIN32
#include <sys/types.h>
#include <sys/timeb.h>
#include <sys/stat.h>
#include <direct.h>
#include <process.h>
#include <windows.h>
#include <shlobj.h>
#endif

#ifdef ARCANE_OS_UNIX
#include <unistd.h>
#include <sys/resource.h>
#include <time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <fcntl.h>
#endif

#include <malloc.h>

#if !defined(ARCANE_OS_CYGWIN) && !defined(ARCANE_OS_WIN32)
#if defined(__i386__)
#define ARCANE_HAS_I386_FPU_CONTROL_H
#include <fpu_control.h>
#endif
#endif

#ifndef ARCANE_OS_WIN32
#include <pwd.h>
#include <sys/types.h>
#include <unistd.h>
#endif

// Support pour gérer les exceptions flottantes:
// - sous Linux avec la GlibC, cela se fait via les méthodes
// feenableexcept(), fedisableexcept() et fegetexcept()
// - sous Win32, cela se fait via la méthode _controlfp() mais pour
// l'instant ce n'est pas utilisé dans Arcane.
#if defined(ARCANE_OS_LINUX) && defined(__USE_GNU)
#  include <fenv.h>
#  define ARCANE_GLIBC_FENV
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace platform
{
  IOnlineDebuggerService* global_online_debugger_service = nullptr;
  ISymbolizerService* global_symbolizer_service = nullptr;
  IProfilingService* global_profiling_service = nullptr;
  IProcessorAffinityService* global_processor_affinity_service = nullptr;
  IMemoryAllocator* global_accelerator_host_memory_allocator = nullptr;
  IDynamicLibraryLoader* global_dynamic_library_loader = nullptr;
  MemoryRessourceMng global_default_data_memory_ressource_mng;
  IMemoryRessourceMng* global_data_memory_ressource_mng = nullptr;
  IPerformanceCounterService* global_performance_counter_service = nullptr;
  bool global_has_color_console = false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_UTILS_EXPORT ISymbolizerService* platform::
getSymbolizerService()
{
  return global_symbolizer_service;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_UTILS_EXPORT ISymbolizerService* platform::
setSymbolizerService(ISymbolizerService* service)
{
  ISymbolizerService* old_service = global_symbolizer_service;
  global_symbolizer_service = service;
  return old_service;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_UTILS_EXPORT IOnlineDebuggerService* platform::
getOnlineDebuggerService()
{
  return global_online_debugger_service;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_UTILS_EXPORT IOnlineDebuggerService* platform::
setOnlineDebuggerService(IOnlineDebuggerService* service) 
{
  IOnlineDebuggerService* old_service = global_online_debugger_service;
  global_online_debugger_service = service;
  return old_service;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_UTILS_EXPORT IProfilingService* platform::
getProfilingService()
{
  return global_profiling_service;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_UTILS_EXPORT IProfilingService* platform::
setProfilingService(IProfilingService* service) 
{
  IProfilingService* old_service = global_profiling_service;
  global_profiling_service = service;
  return old_service;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_UTILS_EXPORT IDynamicLibraryLoader* platform::
getDynamicLibraryLoader()
{
  return global_dynamic_library_loader;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_UTILS_EXPORT IDynamicLibraryLoader* platform::
setDynamicLibraryLoader(IDynamicLibraryLoader* idll)
{
  IDynamicLibraryLoader* old_service = global_dynamic_library_loader;
  global_dynamic_library_loader = idll;
  return old_service;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_UTILS_EXPORT IProcessorAffinityService* platform::
getProcessorAffinityService()
{
  return global_processor_affinity_service;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_UTILS_EXPORT IProcessorAffinityService* platform::
setProcessorAffinityService(IProcessorAffinityService* service) 
{
  IProcessorAffinityService* old_service = global_processor_affinity_service;
  global_processor_affinity_service = service;
  return old_service;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_UTILS_EXPORT IPerformanceCounterService* platform::
getPerformanceCounterService()
{
  return global_performance_counter_service;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_UTILS_EXPORT IPerformanceCounterService* platform::
setPerformanceCounterService(IPerformanceCounterService* service)
{
  auto* old_service = global_performance_counter_service;
  global_performance_counter_service = service;
  return old_service;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_UTILS_EXPORT IMemoryAllocator* platform::
getAcceleratorHostMemoryAllocator()
{
  return global_accelerator_host_memory_allocator;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_UTILS_EXPORT IMemoryAllocator* platform::
setAcceleratorHostMemoryAllocator(IMemoryAllocator* a)
{
  IMemoryAllocator* old = global_accelerator_host_memory_allocator;
  global_accelerator_host_memory_allocator = a;
  return old;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_UTILS_EXPORT IMemoryAllocator* platform::
getDefaultDataAllocator()
{
  return getDataMemoryRessourceMng()->getAllocator(eMemoryRessource::UnifiedMemory);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_UTILS_EXPORT IMemoryRessourceMng* platform::
setDataMemoryRessourceMng(IMemoryRessourceMng* mng)
{
  ARCANE_CHECK_POINTER(mng);
  IMemoryRessourceMng* old = global_data_memory_ressource_mng;
  global_data_memory_ressource_mng = mng;
  return old;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_UTILS_EXPORT IMemoryRessourceMng* platform::
getDataMemoryRessourceMng()
{
  IMemoryRessourceMng* a = global_data_memory_ressource_mng;
  if (!a)
    return &global_default_data_memory_ressource_mng;
  return a;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_UTILS_EXPORT IThreadImplementation* platform::
getThreadImplementationService()
{
  return Arccore::Concurrency::getThreadImplementation();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_UTILS_EXPORT IThreadImplementation* platform::
setThreadImplementationService(IThreadImplementation* service)
{
  return Arccore::Concurrency::setThreadImplementation(service);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_UTILS_EXPORT void platform::
resetAlarmTimer(Integer nb_second)
{
#ifdef ARCANE_OS_UNIX
  struct itimerval time_val;
  struct itimerval otime_val;
  time_val.it_value.tv_sec     = nb_second;
  time_val.it_value.tv_usec    = 0;
  time_val.it_interval.tv_sec  = 0;
  time_val.it_interval.tv_usec = 0;
  // Utilise le temps virtuel et pas le temps réel.
  // Cela permet de suspendre temporairement un job (par exemple
  // pour régler des problèmes systèmes) sans déclencher l'alarme.
  int r = setitimer(ITIMER_VIRTUAL,&time_val,&otime_val);
  if (r!=0)
    cout << "** ERROR in setitimer r=" << r << '\n';
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_UTILS_EXPORT void platform::
platformInitialize()
{
  Arccore::Platform::platformInitialize();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_UTILS_EXPORT void platform::
platformTerminate()
{
  Arccore::Platform::platformTerminate();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{
template<typename ByteType> bool
_readAllFile(StringView filename, bool is_binary, Array<ByteType>& out_bytes)
{
  using namespace std;
  long unsigned int file_length = platform::getFileLength(filename);
  if (file_length == 0) {
    //cerr << "** FAIL LENGTH\n";
    return true;
  }
  ifstream ifile;
  ios::openmode mode = ios::in;
  if (is_binary)
    mode |= ios::binary;
  ifile.open(filename.toStdStringView().data());
  if (ifile.fail()) {
    //cerr << "** FAIL OPEN\n";
    return true;
  }
  out_bytes.resize(file_length);
  ifile.read((char*)(out_bytes.data()), file_length);
  if (ifile.bad()) {
    // cerr << "** BAD READ\n";
    return true;
  }
  // Il est possible que le nombre d'octets lus soit inférieur
  // à la longueur du fichier, notamment sous Windows avec les fichiers
  // texte et la conversion des retour-chariots. Il faut donc redimensionner
  // \a bytes à la bonne longueur.
  size_t nb_read = ifile.gcount();
  out_bytes.resize(nb_read);
  //cerr << "** READ " << file_length << " bytes " << (const char*)(bytes.begin()) << "\n";
  return false;
}
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool platform::
readAllFile(StringView filename, bool is_binary, ByteArray& out_bytes)
{
  return _readAllFile(filename,is_binary,out_bytes);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool platform::
readAllFile(StringView filename, bool is_binary, Array<std::byte>& out_bytes)
{
  return _readAllFile(filename,is_binary,out_bytes);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

static bool global_has_dotnet_runtime = false;
extern "C++" bool platform::
hasDotNETRuntime()
{
  return global_has_dotnet_runtime;
}

extern "C++" void platform::
setHasDotNETRuntime(bool v)
{
  global_has_dotnet_runtime = v;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" String platform::
getExeFullPath()
{
  String full_path;
#if defined(ARCANE_OS_LINUX)
  char* buf = ::realpath("/proc/self/exe",nullptr);
  if (buf){
    full_path = StringView(buf);
    ::free(buf);
  }
#elif defined(ARCANE_OS_WIN32)
  char buf[2048];
  int r = GetModuleFileNameA(NULL,buf,2000);
  if (r>0){
    full_path = StringView(buf);
  }
#else
#error "platform::getExeFullPath() not implemented for this platform"
#endif
  return full_path;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" String platform::
getLoadedSharedLibraryFullPath(const String& dll_name)
{
  String full_path;
  if (dll_name.null())
    return full_path;
#if defined(ARCANE_OS_LINUX)
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
#elif defined(ARCANE_OS_WIN32)
  HMODULE hModule = GetModuleHandleA(dll_name.localstr());
  if (!hModule)
    return full_path;
  TCHAR dllPath[_MAX_PATH];
  GetModuleFileName(hModule, dllPath, _MAX_PATH);
  full_path = StringView(dllPath);
#else
  throw NotSupportedException(A_FUNCINFO);
//#error "platform::getSymbolFullPath() not implemented for this platform"
#endif
  return full_path;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" void platform::
fillCommandLineArguments(StringList& arg_list)
{
  arg_list.clear();
#if defined(ARCANE_OS_LINUX)

  const int BUFSIZE = 1024;
  char buffer[BUFSIZE + 1];

  UniqueArray<char> bytes;
  bytes.reserve(1024);

  {
    const char* filename = "/proc/self/cmdline";
    int fd = open(filename, O_RDONLY);
    if (fd<0)
      return;
    ssize_t nb_read = 0;
    // TODO: traiter les interruptions
    while ((nb_read = read(fd, buffer, BUFSIZE)) > 0) {
      buffer[BUFSIZE] = '\0';
      bytes.addRange(Span<const char>(buffer, nb_read));
    }
    close(fd);
  }

  int size = bytes.size();
  const char* ptr = bytes.data();
  const char* end = ptr + size;
  while (ptr < end) {
    arg_list.add(StringView(ptr));
    while (*ptr++ && ptr < end)
      ;
  }
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
// TODO: mettre ensuite dans 'Arccore'

extern "C++" ARCANE_UTILS_EXPORT Int64 platform::
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

extern "C++" ARCANE_UTILS_EXPORT Int64 platform::
getPageSize()
{
#if defined(ARCCORE_OS_WIN32)
  SYSTEM_INFO si;
  GetSystemInfo(&si);
  return si.dwPageSize;
#elif defined(ARCANE_OS_LINUX)
  return ::sysconf(_SC_PAGESIZE);
#else
#warning "getPageSize() not implemented for your platform. Default is 4096"
  Int64 page_size = 4096;
  return page_size;
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{
void (*global_garbage_collector_delegate)() = nullptr;
}

extern "C" ARCANE_UTILS_EXPORT void
_ArcaneSetCallGarbageCollectorDelegate(void(*f)())
{
  global_garbage_collector_delegate = f;
}

extern "C++" void platform::
callDotNETGarbageCollector()
{
  if (global_garbage_collector_delegate)
    (*global_garbage_collector_delegate)();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
