// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ProfilingInfo.cc                                            (C) 2000-2023 */
/*                                                                           */
/* Informations de profiling.                                                */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcaneGlobal.h"
#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/utils/Array.h"
#include "arcane/utils/String.h"
#include "arcane/utils/StringBuilder.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/JSONWriter.h"

#include "arcane/std/ProfilingInfo.h"

#include <set>
#include <array>

#include "arcane_packages.h"

//#undef ARCANE_HAS_PACKAGE_LIBUNWIND

#ifdef ARCANE_HAS_PACKAGE_LIBUNWIND
#define UNW_LOCAL_ONLY
#include <libunwind.h>
#include <stdio.h>
#endif
#ifdef __GNUG__
#include <cxxabi.h>
#endif

#if defined(ARCANE_OS_LINUX)
#include <execinfo.h>
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#endif

#if defined(ARCANE_OS_LINUX)
#define ARCANE_HAS_GLIBC_BACKTRACE
#endif

#ifdef ARCANE_USE_MALLOC_HOOK
#include <malloc.h>
#endif

#include <atomic>
#include <algorithm>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface pour récupérer les infos d'une pile d'appel.
 *
 * Il faut d'abord appeler fillStack() avant d'utiliser les autres méthodes.
 */
class ProfInfos::IStackInfoProvider
{
 public:
  virtual ~IStackInfoProvider(){}
  virtual Integer nbIndex() const =0;
  virtual intptr_t functionStartAddress(Int32 stack_index) =0;
  virtual void fillStack(Integer function_depth) =0;
  virtual void setFunc(Int32 func_index,Int32 stack_index) =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ProfInfos::IFuncInfoProvider
{
 public:
  virtual ~IFuncInfoProvider(){}
  virtual void fillFuncName(ProfFuncInfo& pfi) =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ProfInfos::NullFuncInfoProvider
: public IFuncInfoProvider
{
 public:
  virtual ~NullFuncInfoProvider(){}
  virtual void fillFuncName(ProfFuncInfo& pfi) override
  {
    pfi.m_func_name[0] = '\0';
    pfi.setHasFuncName(true);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifdef ARCANE_HAS_PACKAGE_LIBUNWIND
class ProfInfos::LibUnwindFuncInfos
: public IFuncInfoProvider
{
 public:
  void fillFuncName(ProfFuncInfo& pfi) override
  {
    Int32 func_index = pfi.index();
    unw_cursor_t* cursor = &func_cursor[func_index];
    unw_word_t offset;
    unw_get_proc_name(cursor,pfi.m_func_name,MAX_FUNC_LEN,&offset);
    pfi.setHasFuncName(true);
  }
  void setFunc(Int32 func_index,const unw_cursor_t& cursor)
  {
    func_cursor[func_index] = cursor;
  }
 private:
  unw_cursor_t func_cursor[MAX_FUNC];
};

class ProfInfos::LibUnwindStackInfo
: public IStackInfoProvider
{
 public:
  LibUnwindStackInfo(LibUnwindFuncInfos* func_infos)
  : m_nb_index(0), m_func_infos(func_infos){}
 public:
  void fillStack(Integer function_depth) override
  {
    Integer index = 0;
    unw_context_t uc;
    unw_cursor_t cursor;
    unw_getcontext(&uc);
    unw_init_local(&cursor, &uc);
    unw_proc_info_t proc_info;
    while (unw_step(&cursor) > 0 && index<(MAX_STACK+function_depth)) {
      unw_get_proc_info(&cursor,&proc_info);
      m_cursors[index] = cursor;
      m_proc_start[index] = proc_info.start_ip;
      ++index;
    }
    m_nb_index = index;
  }
  void setFunc(Int32 func_index,Int32 stack_index) override
  {
    m_func_infos->setFunc(func_index,m_cursors[stack_index]);
  }
 public:
  Integer nbIndex() const override { return m_nb_index; }
  intptr_t functionStartAddress(Int32 stack_index) override
  {
    return (intptr_t)m_proc_start[stack_index];
  }
 private:
  // TODO: vérifier taille
  unw_cursor_t m_cursors[256];
  unw_word_t m_proc_start[256];
  Integer m_nb_index;
  LibUnwindFuncInfos* m_func_infos;
};
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifdef ARCANE_HAS_GLIBC_BACKTRACE
class ProfInfos::BacktraceFuncInfos
: public IFuncInfoProvider
{
 public:
  void fillFuncName(ProfFuncInfo& pfi) override
  {
    Int32 func_index = pfi.index();
    int copy_index = 0;
    const char* func_name = func_dl_info[func_index].dli_sname;
    if (func_name){
      for( ; copy_index<MAX_FUNC_LEN; ++copy_index ){
        char c = func_name[copy_index];
        pfi.m_func_name[copy_index] = c;
        if (c=='\0')
          break;
      }
    }
    pfi.m_func_name[copy_index] = '\0';
    pfi.setHasFuncName(true);
  }
  void setFunc(Int32 func_index,const Dl_info& dl_info)
  {
    func_dl_info[func_index] = dl_info;
  }
 private:
  Dl_info func_dl_info[MAX_FUNC];
};
class ProfInfos::BacktraceStackInfo
: public IStackInfoProvider
{
 public:
  BacktraceStackInfo(BacktraceFuncInfos* func_infos)
  : m_nb_index(0), m_func_infos(func_infos){}
 public:
  void fillStack(Integer function_depth) override
  {
    ARCANE_UNUSED(function_depth);
    void* addrs[64];
    m_nb_index = backtrace(addrs,64);
    for( Integer index=0; index<m_nb_index; ++index ){
      int err_code = dladdr(addrs[index],&m_dl_infos[index]);
      if (err_code!=0)
        m_proc_start[index] = (intptr_t)m_dl_infos[index].dli_saddr;
      else
        m_proc_start[index] = 0;
    }
  }
  void setFunc(Int32 func_index,Int32 stack_index) override
  {
    m_func_infos->setFunc(func_index,m_dl_infos[stack_index]);
  }
 public:
  Integer nbIndex() const override { return m_nb_index; }
  intptr_t functionStartAddress(Int32 stack_index) override
  {
    return (intptr_t)m_proc_start[stack_index];
  }
 private:
  // TODO: vérifier taille
  std::array<Dl_info,256> m_dl_infos = { };
  std::array<intptr_t,256> m_proc_start = { };
  Integer m_nb_index;
  BacktraceFuncInfos* m_func_infos;
};
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class MonoFuncAddrGetter
{
 private:
  struct _MonoJitInfo;
  struct _MonoDomain;
  struct _MonoMethod;

  typedef void* (*mono_jit_info_get_code_start_func)(_MonoJitInfo* ji);
	typedef _MonoDomain* (*mono_domain_get_func)();
	typedef _MonoJitInfo* (*mono_jit_info_table_find_func)(_MonoDomain*,void* ip);
	typedef char* (*mono_pmip_func)(void* ip);
	typedef _MonoMethod* (*mono_jit_info_get_method_func)(_MonoJitInfo* ji);
	typedef char* (*mono_method_full_name_func)(_MonoMethod* method,bool full);
 public:
  MonoFuncAddrGetter() : m_is_valid(false), m_handle(0)
  {
    empty_func_name[0] = '\0';
#if defined(ARCANE_OS_LINUX)
    void* handle = dlopen (0, RTLD_LAZY);
    if (!handle)
      return;
    m_handle = handle;
    m_mono_jit_info_get_code_start_ptr = (mono_jit_info_get_code_start_func)(dlsym(handle,"mono_jit_info_get_code_start"));
    m_mono_domain_get_ptr = (mono_domain_get_func)(dlsym(handle,"mono_domain_get"));
    m_mono_jit_info_table_find_ptr = (mono_jit_info_table_find_func)(dlsym(handle,"mono_jit_info_table_find"));
    m_mono_pmip_ptr = (mono_pmip_func)(dlsym(handle,"mono_pmip"));
    m_mono_jit_info_get_method_ptr = (mono_jit_info_get_method_func)(dlsym(handle,"mono_jit_info_get_method"));
    m_mono_method_full_name_ptr = (mono_method_full_name_func)(dlsym(handle,"mono_method_full_name"));

    if (!m_mono_jit_info_get_code_start_ptr)
      return;
    if (!m_mono_domain_get_ptr)
      return;
    if (!m_mono_jit_info_table_find_ptr)
      return;
    if (!m_mono_pmip_ptr)
      return;
    if (!m_mono_jit_info_get_method_ptr)
      return;
    if (!m_mono_method_full_name_ptr)
      return;
    m_is_valid = true;
#endif
  }
  ~MonoFuncAddrGetter()
  {
#if defined(ARCANE_OS_LINUX)
    dlclose(m_handle);
#endif
  }

  bool isValid() const
  {
    return m_is_valid;
  }
  char empty_func_name[1];
  char* getInfo(void* ip,void** _start_addr)
  {
    if (!m_is_valid)
      return 0;
    char* func_name = empty_func_name; //(*m_mono_pmip_ptr)(ip);
    _MonoDomain* d = (*m_mono_domain_get_ptr)();
    _MonoJitInfo* ji = (*m_mono_jit_info_table_find_ptr)(d,ip);
    *_start_addr = 0;
    void* start_addr = 0;
    if (ji){
      start_addr = (*m_mono_jit_info_get_code_start_ptr)(ji);
      _MonoMethod* method = (*m_mono_jit_info_get_method_ptr)(ji);
      func_name = (*m_mono_method_full_name_ptr)(method,true);
    }
    //cout << "** START ADDR=" << start_addr << " func_name=" << (void*)func_name << " IP=" << ip << '\n';
    //if (func_name)
    //cout << "** FUNC=" << func_name << '\n';
    *_start_addr = start_addr;
    return func_name;
  }

 private:
  bool m_is_valid;
  mono_jit_info_get_code_start_func m_mono_jit_info_get_code_start_ptr;
	mono_domain_get_func m_mono_domain_get_ptr;
	mono_jit_info_table_find_func m_mono_jit_info_table_find_ptr;
	mono_pmip_func m_mono_pmip_ptr;
  mono_jit_info_get_method_func m_mono_jit_info_get_method_ptr;
  mono_method_full_name_func m_mono_method_full_name_ptr;
  void* m_handle;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

static std::atomic<Int32> global_in_malloc;

#ifdef ARCANE_USE_MALLOC_HOOK

namespace{
void*(*my_old_malloc_hook)(size_t,const void*);
void(*my_old_free_hook)(void*,const void*);
void*(*my_old_realloc_hook)(void* __ptr,size_t __size, __const void*);

}

//static Arcane::Int64 global_nb_malloc = 1;
extern void* prof_malloc_hook(size_t size,const void* caller);
extern void prof_free_hook(void* ptr,const void* caller);
extern void* prof_realloc_hook(void* __ptr,size_t __size,__const void*);

// Ces fonctions ne doivent pas être statiques pour éviter une optimisation
// de GCC 4.7.1 qui fait une boucle infinie dans prof_malloc_hook
// (Note: clang 3.4 a le meme comportement)
extern ARCANE_STD_EXPORT void _pushHook()
{
  __malloc_hook = my_old_malloc_hook;
  __free_hook = my_old_free_hook;
  __realloc_hook = my_old_realloc_hook;
}

extern ARCANE_STD_EXPORT void _popHook()
{
  __malloc_hook = prof_malloc_hook;
  __free_hook = prof_free_hook;
  __realloc_hook = prof_realloc_hook;
}

void* prof_malloc_hook(size_t size,const void* /*caller*/)
{
  ++global_in_malloc;
  _pushHook();
  void* r = malloc(size);
  _popHook();
  --global_in_malloc;
  return r;
}

void prof_free_hook(void* ptr,const void* /*caller*/)
{
  _pushHook();
  free(ptr);
  _popHook();
}

void* prof_realloc_hook(void* ptr,size_t size,const void* /*caller*/)
{
  ++global_in_malloc;
  _pushHook();
  void* r = realloc(ptr,size);
  _popHook();
  --global_in_malloc;
  return r;
}

void _profInitMallocHook()
{
  global_in_malloc = 0;
  my_old_malloc_hook = __malloc_hook;
  __malloc_hook = prof_malloc_hook;
  my_old_free_hook = __free_hook;
  __free_hook = prof_free_hook;
  my_old_realloc_hook = __realloc_hook;
  __realloc_hook = prof_realloc_hook;
}

void _profRestoreMallocHook()
{
  __free_hook = my_old_free_hook;
  __malloc_hook = my_old_malloc_hook;
  __realloc_hook = my_old_realloc_hook;
}
#else
void _profInitMallocHook()
{
}
void _profRestoreMallocHook()
{
}
#endif


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{
/*!
 * \brief Retourne le nom C++ de la méthode \a true_func_name.
 *
 * Le nom C++ est rangé dans le buffer \a demangled_buf de taille maximale
 * \a demangled_buf_len.
 *
 * Retourne \a demangled_buf si le nom a pu être démanglé. Sinon retourne
 * \a true_func_name.
 */
const char*
_getDemangledName(const char* true_func_name,char* demangled_buf,size_t demangled_buf_len)
{
  if (demangled_buf_len<=1)
    return true_func_name;
  size_t len = demangled_buf_len - 1;
  int dstatus = 0;
  const char* buf = nullptr;
#ifdef __GNUG__
  buf = abi::__cxa_demangle(true_func_name,demangled_buf,&len,&dstatus);
#endif
  if (!buf)
    buf = true_func_name;
  return buf;
}
} // End anonymous namespace

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ProfInfos::
ProfInfos(ITraceMng* tm)
: TraceAccessor(tm)
, m_total_event(0)
, m_total_stack(0)
, m_current_func_info(0)
, m_period(0)
, m_nb_event_before_getting_stack(5000)
, m_function_depth(3)
, m_use_backtrace(false)
, m_use_libunwind(false)
, m_mono_func_getter(nullptr)
, m_is_started(false)
, m_default_func_info_provider(new NullFuncInfoProvider())
, m_libunwind_func_info_provider(nullptr)
, m_backtrace_func_info_provider(nullptr)
, m_func_info_provider(m_default_func_info_provider)
{
  for( Integer i=0; i<MAX_COUNTER; ++i )
    m_counters[i] = 0;
  if (platform::hasDotNETRuntime()){
    // Pour l'instant, active uniquement si variable d'environnement
    // positionnée.
    if (!platform::getEnvironmentVariable("ARCANE_DOTNET_BACKTRACE").null()){
      m_mono_func_getter = new MonoFuncAddrGetter();
      if (!m_mono_func_getter->isValid()){
        delete m_mono_func_getter;
        m_mono_func_getter = 0;
      }
    }
  }
#ifdef ARCANE_HAS_PACKAGE_LIBUNWIND
  m_libunwind_func_info_provider = new LibUnwindFuncInfos();
  m_use_libunwind = true;
#endif
#ifdef ARCANE_HAS_GLIBC_BACKTRACE
  m_backtrace_func_info_provider = new BacktraceFuncInfos();
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ProfInfos::
~ProfInfos()
{
  delete m_default_func_info_provider;
  delete m_backtrace_func_info_provider;
  delete m_libunwind_func_info_provider;
  delete m_mono_func_getter;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ProfFuncInfo* ProfInfos::
_getNextFuncInfo()
{
  // TODO: utiliser un atomic pour m_current_func_info lorsqu'il faudra le
  // rendre thread-safe.
  ProfFuncInfo* fi = &m_func_info_buffer[m_current_func_info];
  fi->setIndex(m_current_func_info);
  ++m_current_func_info;
  return fi;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ProfInfos::
startProfiling()
{
  _checkNotStarted();
  m_is_started = true;
  info() << "START PROFILING";
#ifndef __clang__
  //GG Si on utilise ces hook avec clang il part en boucle infini.
  // Pour éviter cela, on le désactive et cela semble fonctionner.
  // De toute facon ces hook sont obsolètes et il faudra penser à faire
  // autrement.
  _profInitMallocHook();
#endif

  String stack_unwinder_str = platform::getEnvironmentVariable("ARCANE_PROFILING_STACKUNWINDING");
  info() << "STACK_UNWIND=" << stack_unwinder_str;
  if (stack_unwinder_str=="backtrace"){
    m_use_backtrace = true;
    m_use_libunwind = false;
  }
  m_func_info_provider = m_default_func_info_provider;
  if (m_use_backtrace)
    m_func_info_provider = m_backtrace_func_info_provider;
  else if (m_use_libunwind)
    m_func_info_provider = m_libunwind_func_info_provider;
  info() << "Start profiling: use_backtrace=" << m_use_backtrace
         << " use_libunwind=" << m_use_libunwind
         << " has_mono=" << m_mono_func_getter;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ProfInfos::
stopProfiling()
{
  m_is_started = false;
  _profRestoreMallocHook();

  // TODO: comme on incrémente, ne faire que à partir de la dernière méthode
  // dont le nom est inconnu.
  ARCANE_CHECK_POINTER(m_func_info_provider);
  for( Int32 i=0, n=m_current_func_info; i<n; ++i ){
    ProfFuncInfo& pfi = m_func_info_buffer[i];
    if (!pfi.hasFuncName())
      m_func_info_provider->fillFuncName(pfi);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ProfInfos::
_checkNotStarted()
{
  if (m_is_started)
    ARCANE_FATAL("Invalid call when profiling is active");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ProfInfos::
setFunctionDepth(int v)
{
  _checkNotStarted();
  m_function_depth = v;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ProfInfos::
setPeriod(int v)
{
  _checkNotStarted();
  m_period = v;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
namespace
{
int global_nb_in_handler = 0;
}

void ProfInfos::
addEvent(void* address,int overflow_event[MAX_COUNTER],int nb_overflow_event)
{
  static bool is_in_handler = false;

  if (is_in_handler){
    ++global_nb_in_handler;
    return;
  }
  is_in_handler = true;
  if (m_use_backtrace){
    //_addEventBacktrace(address,overflow_event,nb_overflow_event);
#ifdef ARCANE_HAS_GLIBC_BACKTRACE
    BacktraceStackInfo stack_info((BacktraceFuncInfos*)m_backtrace_func_info_provider);
    _addEvent(address,overflow_event,nb_overflow_event,stack_info,m_function_depth+2);
#endif
  }
  else if (m_use_libunwind){
#ifdef ARCANE_HAS_PACKAGE_LIBUNWIND
    LibUnwindStackInfo stack_info((LibUnwindFuncInfos*)m_libunwind_func_info_provider);
    _addEvent(address,overflow_event,nb_overflow_event,stack_info,m_function_depth+1);
#endif
  }
  is_in_handler = false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ProfInfos::
_sortFunctions(std::set<ProfFuncInfo*,ProfFuncComparer>& sorted_func)
{
  for( const auto& x : m_func_map )
    if (x.second)
      sorted_func.insert(x.second);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ProfInfos::
_storeAddress(void* address,bool is_counter0,int overflow_event[MAX_COUNTER],int nb_overflow_event,
              bool* do_add,bool* do_stack,bool* func_already_added)
{
  ++m_total_event;
  ProfInfos::AddrMap::iterator v = m_addr_map.find(address);
  if (v!=m_addr_map.end()){
    ProfAddrInfo& ai = v->second;
    for( int i=0; i<nb_overflow_event; ++i )
      ++ai.m_counters[ overflow_event[i] ];
    if (ai.m_func_info){
      for( int i=0; i<nb_overflow_event; ++i )
        ++ai.m_func_info->m_counters[ overflow_event[i] ];
      *func_already_added = true;
      // Si on a déjà suffisamment d'évènements et que notre méthode
      // dépasse les 1% du temps passé, conserve la pile associée.
      if (m_total_event>m_nb_event_before_getting_stack){
        if ((ai.m_func_info->m_counters[0]*100)>m_total_event){
          ai.m_func_info->m_do_stack = true;
          if (is_counter0)
            *do_stack = true;
        }
      }
    }
  }
  else
    *do_add = true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ProfInfos::
_addEvent(void* address,int overflow_event[MAX_COUNTER],int nb_overflow_event,
          IStackInfoProvider& stack_info,Integer function_depth)
{
  // Si on est dans un malloc, ne fait rien.
  // TODO: il faudrait quand meme incrementer le compteur correspondant
  // car dans ce cas le temps passé dans les malloc/realloc/free n'est pas pris en compte
  // TODO: faire test atomic
  if (global_in_malloc!=0){
    //cout << "V=" <<global_in_malloc << '\n';
    return;
  }

  bool is_counter0 = false;

  for( int i=0; i<nb_overflow_event; ++i ){
    if (overflow_event[i]==0)
      is_counter0 = true;
    if (overflow_event[i]<0 || overflow_event[i]>=MAX_COUNTER)
      cerr << "arcane_papi_handler: EVENT ERROR n=" << overflow_event[i] << '\n';
  }

  bool do_stack = false;
  bool do_add = false;
  bool func_already_added = false;
  _storeAddress(address,is_counter0,overflow_event,nb_overflow_event,
                &do_add,&do_stack,&func_already_added);

  if (do_add || do_stack){
    ProfAddrInfo papi_address_info;
    ProfStackInfo papi_stack_info;
    for( int i=0; i<nb_overflow_event; ++i )
      ++papi_address_info.m_counters[ overflow_event[i] ];

    stack_info.fillStack(function_depth);

    for (Integer index=function_depth, nb_index=stack_info.nbIndex(); index<nb_index; ++index ){
      intptr_t proc_start = stack_info.functionStartAddress(index);

      ProfInfos::FuncMap::iterator func = m_func_map.find(proc_start);
      ProfFuncInfo* papi_func_info = nullptr;
      if (func==m_func_map.end()){
        if (m_current_func_info>=MAX_FUNC){
          cerr << "arcane_papi_handler: MAX_FUNC reached !\n";
          break;
        }
        papi_func_info = _getNextFuncInfo();
        papi_address_info.m_func_info = papi_func_info;
        Int32 func_index = papi_func_info->index();
        stack_info.setFunc(func_index,index);
        m_func_map.insert(ProfInfos::FuncMap::value_type(proc_start,papi_func_info));
      }
      else{
        papi_func_info = func->second;
      }

      if (index<(MAX_STACK+function_depth))
        papi_stack_info.m_funcs_info_indexes[index-function_depth] = papi_func_info->index();
      if (index==function_depth){
        papi_address_info.m_func_info = papi_func_info;
        if (!func_already_added)
          for( int i=0; i<nb_overflow_event; ++i )
            ++papi_func_info->m_counters[ overflow_event[i] ];
        if (!papi_func_info->m_do_stack || !is_counter0)
          break;
        else
          do_stack = true;
      }
    }
    if (do_stack){
      ++m_total_stack;
      ProfInfos::StackMap::iterator st = m_stack_map.find(papi_stack_info);
      if (st!=m_stack_map.end()){
        ++st->second;
      }
      else
        m_stack_map.insert(ProfInfos::StackMap::value_type(papi_stack_info,1));
    }
    if (do_add)
      m_addr_map.insert(ProfInfos::AddrMap::value_type(address,papi_address_info));
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
// NOTE: cette méthode n'est plus utilisée mais le mécanisme d'accès via mono
// doit être intégré au code de libunwind ou backtrace.
bool ProfInfos::
_getFunc(void* addr,FuncAddrInfo& info)
{
#ifdef ARCANE_OS_LINUX
  Dl_info dl_info;
  int r = dladdr(addr,&dl_info);
  info.func_name = "unknown";
  info.start_addr = 0;
  if (r!=0){
    // Il est possible que dladdr ne retourne pas d'erreur mais ne trouve
    // pas le symbole. Dans ce cas, essaie de voir s'il s'agit d'un
    // symbole C#.
    info.start_addr = dl_info.dli_saddr;
    if (dl_info.dli_sname){
      info.func_name = dl_info.dli_sname;
      return false;
    }
  }
  if (m_mono_func_getter){
    void* start_addr = 0;
    char* func_name = m_mono_func_getter->getInfo(addr,&start_addr);
    if (func_name && start_addr){
      info.start_addr = start_addr;
      info.func_name = func_name;
      return false;
    }
  }
#endif
  return true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ProfInfos::
reset()
{
  ProfInfos* global_infos = this;
  {
    ProfInfos::AddrMap::iterator begin = global_infos->m_addr_map.begin();
    for( ; begin!=global_infos->m_addr_map.end(); ++begin ){
      for( Integer i=0; i<MAX_COUNTER; ++i )
        begin->second.m_counters[i] = 0;
    }
  }
  {
    ProfInfos::FuncMap::iterator begin = global_infos->m_func_map.begin();
    for( ; begin!=global_infos->m_func_map.end(); ++begin ){
      ProfFuncInfo* pf = begin->second;
      for( Integer i=0; i<MAX_COUNTER; ++i )
        pf->m_counters[i] =0;
    }
  }
  m_total_event = 0;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ProfInfos::
printInfos(bool dump_file)
{
  ProfInfos* global_infos = this;

  Int64 total_event = 1;
  Int64 total_fp = 1;
  int process_id = platform::getProcessId();
  {
    info() << " PROCESS_ID = " << process_id;
    info() << " NB ADDRESS MAP = " << global_infos->m_addr_map.size();
    info() << " NB FUNC MAP = " << global_infos->m_func_map.size();
    info() << " NB STACK MAP = " << global_infos->m_stack_map.size();
    info() << " TOTAL STACK = " << global_infos->m_total_stack;
    {
      ProfInfos::AddrMap::const_iterator begin = global_infos->m_addr_map.begin();
      for( ; begin!=global_infos->m_addr_map.end(); ++begin ){
        Int64 nb_event = begin->second.m_counters[0];
        total_event += nb_event;
        Int64 nb_fp = begin->second.m_counters[2];
        total_fp += nb_fp;
      }
    }
    {
      ProfInfos::FuncMap::const_iterator begin = global_infos->m_func_map.begin();
      Int64 total_func_event = 0;
      Int64 total_func_fp = 0;
      for( ; begin!=global_infos->m_func_map.end(); ++begin ){
        ProfFuncInfo* pf = begin->second;
        Int64 nb_event = pf->m_counters[0];
        total_func_event += nb_event;
        Int64 nb_fp = pf->m_counters[2];
        total_func_fp += nb_fp;
      }
      info() << " FUNC EVENT=" << total_func_event;
      info() << " FUNC FP=" << total_func_fp;
    }

    if (dump_file){
      StringBuilder sfile_name = "profiling.addr";
      sfile_name += process_id;
      sfile_name += ".xml";
      String file_name = sfile_name;
      std::ofstream ofile(file_name.localstr());
      ofile << "<?xml version='1.0'?>\n";
      ofile << "<addresses>\n";
      ProfInfos::AddrMap::const_iterator begin = global_infos->m_addr_map.begin();
      for( ; begin!=global_infos->m_addr_map.end(); ++begin ){
        void* addr = begin->first;
        const ProfAddrInfo& pa = begin->second;
        ProfFuncInfo* fi = pa.m_func_info;
        if (fi){
          ofile << "<addr addr='" << addr << "'"
                << " count='" << pa.m_counters[0] << "'"
                << " fi='" << fi << "'"
                << " fi_count='" << fi->m_counters[0] << "'"
                << " func='" << fi->m_func_name << "'"
                << ">\n";
        }
      }
      ofile << "/<addresses>\n";
    }
  }

  std::set<ProfFuncInfo*,ProfFuncComparer> sorted_func;
  _sortFunctions(sorted_func);

  const size_t NAME_BUF_SIZE = 8120;
  char demangled_func_name[NAME_BUF_SIZE];
  {
    info() << " TOTAL EVENT  = " << total_event;
    info() << " TOTAL NB_IN_HANDLER  = " << global_nb_in_handler;
    Real nb_gflop = ((Real)total_fp*(Real)m_period) * 1e-09;
    info() << " TOTAL FP     = " << total_fp << " (nb_giga_flip=" << nb_gflop << ")";
    info() << " RATIO FP/CYC = " << ((Real)total_fp/(Real)total_event);
      info() << "  " << Trace::Width(10) << "nb_event"
             << "  " << Trace::Width(5) << "%"
             << "  " << Trace::Width(5) << "cum%"
             << " " << Trace::Width(12) << "event1"
             << " " << Trace::Width(12) << "event2"
             << " " << Trace::Width(12) << "event3"
             << " " << " "
             << "  " << "function";
    Integer index = 0;
    Int64 cumulative_nb_event = 0;
    for( ProfFuncInfo* pfi : sorted_func ){
      const char* func_name = pfi->m_func_name;
      const char* buf = _getDemangledName(func_name,demangled_func_name,NAME_BUF_SIZE);
      Int64 nb_event = pfi->m_counters[0];
      cumulative_nb_event += nb_event;
      Int64 cumulative_total_percent = (cumulative_nb_event * 1000) / total_event;
      Int64 total_percent = (nb_event * 1000) / total_event;
      Int64 cumulative_percent = (cumulative_total_percent/10);
      Int64 percent = (total_percent/10);
      info() << "  " << Trace::Width(10) << nb_event
             << "  " << Trace::Width(3) << percent << "." << (total_percent % 10)
             << "  " << Trace::Width(3) << cumulative_percent << "." << (cumulative_total_percent % 10)
             << " " << Trace::Width(12) << pfi->m_counters[0]
             << " " << Trace::Width(12) << pfi->m_counters[1]
             << " " << Trace::Width(12) << pfi->m_counters[2]
             << " " << pfi->m_do_stack
             << "  " << buf;
      if (total_percent<5 && index>20)
        break;
      ++index;
    }
  }
  // TODO: Calculer ces informations lors de l'arrêt du profiling.
  if (dump_file){
    // Créée une liste des piles triée par nombre d'évènements décroissant.
    UniqueArray<SortedProfStackInfo> sorted_stacks;
    for( const auto& x : global_infos->m_stack_map ){
      const ProfStackInfo& psi = x.first;
      Int64 nb_stack = x.second;
      sorted_stacks.add(SortedProfStackInfo(psi,nb_stack));
    }
    std::sort(std::begin(sorted_stacks),std::end(sorted_stacks));

    String file_name = String("profiling.callstack") + platform::getProcessId();
    std::ofstream ofile;
    ofile.open(file_name.localstr());

    for( const auto& x : sorted_stacks ){
      const ProfStackInfo& psi = x.stackInfo();
      Int64 nb_stack = x.nbCount();
      if (nb_stack<2)
        continue;
      ofile << " Stack nb=" << nb_stack << '\n';
      for( Integer z=0; z<MAX_STACK; ++z ){
        Int32 fidx = psi.m_funcs_info_indexes[z];
        if (fidx>=0){
          ProfFuncInfo& pfi = _funcInfoFromIndex(fidx);
          const char* func_name = pfi.m_func_name;
          const char* buf = _getDemangledName(func_name,demangled_func_name,NAME_BUF_SIZE);
          ofile << "  " << buf << '\n';
        }
      }
    }
    ofile.close();
  }
}


// ****************************************************************************
// * getInfos for someone else
// *  [0] 0xb80dd1a3ul
// *  [1] Size
// *  [2] total_event
// *  [3] total_fp
// *  [4] (Int64) int m_period
// *  [5] (Int64) Integer index;
// *     nb_event
// *     total_percent
// *     m_counters[0]: PAPI_TOT_CYC Total cycles
// *     m_counters[1]: PAPI_RES_STL Cycles stalled on any resource
// *     m_counters[2]: PAPI_FP_INS  Floating point instructions
// *     strlen(func_name)
// *     
// *     
// ****************************************************************************
void ProfInfos::
getInfos(Int64Array& pkt)
{
  const size_t NAME_BUF_SIZE = 8192;
  char demangled_func_name[NAME_BUF_SIZE];
  ProfInfos* global_infos = this;
  Int64 total_event = 1;
  Int64 total_fp = 1;
  Integer index = 0;
  
  {
    ProfInfos::AddrMap::const_iterator begin = global_infos->m_addr_map.begin();
    for( ; begin!=global_infos->m_addr_map.end(); ++begin ){
      Int64 nb_event = begin->second.m_counters[0];
      total_event += nb_event;
      Int64 nb_fp = begin->second.m_counters[2];
      total_fp += nb_fp;
    }
  }
  //Real nb_gflop = ((Real)total_fp*(Real)m_period) * 1e-09;
  pkt.add(total_event);
  pkt.add(total_fp);
  pkt.add((Int64)m_period);
  info()<<"[ProfInfos::getInfos] total_event="<<total_event<<", total_fp="<<total_fp<<", m_period="<<m_period;
  
  std::set<ProfFuncInfo*,ProfFuncComparer> sorted_func;
  _sortFunctions(sorted_func);
 
  // On pousse l'index qui vaut zero
  pkt.add((Int64)index);

  // Et on continue à remplir l'array
  for( ProfFuncInfo* pfi : sorted_func ){
    const char* func_name = pfi->m_func_name;
    const char* buf = _getDemangledName(func_name,demangled_func_name,NAME_BUF_SIZE);
    Int64 nb_event = pfi->m_counters[0];
    pkt.add(nb_event);
    Int64 total_percent = (nb_event * 1000) / total_event;
    pkt.add(total_percent);
    pkt.add(pfi->m_counters[0]);
    pkt.add(pfi->m_counters[1]);
    pkt.add(pfi->m_counters[2]);

    Int64 mx = strlen(buf);
    pkt.add(mx);
    for(Int64 i=0; i<mx; i+=8)
      pkt.add(*(Int64*)&buf[i]);
    
    if (total_percent<1 && index>16)
      break;
    ++index;
  }
  pkt[5]=(Int64)(index);
  info()<<"[ProfInfos::getInfos] index @[4] ="<<pkt[5];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ProfInfos::
dumpJSON(JSONWriter& writer)
{
  // TODO: utiliser un identifiant pour les noms de fonction au lieu de
  // mettre directement le nom de la méthode.

  ProfInfos* global_infos = this;

  Int64 total_event = 1;
  for( const auto& x : global_infos->m_addr_map ){
    Int64 nb_event = x.second.m_counters[0];
    total_event += nb_event;
  }

  {
    int process_id = platform::getProcessId();
    writer.write("ProcessId",(Int64)process_id);
    writer.write("NbAddressMap",global_infos->m_addr_map.size());
    writer.write("NbFuncMap",global_infos->m_func_map.size());
    writer.write("NbStackMap",global_infos->m_stack_map.size());
    writer.write("TotalStack",global_infos->m_total_stack);
    writer.write("TotalEvent",total_event);
  }

  std::set<ProfFuncInfo*,ProfFuncComparer> sorted_func;
  _sortFunctions(sorted_func);

  const size_t NAME_BUF_SIZE = 8192;
  char demangled_func_name[NAME_BUF_SIZE];

  // Ecrit la liste des méthodes référencées et leur nom manglé et démanglé
  {
    writer.writeKey("Functions");
    writer.beginArray();
    for( Int32 i=0, n=m_current_func_info; i<n; ++i ){
      const ProfFuncInfo& pfi = m_func_info_buffer[i];
      const char* func_name = pfi.m_func_name;
      const char* buf = _getDemangledName(func_name,demangled_func_name,NAME_BUF_SIZE);

      {
        JSONWriter::Object o(writer);
        writer.write("Index",(Int64)pfi.index());
        writer.write("Name",func_name);
        writer.write("DemangledName",buf);
      }

    }
    writer.endArray();
  }

  // Ecrit la liste des méthodes triées par ordre décroissant des valeurs
  // des compteurs.
  {
    Integer index = 0;
    writer.writeKey("SortedFuncTimes");
    writer.beginArray();
    for( ProfFuncInfo* pfi : sorted_func ){
      const char* func_name = pfi->m_func_name;
      const char* buf = _getDemangledName(func_name,demangled_func_name,NAME_BUF_SIZE);
      {
        JSONWriter::Object o(writer);
        writer.write("Index",(Int64)(pfi->index()));
        writer.write("Name",buf);
        writer.write("Events",Int64ConstArrayView(3,pfi->m_counters));
      }
      if (index>100)
        break;
      ++index;
    }
    writer.endArray();
  }

  {
    // Créée une liste des piles triée par nombre d'évènements décroissant.
    UniqueArray<SortedProfStackInfo> sorted_stacks;
    for( const auto& x : global_infos->m_stack_map ){
      const ProfStackInfo& psi = x.first;
      Int64 nb_stack = x.second;
      sorted_stacks.add(SortedProfStackInfo(psi,nb_stack));
    }
    std::sort(std::begin(sorted_stacks),std::end(sorted_stacks));

    writer.writeKey("StackMap");
    writer.beginArray();
    for( const auto& x : sorted_stacks ){
      const ProfStackInfo& psi = x.stackInfo();
      Int64 nb_stack = x.nbCount();
      if (nb_stack<2)
        continue;
      {
        JSONWriter::Object o(writer);
        writer.write("Count",nb_stack);
        writer.write("Stacks",Int32ConstArrayView(MAX_STACK,psi.m_funcs_info_indexes));

        writer.writeKey("StackNames");
        writer.beginArray();
        for( Integer z=0; z<MAX_STACK; ++z ){
          Int32 fidx = psi.m_funcs_info_indexes[z];
          if (fidx>=0){
            ProfFuncInfo& pfi = _funcInfoFromIndex(fidx);
            const char* func_name = pfi.m_func_name;
            const char* buf = _getDemangledName(func_name,demangled_func_name,NAME_BUF_SIZE);
            writer.writeValue(buf);
          }
          else
            writer.writeValue(String());
        }
        writer.endArray();
      }
    }
    writer.endArray();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
