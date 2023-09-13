// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ProfilingInfo.h                                             (C) 2000-2021 */
/*                                                                           */
/* Structures d'informations pour le profiling.                              */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_STD_PROFILINGINFO_H
#define ARCANE_STD_PROFILINGINFO_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"
#include "arcane/utils/FatalErrorException.h"

#include <map>
#include <set>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

static const int MAX_COUNTER = 3;
static const int MAX_STACK = 16;
static const int MAX_FUNC = 10000;
static const int MAX_FUNC_LEN = 500;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ProfFuncInfo
{
 public:
  ProfFuncInfo() : m_index(0), m_do_stack(false), m_has_func_name(false)
  {
    m_func_name[0]='\0';
    for( Integer i=0; i<MAX_COUNTER; ++i )
      m_counters[i] = 0;
  }
 public:
  Int32 index() const { return m_index; }
  void setIndex(Int32 v) { m_index = v; }
  bool hasFuncName() const { return m_has_func_name; }
  void setHasFuncName(bool v) { m_has_func_name = v; }
 private:
  Int32 m_index;
 public:
  bool m_do_stack;
 private:
  bool m_has_func_name;
 public:
  Int64 m_counters[MAX_COUNTER];
  // TODO: Ne pas utiliser une taille max mais utiliser un buffer contenant
  // tous les noms
  char m_func_name[MAX_FUNC_LEN+10];
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ProfStackInfo
{
 public:
  ProfStackInfo()
  {
    for( Integer i=0; i<MAX_STACK; ++i )
      m_funcs_info_indexes[i] = (-1);
  }
 public:
  bool operator<(const ProfStackInfo& pfi) const
  {
    return ::memcmp(m_funcs_info_indexes,pfi.m_funcs_info_indexes,MAX_STACK*sizeof(Int32))<0;
  }
 public:
  Int32 m_funcs_info_indexes[MAX_STACK];
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ProfAddrInfo
{
 public:
 public:
  ProfAddrInfo() : m_func_info(0)
  {
    for( Integer i=0; i<MAX_COUNTER; ++i )
      m_counters[i] = 0;
  }
 public:
  Int64 m_counters[MAX_COUNTER];
  ProfFuncInfo* m_func_info;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ProfFuncComparer
{
 public:
  typedef ProfFuncInfo* ProfFuncInfoPtr;
 public:
  bool operator()(const ProfFuncInfoPtr& lhs,const ProfFuncInfoPtr& rhs) const
  {
    return (lhs->m_counters[0]>rhs->m_counters[0]);
  }
};

static const int MAX_STATIC_ALLOC = 100000;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Allocateur static pour le profiling.
 *
 * Comme les infos de profiling sont geres lors d'interruption
 * et peuvent survenir n'importe ou, il ne faut pas faire d'allocation
 * standard (new/malloc) mais utiliser un buffer statique.
 * NOTE: cet allocateur n'est valide que pour la classe ProfilingInfo
 * et ne doit pas être utilisé ailleurs.
 */
template<typename _Tp>
class StaticAlloc
{
 public:
  typedef size_t     size_type;
  typedef ptrdiff_t  difference_type;
  typedef _Tp*       pointer;
  typedef const _Tp* const_pointer;
  typedef _Tp&       reference;
  typedef const _Tp& const_reference;
  typedef _Tp        value_type;

  template<typename _Tp1>
  struct rebind
  { typedef StaticAlloc<_Tp1> other; };

  StaticAlloc() ARCANE_NOEXCEPT
  {
  }
  
  StaticAlloc(const StaticAlloc&) ARCANE_NOEXCEPT
  {
  }
  
  template<typename _Tp1>
  StaticAlloc(const StaticAlloc<_Tp1>&) ARCANE_NOEXCEPT { }

  ~StaticAlloc() ARCANE_NOEXCEPT { }

  void construct(pointer __p, const _Tp& __val) 
  { ::new((void *)__p) _Tp(__val); }

  void destroy(pointer __p) { __p->~_Tp(); }

  pointer allocate(size_type /*__n*/, const void* = 0)
  { 
    //return static_cast<_Tp*>(::operator new(__n * sizeof(_Tp)));
    pointer p = &m_buf[m_buf_index];
    //TODO: rendre atomic + verifier debordement
    ++m_buf_index;
    if (m_buf_index>=(int)(0.9*MAX_STATIC_ALLOC))
      cout << "** WARNING: allocate near max memory\n";
    if (m_buf_index>=MAX_STATIC_ALLOC)
      throw FatalErrorException("StaticAlloc","max static alloc reached");
    return p;
  }
  void deallocate(pointer, size_type)
  {
    // Les deallocate ne sont pas utilisés car on ne doit pas supprimer des
    // elements de la map
    //::operator delete(__p);
  }
 private:
  static _Tp m_buf[MAX_STATIC_ALLOC];
  static int m_buf_index ;
};

template<typename _Tp> int
StaticAlloc<_Tp>::m_buf_index = 0;

template<typename _Tp> _Tp
StaticAlloc<_Tp>::m_buf[MAX_STATIC_ALLOC];

class MonoFuncAddrGetter;

class ProfInfos
: public TraceAccessor
{
  class SortedProfStackInfo
  {
   public:
    SortedProfStackInfo(const ProfStackInfo& psi,Int64 nb_count)
    : m_stack_info(psi), m_nb_count(nb_count)
    {
    }
   public:
    bool operator<(const SortedProfStackInfo& pfi) const
    {
      return this->m_nb_count > pfi.m_nb_count;
    }
    const ProfStackInfo& stackInfo() const { return m_stack_info; }
    Int64 nbCount() const { return m_nb_count; }
   private:
    ProfStackInfo m_stack_info;
    Int64 m_nb_count;
  };
 private:
  class IStackInfoProvider;
  class IFuncInfoProvider;
  class NullFuncInfoProvider;
  class LibUnwindFuncInfos;
  class LibUnwindStackInfo;
  class BacktraceFuncInfos;
  class BacktraceStackInfo;
 private:
  struct FuncAddrInfo
  {
    FuncAddrInfo() : start_addr(0), func_name(0) {}
    void* start_addr;
    const char* func_name;
  };
 public:
  ProfInfos(ITraceMng* tm);
  ~ProfInfos();
 public:
  // IMPORTANT:
  // les std::map ne doivent pas utiliser d'allocation dynamique
  // car elles sont utilisées dans la methode addEvent() qui peut être
  // appelée n'importe quand et donc dans un malloc/realloc/free
  // et cela peut donc provoquer un blocage avec les thread.
#if defined(ARCCORE_OS_WIN32) || defined(ARCCORE_OS_MACOS)
  typedef std::map<void*,ProfAddrInfo> AddrMap;
  typedef std::map<Int64,ProfFuncInfo*> FuncMap;
  typedef std::map<ProfStackInfo,Int64> StackMap;
#else
  typedef std::map<void*,ProfAddrInfo,std::less<void*>,StaticAlloc<std::pair<void* const,ProfAddrInfo> > > AddrMap;
  typedef std::map<Int64,ProfFuncInfo*,std::less<Int64>,StaticAlloc<std::pair<const Int64,ProfFuncInfo*> > > FuncMap;
  typedef std::map<ProfStackInfo,Int64,std::less<ProfStackInfo>,StaticAlloc<std::pair<const ProfStackInfo,Int64> > > StackMap;
#endif

 public:
  void printInfos(bool dump_file);
  void dumpJSON(JSONWriter& writer);
  void getInfos(Int64Array&);
  void startProfiling();
  void addEvent(void* address,int overflow_event[MAX_COUNTER],int nb_overflow_event);
  void stopProfiling();
  void reset();
 public:
  void setFunctionDepth(int v);
  void setPeriod(int v);
  void setNbEventBeforeGettingStack(Integer v) { m_nb_event_before_getting_stack = v; }
  Int64 nbEventBeforeGettingStack() const { return m_nb_event_before_getting_stack; }
 private:
  AddrMap m_addr_map;
  FuncMap m_func_map;
  StackMap m_stack_map;
  Int64 m_total_event;
  Int64 m_total_stack;
  Int64 m_counters[MAX_COUNTER];
  Int32 m_current_func_info;
  ProfFuncInfo m_func_info_buffer[MAX_FUNC];
  int m_period;
  Int64 m_nb_event_before_getting_stack;
  int m_function_depth;
  bool m_use_backtrace;
  bool m_use_libunwind;
  MonoFuncAddrGetter* m_mono_func_getter;
  bool m_is_started;
  IFuncInfoProvider* m_default_func_info_provider;
  IFuncInfoProvider* m_libunwind_func_info_provider;
  IFuncInfoProvider* m_backtrace_func_info_provider;
  IFuncInfoProvider* m_func_info_provider;
 protected:
  void _addEvent(void* address,int overflow_event[MAX_COUNTER],int nb_overflow_event,
                 IStackInfoProvider& stack_info,Integer function_depth);
  bool _getFunc(void* addr,FuncAddrInfo& info);
  void _sortFunctions(std::set<ProfFuncInfo*,ProfFuncComparer>& sorted_func);
 private:
  ProfFuncInfo* _getNextFuncInfo();
  ProfFuncInfo& _funcInfoFromIndex(Int32 index) { return m_func_info_buffer[index]; }
  void _storeAddress(void* address,bool is_counter0,int overflow_event[MAX_COUNTER],int nb_overflow_event,
                     bool* do_add,bool* do_stack,bool* func_already_added);
  void _checkNotStarted();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
