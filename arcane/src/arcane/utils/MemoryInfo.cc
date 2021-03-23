// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MemoryInfo.cc                                               (C) 2000-2015 */
/*                                                                           */
/* Collecteur d'informations sur l'usage mémoire.                            */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/utils/String.h"
#include "arcane/utils/Iostream.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/Iterator.h"
#include "arcane/utils/MemoryInfo.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/IStackTraceService.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/ValueConvert.h"

#include <vector>
#include <algorithm>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \page arcanedoc_check_memory Détection des problèmes mémoire.
 *
 Arcane dispose d'un mécanisme permettant de détecter certains problèmes
 mémoire, en particulier:
 - les fuites mémoire
 - les désallocations qui ne correspondent à aucune allocation.
 
 De plus, cela permet d'obtenir des statistiques sur l'utilisation
 mémoire.
 
 \warning Ce mécanisme ne fonctionne actuellement que sur les OS Linux.

 \warning Ce mécanisme ne fonctionne pas lorsque le multi-treading est activé.

 Pour l'activer, il suffit de positionner la variable d'environnement
 ARCANE_CHECK_MEMORY à \c true. Toutes les allocations et désallocations
 sont tracées. Cependant, pour des problèmes de performance, on ne
 conserve et n'affiche la pile d'appel que pour les allocations supérieures
 à une certaine taille. Par défaut, la valeur est de 1Mo (1000000). Il est possible
 de spécifier une autre valeur via la variable d'environnement
 ARCANE_CHECK_MEMORY_BLOCK_SIZE. La variable d'environnement
 ARCANE_CHECK_MEMORY_BLOCK_SIZE_ITERATION permet de spécifier une valeur
 de bloc qui sera utilisé pour la boucle en temps après
 l'initialisation. Cela permet de tracer plus finement les allocations
 durant le calcul que celles qui ont lieu lors de l'initialisation.

 Les appels sont tracés depuis l'appel à ArcaneMain::arcaneInitialize()
 jusqu'à l'appel à ArcaneMain::arcaneFinalize(). Lors de ce dernier appel,
 une liste des blocs alloués qui n'ont pas été désalloués est affiché.

 Il est possible de gérer plus finement le vérificateur mémoire
 via l'interface IMemoryInfo. Cette interface est un singleton accessible
 via la méthode arcaneGlobalMemoryInfo();

 \note INTERNE: Pour l'instant, les éventuelles incohérence entre allocation
 et désallocations sont indiquées sur std::cout. Cela peut poser des problèmes
 de lisibilités en parallèle. A terme, il faudra utiliser ITraceMng, mais
 cela est délicat actuellement car ce mécanisme effectue lui aussi des
 appels mémoire et il difficile de le rendre compatible avec les fonctions
 de débug actuelles.
*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// TODO: les hooks sont obsolètes car non thread-safe.
// Il faudrait utiliser LD_PRELOAD
// d'une bibliothèque qui surcharge malloc(), realloc(), ...,
// puis faire un dlopen sur la libc et appeler dans notre
// bibliothèque les routinies d'allocation de la libc.

#if defined(ARCANE_OS_LINUX)
#define ARCANE_CHECK_MEMORY_USE_MALLOC_HOOK
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifdef ARCANE_CHECK_MEMORY_USE_MALLOC_HOOK
#include <malloc.h>
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" MemoryInfo*
arcaneGlobalTrueMemoryInfo()
{
  static MemoryInfo mem_info;
  return &mem_info;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifdef ARCANE_CHECK_MEMORY_USE_MALLOC_HOOK
static void*(*old_malloc_hook)(size_t,const void*);
static void(*old_free_hook)(void*,const void*);
static void*(*old_realloc_hook)(void* __ptr,
                                size_t __size,
                                __const void*);


static Arcane::Int64 global_nb_malloc = 1;
static void* my_malloc_hook(size_t size,const void* caller);
static void my_free_hook(void* ptr,const void* caller);
static void* my_realloc_hook(void* __ptr,
                             size_t __size,
                             __const void*);
static void _pushHook()
{
  __malloc_hook = old_malloc_hook;
  __free_hook = old_free_hook;
  __realloc_hook = old_realloc_hook;
}
static void _popHook()
{
  __malloc_hook = my_malloc_hook;
  __free_hook = my_free_hook;
  __realloc_hook = my_realloc_hook;
}

static void* my_malloc_hook(size_t size,const void* /*caller*/)
{
  _pushHook();
  void* r = malloc(size);
  ++global_nb_malloc;
  //std::cerr << "*ALLOC = " << r << " s=" << size << '\n';
  arcaneGlobalTrueMemoryInfo()->addInfo(0,r,size);
  arcaneGlobalTrueMemoryInfo()->checkMemory(0,size);
  _popHook();
  return r;
}
static void my_free_hook(void* ptr,const void* /*caller*/)
{
  _pushHook();
  arcaneGlobalTrueMemoryInfo()->removeInfo(0,ptr,true);
  //std::cerr << "*FREE = " << ptr << '\n';
  ++global_nb_malloc;
  free(ptr);
  _popHook();
}
static void* my_realloc_hook(void* ptr,size_t size,const void* /*caller*/)
{
  _pushHook();
  //free(ptr);
  ++global_nb_malloc;
  //std::cerr << "*REFREE = " << ptr << '\n';
  arcaneGlobalTrueMemoryInfo()->removeInfo(0,ptr,true);
  void* r = realloc(ptr,size);
  ++global_nb_malloc;
  //std::cerr << "*REALLOC = " << r << " s=" << size << '\n';
  arcaneGlobalTrueMemoryInfo()->addInfo(0,r,size);
  arcaneGlobalTrueMemoryInfo()->checkMemory(0,size);
  _popHook();
  return r;
}

static void _initMallocHook()
{
  old_malloc_hook = __malloc_hook;
  __malloc_hook = my_malloc_hook;
  old_free_hook = __free_hook;
  __free_hook = my_free_hook;
  old_realloc_hook = __realloc_hook;
  __realloc_hook = my_realloc_hook;
}

static void _restoreMallocHook()
{
  __free_hook = old_free_hook;
  __malloc_hook = old_malloc_hook;
  __realloc_hook = old_realloc_hook;
}
//void (*__malloc_initialize_hook)(void) = my_init_hook;
#else
static void _initMallocHook()
{
}
static void _restoreMallocHook()
{
}
static void _pushHook()
{
}
static void _popHook()
{
}
#endif

static bool global_check_memory = false;
extern "C++" void
arcaneInitCheckMemory()
{
  String s = platform::getEnvironmentVariable("ARCANE_CHECK_MEMORY");
  if (!s.null()){
    global_check_memory = true;
    arcaneGlobalMemoryInfo()->beginCollect();
  }
}

extern "C++" void
arcaneExitCheckMemory()
{
  if (global_check_memory)
    arcaneGlobalMemoryInfo()->endCollect();
  global_check_memory = false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MemoryInfo::
MemoryInfo()
: m_alloc_id(0)
, m_max_allocated(0)
, m_current_allocated(0)
, m_biggest_allocated(0)
, m_info_big_alloc(1000000)
, m_info_biggest_minimal(2000000)
, m_info_peak_minimal(10000000)
, m_iteration(0)
, m_trace(0)
, m_display_max_alloc(true)
, m_in_display(false)
, m_is_first_collect(true)
, m_is_stack_trace_active(true)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MemoryInfo::
~MemoryInfo()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MemoryInfo::
setOwner(const void* owner,const TraceInfo& ti)
{
  MemoryTraceInfoMap::iterator i = m_owner_infos.find(owner);
  if (i!=m_owner_infos.end()){
    i->second = ti;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MemoryInfo::
addInfo(const void* owner,const void* ptr,Int64 size)
{
  //NOTE: Cette méthode doit être réentrente.
  //TODO: verifier owner present.
  MemoryInfoMap::const_iterator i = m_infos.find(ptr);
  String stack_value;
  //cout << "** ADD: " << ptr << '\n';
  if (i==m_infos.end()){
    MemoryInfoChunk c(owner,size,m_alloc_id,m_iteration);
    //if (size>5000)
    //std::cout << " ALLOC size=" << size << " ptr=" << ptr << '\n';
    if (size>=m_info_big_alloc && m_is_stack_trace_active){
      IStackTraceService* s = platform::getStackTraceService();
      if (s){
        stack_value = s->stackTrace(2).toString();
        c.setStackTrace(stack_value);
      }
    }
    m_infos.insert(MemoryInfoMap::value_type(ptr,c));
  }
  else{
    //cout << "** OLD VALUE file=" << i->second->m_file << " line=" << i->second->m_line;
    //if (i->second->m_name)
    //cout << " name=" << i->second->m_name << '\n';
    cout << "** addInfo() ALREADY IN MAP VALUE=" << ptr << " size=" << size << '\n';
    //throw FatalErrorException("MemoryInfo::addInfo() pointer already in map");
  }
  m_current_allocated += size;
  ++m_alloc_id;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MemoryInfo::
createOwner(const void* owner,const TraceInfo& trace_info)
{
  MemoryTraceInfoMap::iterator i = m_owner_infos.find(owner);
  if (i==m_owner_infos.end()){
    //cout << "** CREATE OWNER " << owner << "\n";
    m_owner_infos.insert(MemoryTraceInfoMap::value_type(owner,trace_info));
  }
  else{
    cout << "** createOwner() ALREADY IN MAP VALUE=" << owner << '\n';
    //throw FatalErrorException("MemoryInfo::createOwner() owner already in map");
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MemoryInfo::
addInfo(const void* owner,const void* ptr,Int64 size,const void* /*old_ptr*/)
{
  addInfo(owner,ptr,size);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MemoryInfo::
changeOwner(const void* new_owner,const void* ptr)
{
  MemoryTraceInfoMap::iterator i_owner = m_owner_infos.find(new_owner);
  if (i_owner==m_owner_infos.end()){
    cerr << "** UNKNOWN NEW OWNER " << new_owner << '\n';
    throw FatalErrorException("MemoryInfo::changeOwner() unknown new owner");
  }
  if (ptr){
    MemoryInfoMap::iterator i = m_infos.find(ptr);
    if (i==m_infos.end()){
      cout << "** BAD VALUE=" << ptr << '\n';
      throw FatalErrorException("MemoryInfo::changeOwner() pointer not in map");
    }
    else{
      i->second.setOwner(i_owner->first);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MemoryInfo::
_removeOwner(const void* owner)
{
  MemoryTraceInfoMap::iterator i = m_owner_infos.find(owner);
  if (i!=m_owner_infos.end()){
    //cout << "** REMOVE OWNER " << owner << "\n";
    m_owner_infos.erase(i);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MemoryInfo::
removeOwner(const void* owner)
{
  _removeOwner(owner);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MemoryInfo::
removeInfo(const void* owner,const void* ptr,bool can_fail)
{
  if (!ptr)
    return;
  MemoryInfoMap::iterator i = m_infos.find(ptr);
  //cout << "** REMOVE: " << ptr << '\n';
  if (i==m_infos.end()){
    if (can_fail)
      return;
    cout << "MemoryInfo::removeInfo() pointer not in map";
    //throw FatalErrorException("MemoryInfo::removeInfo() pointer not in map");
  }
  else{
    MemoryInfoChunk& chunk = i->second;
    Int64 size = chunk.size();
    //if (size>5000)
    //std::cout << " FREE size=" << size << " ptr=" << ptr << '\n';
    _removeMemory(owner,size);
    m_infos.erase(i);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class MemoryInfo::MemoryInfoSorter
{
 public:
  MemoryInfoSorter() : m_size(0), m_alloc_id(-1), m_iteration(0), m_ptr(0), m_owner(0) {}
  MemoryInfoSorter(Int64 size,Int64 alloc_id,Integer iteration, const void* ptr,
                   const void* owner,const String& stack_trace) 
    : m_size(size), m_alloc_id(alloc_id), m_iteration(iteration), m_ptr(ptr)
    , m_owner(owner), m_stack_trace(stack_trace) {}
 public:
  Int64 m_size;
  Int64 m_alloc_id;
  Integer m_iteration;
  const void* m_ptr;
  const void* m_owner;
  String m_stack_trace;
 public:
  bool operator<(const MemoryInfoSorter& rhs) const
  {
    return m_size > rhs.m_size;
  }
};


class MemoryInfo::TracePrinter
{
 public:
  TracePrinter(const TraceInfo* ti) : m_trace_info(ti) {}
  void print(std::ostream& o) const
  {
    if (m_trace_info){
      o << " name=" << m_trace_info->name()
        << " file=" << m_trace_info->file()
        << " line=" << m_trace_info->line();
    }
  }
 private:
  const TraceInfo* m_trace_info;
};

std::ostream&
operator<<(std::ostream& o,const MemoryInfo::TracePrinter& tp)
{
  tp.print(o);
  return o;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MemoryInfo::
printInfos(std::ostream& ostr)
{
  bool is_collecting = global_check_memory;
  // Comme _printInfos() utilise m_infos et peut allouer de la mémoire ce
  // qui va provoquer une modification de m_infos is on est en cours
  // de collection, on désactive les hooks le temps de l'appel.
  if (is_collecting)
    _pushHook();
  try{
    _printInfos(ostr);
  }
  catch(...){
    if (is_collecting)
      _popHook();
    throw;
  }
  if (is_collecting)
    _popHook();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MemoryInfo::
_printInfos(std::ostream& ostr)
{
  Int64 total_size = 0;
  ostr << "MemoryInfos: " << m_infos.size() << '\n';

  size_t nb_chunk = m_infos.size();
  std::vector<MemoryInfoSorter> sorted_chunk;
  sorted_chunk.reserve(nb_chunk);
  for( auto i : m_infos ){
    sorted_chunk.push_back(MemoryInfoSorter(i.second.size(),i.second.allocId(),
                                            i.second.iteration(),
                                            i.first,i.second.owner(),i.second.stackTrace()));
  }
  std::sort(sorted_chunk.begin(),sorted_chunk.end());

  for( auto i : sorted_chunk){
    const void* v = i.m_ptr;
    const void* owner = i.m_owner;
    Int64 size = i.m_size;
    const TraceInfo* ti = 0;
    {
      MemoryTraceInfoMap::iterator i_owner = m_owner_infos.find(owner);
      if (i_owner!=m_owner_infos.end()){
        ti = &i_owner->second;
      }
    }
    total_size += size;
    if (size>=m_info_big_alloc){
      ostr << " Remaining: " << v << " size=" << size << " id=" << i.m_alloc_id
           << " iteration=" << i.m_iteration
           << TracePrinter(ti) << " trace=" << i.m_stack_trace << '\n';
    }
  }
  ostr << "Total size=" << total_size;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MemoryInfo::
printAllocatedMemory(std::ostream& ostr,Integer iteration)
{
  ostr << " INFO_ALLOCATION: current= " << m_current_allocated
       << " ITERATION= " << iteration
       << " NB_CHUNK=" << m_infos.size()
       << " ID=" << m_alloc_id
       << '\n';
  for( ConstIterT<MemoryInfoMap> i(m_infos); i(); ++i ){
    const MemoryInfoChunk& mi = i->second;
    if (mi.iteration()!=iteration)
      continue;
    Int64 size = mi.size();
    if (size>=m_info_big_alloc){
      ostr << " Allocated: " << " iteration=" << iteration
           << " size=" << size << " id=" << mi.allocId()
           << " trace=" << mi.stackTrace() << '\n';
    }
  }
  for( ConstIterT<MemoryInfoMap> i(m_infos); i(); ++i ){
    const MemoryInfoChunk& mi = i->second;
    if (mi.iteration()!=iteration)
      continue;
    Int64 size = mi.size();
    if (size<m_info_big_alloc){
      ostr << " Allocated: " << " iteration=" << iteration
           << " size=" << size << " id=" << mi.allocId() << '\n';
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MemoryInfo::
setTraceMng(ITraceMng* trace)
{
  m_trace = trace;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MemoryInfo::
beginCollect()
{
  if (m_is_first_collect){
    String s = platform::getEnvironmentVariable("ARCANE_CHECK_MEMORY_BLOCK_SIZE");
    if (!s.null()){
      Int64 block_size = 0;
      bool is_bad = builtInGetValue(block_size,s);
      if (!is_bad && block_size>2){
        m_info_big_alloc = block_size;
        m_info_biggest_minimal = block_size * 2;
        m_info_peak_minimal = block_size * 10;
      }
      if (m_trace)
        m_trace->info() << " BLOCK SIZE '" << s;
    }
    m_is_first_collect = false;
  }
  _initMallocHook();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MemoryInfo::
endCollect()
{
  _restoreMallocHook();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool MemoryInfo::
isCollecting() const
{
  return global_check_memory;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MemoryInfo::
checkMemory(const void* owner,Int64 size)
{
  if (m_current_allocated>m_max_allocated){
    m_max_allocated = m_current_allocated;
    if (m_display_max_alloc && m_max_allocated>m_info_peak_minimal && size>5000 && m_trace && !m_in_display){
      m_in_display = true;
      m_trace->info() << "Memory:PEAK_MEM: iteration=" << m_iteration
                      << " max allocation reached: max="
                      << m_max_allocated << " size=" << size
                      << " id=" << m_alloc_id << " "
                      << TracePrinter(_getTraceInfo(owner));
      m_in_display = false;
    }
  }
  if (size>m_biggest_allocated){
    m_biggest_allocated = size;
    if (m_display_max_alloc && m_biggest_allocated>m_info_biggest_minimal && m_trace && !m_in_display){
      m_in_display = true;
      m_trace->info() << "Memory:PEAK_ALLOC: biggest allocation : " << size << " "
                    << " id=" << m_alloc_id << " "
                    << TracePrinter(_getTraceInfo(owner));
      m_in_display = false;
    }
  }
  if (m_info_big_alloc>0 && size>m_info_big_alloc){
    if (m_display_max_alloc && m_trace && !m_in_display){
      m_in_display = true;
      String stack_value;
      IStackTraceService* s = platform::getStackTraceService();
      if (s){
        stack_value= s->stackTrace(2).toString();
      }
#if 0
      m_trace->info() << "Memory:BIG_ALLOC: iteration=" << m_iteration
                      << " big alloc= " << size
                      << " id=" << m_alloc_id
                      << " current=" << m_current_allocated
      //<< " " << TracePrinter(_getTraceInfo(owner))
                      << " stack=" << stack_value;
#endif
      m_in_display = false;
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MemoryInfo::
_removeMemory(const void* /*owner*/,Int64 size)
{
  m_current_allocated -= size;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TraceInfo* MemoryInfo::
_getTraceInfo(const void* owner)
{
  MemoryTraceInfoMap::iterator i = m_owner_infos.find(owner);
  if (i==m_owner_infos.end())
    return 0;
  return &i->second;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MemoryInfo::
visitAllocatedBlocks(IFunctorWithArgumentT<const MemoryInfoChunk&>* functor) const
{
  if (!functor)
    return;
  for( ConstIterT<MemoryInfoMap> i(m_infos); i(); ++i ){
    const MemoryInfoChunk& mic = i->second;
    functor->executeFunctor(mic);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" IMemoryInfo*
arcaneGlobalMemoryInfo()
{
  return arcaneGlobalTrueMemoryInfo();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
