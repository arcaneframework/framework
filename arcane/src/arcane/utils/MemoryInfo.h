// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MemoryInfo.h                                                (C) 2000-2015 */
/*                                                                           */
/* Collecteur d'informations sur l'usage mémoire.                            */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_MEMORYINFO_H
#define ARCANE_UTILS_MEMORYINFO_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/IMemoryInfo.h"
#include "arcane/utils/TraceInfo.h"

#include <map>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Collecteur d'informations sur l'usage mémoire.
 */
class ARCANE_UTILS_EXPORT MemoryInfo
: public IMemoryInfo
{
 public:
  class MemoryInfoSorter;
  class TracePrinter;
 public:
 
  MemoryInfo();
  virtual ~MemoryInfo();

 public:

  virtual void createOwner(const void* owner,const TraceInfo& trace_info);

  virtual void setOwner(const void* owner,const TraceInfo& new_info);

  virtual void removeOwner(const void* owner);

 public:

  virtual void addInfo(const void* owner,const void* ptr,Int64 size);

  virtual void addInfo(const void* owner,const void* ptr,Int64 size,const void* old_ptr);

  virtual void removeInfo(const void* owner,const void* ptr,bool can_fail);

  virtual void changeOwner(const void* new_owner,const void* ptr);

  virtual void printInfos(std::ostream& ostr);

  virtual void setIteration(Integer iteration)
    { m_iteration = iteration; }
  virtual void printAllocatedMemory(std::ostream& ostr,Integer iteration);
  virtual void setTraceMng(ITraceMng* trace);

 public:

  virtual void beginCollect();
  virtual void endCollect();
  virtual bool isCollecting() const;

 public:

  virtual void setKeepStackTrace(bool is_active) { m_is_stack_trace_active = is_active; }
  virtual bool keepStackTrace() const { return m_is_stack_trace_active; }

  virtual void setStackTraceMinAllocSize(Int64 alloc_size) { m_info_big_alloc = alloc_size; }
  virtual Int64 stackTraceMinAllocSize() const { return m_info_big_alloc; }

  virtual void visitAllocatedBlocks(IFunctorWithArgumentT<const MemoryInfoChunk&>* functor) const;

  virtual Int64 nbAllocation() const { return m_alloc_id; }

 public:

  void checkMemory(const void* owner,Int64 size);

 private:

  typedef std::map<const void*,MemoryInfoChunk> MemoryInfoMap;
  typedef std::map<const void*,TraceInfo> MemoryTraceInfoMap;
  
  MemoryInfoMap m_infos;
  MemoryTraceInfoMap m_owner_infos;

  Int64 m_alloc_id;
  Int64 m_max_allocated;
  Int64 m_current_allocated;
  Int64 m_biggest_allocated;
  Int64 m_info_big_alloc;
  Int64 m_info_biggest_minimal;
  Int64 m_info_peak_minimal;
  Integer m_iteration;
  ITraceMng* m_trace;
  bool m_display_max_alloc;
  bool m_in_display;
  bool m_is_first_collect;
  bool m_is_stack_trace_active;

 private:

  void _removeOwner(const void* owner);
  void _addMemory(const void* owner,Int64 size,const String& stack_value);
  void _removeMemory(const void* owner,Int64 size);
  TraceInfo* _getTraceInfo(const void* owner);
  void _printInfos(std::ostream& ostr);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_UTILS_EXPORT MemoryInfo*
arcaneGlobalTrueMemoryInfo();
extern "C++" ARCANE_UTILS_EXPORT void
arcaneInitCheckMemory();
extern "C++" ARCANE_UTILS_EXPORT void
arcaneExitCheckMemory();

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

