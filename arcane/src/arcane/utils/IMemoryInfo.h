// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMemoryInfo.h                                               (C) 2000-2018 */
/*                                                                           */
/* Interface for a memory usage information collector.                       */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_IMEMORYINFO_H
#define ARCANE_UTILS_IMEMORYINFO_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/String.h"
#include "arcane/utils/IFunctorWithArgument.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Information about an allocated chunk.
 */
class MemoryInfoChunk
{
 public:

  MemoryInfoChunk()
  : m_owner(0)
  , m_size(0)
  , m_alloc_id(0)
  , m_iteration(0)
  {}
  MemoryInfoChunk(const void* aowner, Int64 asize, Int64 alloc_id, Integer aiteration)
  : m_owner(aowner)
  , m_size(asize)
  , m_alloc_id(alloc_id)
  , m_iteration(aiteration)
  {}

 public:

  const void* owner() const { return m_owner; }
  Int64 size() const { return m_size; }
  Int64 allocId() const { return m_alloc_id; }
  Integer iteration() const { return m_iteration; }
  const String& stackTrace() const { return m_stack_trace; }

 public:

  void setOwner(const void* o) { m_owner = o; }
  void setStackTrace(const String& st) { m_stack_trace = st; }

 private:

  const void* m_owner;
  Int64 m_size;
  Int64 m_alloc_id;
  Integer m_iteration;
  String m_stack_trace;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Interface for a memory usage information collector.
 */
class IMemoryInfo
{
 protected:

  IMemoryInfo() {}

 public:

  //! Releases resources
  virtual ~IMemoryInfo() {}

 public:

  //! Creates a reference on \a owner with the trace info \a trace_info
  virtual void createOwner(const void* owner, const TraceInfo& trace_info) = 0;

  //! Modifies the info of the reference \a owner
  virtual void setOwner(const void* owner, const TraceInfo& new_info) = 0;

  //! Removes the reference on \a owner
  virtual void removeOwner(const void* owner) = 0;

 public:

  virtual void addInfo(const void* owner, const void* ptr, Int64 size) = 0;

  virtual void addInfo(const void* owner, const void* ptr, Int64 size, const void* old_ptr) = 0;

  virtual void changeOwner(const void* new_owner, const void* ptr) = 0;

  virtual void removeInfo(const void* owner, const void* ptr, bool can_fail = false) = 0;

  virtual void printInfos(std::ostream& ostr) = 0;

 public:

  virtual void beginCollect() = 0;
  virtual void endCollect() = 0;
  virtual bool isCollecting() const = 0;

 public:

  //! Sets the current iteration number.
  virtual void setIteration(Integer iteration) = 0;

  virtual void printAllocatedMemory(std::ostream& ostr, Integer iteration) = 0;

  //! Sets the ITraceMng for messages.
  virtual void setTraceMng(ITraceMng* msg) = 0;

  /*!
   * \brief Indicates whether call stack saving is active.
   *
   * If \a is_active is true, it activates tracing the call stack of allocations.
   * Tracing is conditional on the value of stackTraceMinAllocSize().
   */
  virtual void setKeepStackTrace(bool is_active) = 0;

  //! Indicates if call stack saving is enabled.
  virtual bool keepStackTrace() const = 0;

  /*!
   * \brief Sets the minimum size of allocations whose call stack is traced.
   *
   * For all allocations above \a alloc_size,
   * the call stack is preserved in order to identify memory leaks. The memory and CPU cost of preserving
   * a call stack is significant, and it is therefore not recommended to set a value too low (below 1000) for \a alloc_size.
   * Call stack preservation is disabled if \a keepStackTrace() equals \a false.
   */
  virtual void setStackTraceMinAllocSize(Int64 alloc_size) = 0;

  //! Minimum size of allocations whose call stack is traced.
  virtual Int64 stackTraceMinAllocSize() const = 0;

  //! Visitor over all allocated blocks
  virtual void visitAllocatedBlocks(IFunctorWithArgumentT<const MemoryInfoChunk&>* functor) const = 0;

  virtual Int64 nbAllocation() const = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_UTILS_EXPORT IMemoryInfo*
arcaneGlobalMemoryInfo();

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
