// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MemoryPool.h                                                (C) 2000-2026 */
/*                                                                           */
/* Class to manage a list of allocated zones.                                */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_COMMON_INTERNAL_MEMORYPOOL_H
#define ARCCORE_COMMON_INTERNAL_MEMORYPOOL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/common/IMemoryPool.h"

#include <memory>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Interface for an allocator for a MemoryPool.
 *
 * This interface works like a malloc/free, except that you
 * must provide the allocated size for a block when freeing it.
 * The user of this interface must therefore manage the preservation of this
 * information.
 */
class ARCCORE_COMMON_EXPORT IMemoryPoolAllocator
{
 public:

  virtual ~IMemoryPoolAllocator() = default;

 public:

  //! Allocates a block for \a size bytes
  virtual void* allocateMemory(Int64 size) = 0;
  //! Frees the block located at address \a address containing \a size bytes
  virtual void freeMemory(void* address, Int64 size) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Class to manage a list of allocated zones.
 *
 * This class uses a reference semantics.
 *
 * The allocator passed as an argument to the constructor must remain valid
 * throughout the life of the instance.
 */
class ARCCORE_COMMON_EXPORT MemoryPool
: public IMemoryPool
, public IMemoryPoolAllocator
{
  class Impl;

 public:

  explicit MemoryPool(IMemoryPoolAllocator* allocator, const String& name);
  ~MemoryPool() override;

 public:

  MemoryPool(const MemoryPool&) = delete;
  MemoryPool(MemoryPool&&) = delete;
  MemoryPool& operator=(const MemoryPool&) = delete;
  MemoryPool& operator=(MemoryPool&&) = delete;

 public:

  void* allocateMemory(Int64 size) override;
  void freeMemory(void* ptr, Int64 size) override;
  void dumpStats(std::ostream& ostr);
  void dumpFreeMap(std::ostream& ostr);
  String name() const;

  //! Implementation of IMemoryPool
  //@{
  void setMaxCachedBlockSize(Int32 v) override;
  void freeCachedMemory() override;
  Int64 totalAllocated() const override;
  Int64 totalCached() const override;
  //@}

 private:

  std::unique_ptr<Impl> m_p;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
