// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMemoryPool.h                                               (C) 2000-2026 */
/*                                                                           */
/* Interface of a memory pool.                                               */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_COMMON_IMEMORYPOOL_H
#define ARCCORE_COMMON_IMEMORYPOOL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/common/CommonGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Interface of a memory pool.
 */
class ARCCORE_COMMON_EXPORT IMemoryPool
{
 public:

  virtual ~IMemoryPool() = default;

 public:

  /*!
   * \brief Sets the byte size from which a block is not kept in the cache.
   *
   * This method can only be called if there are no blocks in the cache.
   */
  virtual void setMaxCachedBlockSize(Int32 v) = 0;

  //! Frees the memory in the cache
  virtual void freeCachedMemory() = 0;

  //! Total size (in bytes) allocated in the memory pool
  virtual Int64 totalAllocated() const = 0;

  //! Total size (in bytes) in the cache
  virtual Int64 totalCached() const = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
