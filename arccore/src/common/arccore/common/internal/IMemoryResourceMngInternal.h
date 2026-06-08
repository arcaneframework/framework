// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMemoryResourceMngInternal.h                                (C) 2000-2025 */
/*                                                                           */
/* Internal part of Arcane's 'IMemoryResourceMng'.                           */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_COMMON_INTERNAL_IMEMORYRESOURCEMNGINTERNAL_H
#define ARCCORE_COMMON_INTERNAL_IMEMORYRESOURCEMNGINTERNAL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/common/internal/IMemoryCopier.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Internal part of Arcane's 'IMemoryResourceMng'.
 */
class ARCCORE_COMMON_EXPORT IMemoryResourceMngInternal
{
 public:

  virtual ~IMemoryResourceMngInternal() = default;

 public:

  virtual void copy(ConstMemoryView from, eMemoryResource from_mem,
                    MutableMemoryView to, eMemoryResource to_mem, const RunQueue* queue) = 0;

 public:

  //! Sets the allocator for resource \a r
  virtual void setAllocator(eMemoryResource r, IMemoryAllocator* allocator) = 0;

  //! Sets the memory pool for resource \a r
  virtual void setMemoryPool(eMemoryResource r, IMemoryPool* pool) = 0;

  //! Sets the copying instance.
  virtual void setCopier(IMemoryCopier* copier) = 0;

  //! Indicates if an accelerator is available.
  virtual void setIsAccelerator(bool v) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
