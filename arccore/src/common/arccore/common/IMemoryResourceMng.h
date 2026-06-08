// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMemoryResourceMng.h                                        (C) 2000-2025 */
/*                                                                           */
/* Memory resource management for CPUs and accelerators.                     */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_COMMON_IMEMORYRESOURCEMNG_H
#define ARCCORE_COMMON_IMEMORYRESOURCEMNG_H
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
 * \internal
 * \brief Memory resource management for CPUs and accelerators.
 */
class ARCCORE_COMMON_EXPORT IMemoryResourceMng
{
 public:

  virtual ~IMemoryResourceMng() = default;

 public:

  /*!
   * \brief Memory allocator for resource \a r.
   *
   * Throws an exception if no allocator for resource \a v exists.
   */
  virtual IMemoryAllocator* getAllocator(eMemoryResource r) = 0;

  /*!
   * \brief Memory allocator for resource \a r.
   *
   * If no allocator for resource \a v exists, throws an
   * exception if \a throw_if_not_found is true or returns \a nullptr
   * if \a throw_if_not_found is false.
   */
  virtual IMemoryAllocator* getAllocator(eMemoryResource r, bool throw_if_not_found) = 0;

  /*!
   * \brief Memory pool for resource \a r.
   *
   * Returns the memory pool associated with resource \a v or \a nullptr
   * if there is none.
   */
  virtual IMemoryPool* getMemoryPoolOrNull(eMemoryResource r) = 0;

 public:

  //! Internal interface
  virtual IMemoryResourceMngInternal* _internal() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
