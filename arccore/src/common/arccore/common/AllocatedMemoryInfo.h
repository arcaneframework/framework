// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AllocatedMemoryInfo.h                                       (C) 2000-2025 */
/*                                                                           */
/* Information about an allocated memory region.                             */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_COMMON_ALLOCATEDMEMORYINFO_H
#define ARCCORE_COMMON_ALLOCATEDMEMORYINFO_H
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
 * \brief Information about an allocated memory region.
 */
class AllocatedMemoryInfo
{
 public:

  AllocatedMemoryInfo() = default;
  explicit AllocatedMemoryInfo(void* base_address)
  : m_base_address(base_address)
  {}
  AllocatedMemoryInfo(void* base_address, Int64 size)
  : m_base_address(base_address)
  , m_size(size)
  , m_capacity(size)
  {}
  AllocatedMemoryInfo(void* base_address, Int64 size, Int64 capacity)
  : m_base_address(base_address)
  , m_size(size)
  , m_capacity(capacity)
  {}

  //! Address of the start of the allocated region.
  void* baseAddress() const { return m_base_address; }
  //! Size in bytes of the used memory region. (-1) if unknown
  Int64 size() const { return m_size; }
  //! Size in bytes of the allocated memory region. (-1) if unknown
  Int64 capacity() const { return m_capacity; }

 public:

  void* m_base_address = nullptr;
  Int64 m_size = -1;
  Int64 m_capacity = -1;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
