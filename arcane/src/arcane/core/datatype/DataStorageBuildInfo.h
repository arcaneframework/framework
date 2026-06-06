// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IDataStorageFactory.h                                       (C) 2000-2025 */
/*                                                                           */
/* Information to construct an instance of 'IData'.                          */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_DATATYPE_DATASTORAGEBUILDINFO_H
#define ARCANE_CORE_DATATYPE_DATASTORAGEBUILDINFO_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Information to construct an instance of 'IData'.
 */
class ARCANE_CORE_EXPORT DataStorageBuildInfo
{
 public:

  explicit DataStorageBuildInfo(ITraceMng* tm)
  : m_trace_mng(tm)
  {}

 public:

  ITraceMng* traceMng() const { return m_trace_mng; }
  IMemoryAllocator* memoryAllocator() const { return m_memory_allocator; }
  void setMemoryAllocator(IMemoryAllocator* a) { m_memory_allocator = a; }

 private:

  ITraceMng* m_trace_mng;
  IMemoryAllocator* m_memory_allocator = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
