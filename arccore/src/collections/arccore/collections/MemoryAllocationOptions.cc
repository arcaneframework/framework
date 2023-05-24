// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MemoryAllocationOptions.cc                                  (C) 2000-2023 */
/*                                                                           */
/* Options pour configurer les allocations.                                  */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/collections/MemoryAllocationOptions.h"
#include "arccore/collections/MemoryAllocationArgs.h"
#include "arccore/collections/ArrayDebugInfo.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String MemoryAllocationArgs::
arrayName() const
{
  return (m_debug_info) ? m_debug_info->name() : String();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MemoryAllocationOptions::
MemoryAllocationOptions(const MemoryAllocationOptions& rhs)
: m_allocator(rhs.m_allocator)
, m_memory_location_hint(rhs.m_memory_location_hint)
, m_device(rhs.m_device)
, m_debug_info(rhs.m_debug_info)
{
  if (m_debug_info)
    m_debug_info->addReference();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MemoryAllocationOptions& MemoryAllocationOptions::
operator=(const MemoryAllocationOptions& rhs)
{
  if (&rhs == this)
    return (*this);
  if (m_debug_info)
    m_debug_info->removeReference();
  m_allocator = rhs.m_allocator;
  m_memory_location_hint = rhs.m_memory_location_hint;
  m_device = rhs.m_device;
  m_debug_info = rhs.m_debug_info;
  if (m_debug_info)
    m_debug_info->addReference();
  return (*this);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MemoryAllocationOptions::
~MemoryAllocationOptions()
{
  if (m_debug_info)
    m_debug_info->removeReference();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MemoryAllocationArgs MemoryAllocationOptions::
allocationArgs() const
{
  MemoryAllocationArgs x;
  x.setMemoryLocationHint(m_memory_location_hint);
  x.setDevice(m_device);
  x.setDebugInfo(m_debug_info);
  return x;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MemoryAllocationOptions::
setArrayName(const String& name)
{
  if (!m_debug_info) {
    m_debug_info = new ArrayDebugInfo();
    m_debug_info->addReference();
  }
  m_debug_info->setName(name);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String MemoryAllocationOptions::
arrayName() const
{
  return (m_debug_info) ? m_debug_info->name() : String{};
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
