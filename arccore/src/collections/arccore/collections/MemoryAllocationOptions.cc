﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MemoryAllocationOptions.cc                                  (C) 2000-2024 */
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

MemoryAllocationArgs MemoryAllocationOptions::
allocationArgs(RunQueue* queue) const
{
  MemoryAllocationArgs x;
  x.setMemoryLocationHint(m_memory_location_hint);
  x.setDevice(m_device);
  x.setDebugInfo(m_debug_info);
  x.setRunQueue(queue);
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

void MemoryAllocationOptions::
_addDebugReference()
{
  m_debug_info->addReference();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MemoryAllocationOptions::
_removeDebugReference()
{
  m_debug_info->removeReference();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
