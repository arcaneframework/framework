// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DeviceInfo.h                                                (C) 2000-2025 */
/*                                                                           */
/* Information about an accelerator's memory.                                */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_COMMON_ACCELERATOR_DEVICEMEMORYINFO_H
#define ARCCORE_COMMON_ACCELERATOR_DEVICEMEMORYINFO_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/common/accelerator/CommonAcceleratorGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Accelerator memory information.
 */
class ARCCORE_COMMON_EXPORT DeviceMemoryInfo
{
 public:

  //! Amount of free memory (in bytes)
  Int64 freeMemory() const { return m_free_memory; }

  //! Total amount of memory (in bytes)
  Int64 totalMemory() const { return m_total_memory; }

 public:

  void setFreeMemory(Int64 v) { m_free_memory = v; }
  void setTotalMemory(Int64 v) { m_total_memory = v; }

 private:

  Int64 m_free_memory = 0;
  Int64 m_total_memory = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
