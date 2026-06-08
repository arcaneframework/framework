// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MemoryUtilsInternal.h                                       (C) 2000-2025 */
/*                                                                           */
/* Internal memory management utility functions for Arcane.                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_COMMON_INTERNAL_MEMORYUTILSINTERNAL_H
#define ARCCORE_COMMON_INTERNAL_MEMORYUTILSINTERNAL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/common/CommonGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MemoryUtils
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Sets the memory resource manager for data.
 *
 * The manager must remain valid throughout the program execution.
 *
 * Returns the old manager.
 */
extern "C++" ARCCORE_COMMON_EXPORT IMemoryRessourceMng*
setDataMemoryResourceMng(IMemoryRessourceMng* mng);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Memory resource manager for data.
 *
 * It is guaranteed that the alignment is at least that returned by
 * AlignedMemoryAllocator::Simd().
 */
extern "C++" ARCCORE_COMMON_EXPORT IMemoryRessourceMng*
getDataMemoryResourceMng();

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Sets the specific allocator for accelerators.
 *
 * Returns the previously used allocator. The specified allocator must remain
 * valid throughout the application's lifetime.
 */
extern "C++" ARCCORE_COMMON_EXPORT IMemoryAllocator*
setAcceleratorHostMemoryAllocator(IMemoryAllocator* a);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Sets the memory resource used for the data memory allocator.
 *
 * \sa getDefaultDataMemoryResource();
 */
extern "C++" ARCCORE_COMMON_EXPORT void
setDefaultDataMemoryResource(eMemoryResource mem_resource);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::MemoryUtils

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
