// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SyclAccelerator.h                                           (C) 2000-2025 */
/*                                                                           */
/* Backend 'SYCL' pour les accélérateurs.                                    */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ACCELERATOR_SYCL_SYCLACCELERATOR_H
#define ARCANE_ACCELERATOR_SYCL_SYCLACCELERATOR_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/common/CommonGlobal.h"

#include <sycl/sycl.hpp>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifdef ARCANE_COMPONENT_arcane_sycl
#define ARCANE_SYCL_EXPORT ARCCORE_EXPORT
#else
#define ARCANE_SYCL_EXPORT ARCCORE_IMPORT
#endif

namespace Arcane::Accelerator::Sycl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_SYCL_EXPORT IMemoryAllocator*
getSyclMemoryAllocator();

extern "C++" ARCANE_SYCL_EXPORT IMemoryAllocator*
getSyclDeviceMemoryAllocator();

extern "C++" ARCANE_SYCL_EXPORT IMemoryAllocator*
getSyclUnifiedMemoryAllocator();

extern "C++" ARCANE_SYCL_EXPORT IMemoryAllocator*
getSyclHostPinnedMemoryAllocator();

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator::Sycl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
