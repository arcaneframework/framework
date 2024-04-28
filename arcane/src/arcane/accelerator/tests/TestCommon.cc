// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#include "arcane/accelerator/AcceleratorGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#if defined(ARCANE_HAS_CUDA)
extern "C" ARCANE_EXPORT void
arcaneRegisterAcceleratorRuntimecuda();
#endif

#if defined(ARCANE_HAS_HIP)
extern "C" ARCANE_EXPORT void
arcaneRegisterAcceleratorRuntimehip();
#endif

#if defined(ARCANE_HAS_SYCL)
extern "C" ARCANE_EXPORT void
arcaneRegisterAcceleratorRuntimesycl();
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" void
arcaneRegisterDefaultAcceleratorRuntime()
{
#ifdef ARCANE_HAS_CUDA
  arcaneRegisterAcceleratorRuntimecuda();
#elif defined(ARCANE_HAS_HIP)
  arcaneRegisterAcceleratorRuntimehip();
#elif defined(ARCANE_HAS_SYCL)
  arcaneRegisterAcceleratorRuntimesycl();
#endif
}

extern "C++" Arcane::Accelerator::eExecutionPolicy
arcaneGetDefaultExecutionPolicy()
{
#if defined(ARCANE_HAS_CUDA)
  return Arcane::Accelerator::eExecutionPolicy::CUDA;
#elif defined(ARCANE_HAS_HIP)
  return Arcane::Accelerator::eExecutionPolicy::HIP;
#elif defined(ARCANE_HAS_SYCL)
  return Arcane::Accelerator::eExecutionPolicy::SYCL;
#endif
  return Arcane::Accelerator::eExecutionPolicy::Sequential;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
