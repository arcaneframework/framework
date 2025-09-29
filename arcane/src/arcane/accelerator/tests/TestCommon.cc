// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#include "arcane/accelerator/AcceleratorGlobal.h"
#include "arcane/accelerator/core/internal/RegisterRuntimeInfo.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#if defined(ARCANE_HAS_CUDA)
extern "C" ARCANE_EXPORT void
arcaneRegisterAcceleratorRuntimecuda(Arcane::Accelerator::RegisterRuntimeInfo& init_info);
#endif

#if defined(ARCANE_HAS_HIP)
extern "C" ARCANE_EXPORT void
arcaneRegisterAcceleratorRuntimehip(Arcane::Accelerator::RegisterRuntimeInfo& init_info);
#endif

#if defined(ARCANE_HAS_SYCL)
extern "C" ARCANE_EXPORT void
arcaneRegisterAcceleratorRuntimesycl(Arcane::Accelerator::RegisterRuntimeInfo& init_info);
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" void
arcaneRegisterDefaultAcceleratorRuntime()
{
  Arcane::Accelerator::RegisterRuntimeInfo init_info;
  init_info.setVerbose(true);
#ifdef ARCANE_HAS_CUDA
  arcaneRegisterAcceleratorRuntimecuda(init_info);
#elif defined(ARCANE_HAS_HIP)
  arcaneRegisterAcceleratorRuntimehip(init_info);
#elif defined(ARCANE_HAS_SYCL)
  arcaneRegisterAcceleratorRuntimesycl(init_info);
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
