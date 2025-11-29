// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Cupti.h                                                     (C) 2000-2025 */
/*                                                                           */
/* Intégration de CUPTI.                                                     */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ACCELERATOR_CUDA_RUNTIME_INTERNAL_CUPTI_H
#define ARCANE_ACCELERATOR_CUDA_RUNTIME_INTERNAL_CUPTI_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/FixedArray.h"

#include "arcane/accelerator/cuda/CudaAccelerator.h"

#ifdef ARCANE_HAS_CUDA_CUPTI
#include <cuda.h>
#include <cupti.h>
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator::Cuda
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe singleton pour gérer CUPTI.
 */
class CuptiInfo
{
 public:

  void init(Int32 level, bool do_print)
  {
    m_profiling_level = level;
    m_do_print = do_print;
  }
  bool isActive() const { return m_is_active; }

 public:

#ifdef ARCANE_HAS_CUDA_CUPTI
  void start();
  void stop();
  void flush();
#else
  void start() {}
  void stop() {}
  void flush() {}
#endif

 private:

#ifdef ARCANE_HAS_CUDA_CUPTI
  FixedArray<CUpti_ActivityUnifiedMemoryCounterConfig, 4> config;
  CUpti_ActivityPCSamplingConfig configPC;
#endif
  bool m_is_active = false;
  bool m_do_print = true;
  int m_profiling_level = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator::Cuda

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
