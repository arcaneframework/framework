// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* NativeStream.h                                              (C) 2000-2025 */
/*                                                                           */
/* Opaque type to encapsulate a native 'stream'.                             */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_COMMON_ACCELERATOR_PLATFORMSTREAM_H
#define ARCCORE_COMMON_ACCELERATOR_PLATFORMSTREAM_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/common/accelerator/CommonAcceleratorGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
namespace Arcane::Accelerator
{
namespace Cuda
{
  class CudaRunQueueStream;
}
namespace Hip
{
  class HipRunQueueStream;
}
namespace Sycl
{
  class SyclRunQueueStream;
}
} // namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator::Impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Opaque type to encapsulate a native 'stream'.
 *
 * This class allows for the *temporary* retention of a native stream.
 * The exact type depends on the runtime: cudaStream_t, hipStream_to, or sycl::queue.
 *
 * Instances of this class must not be retained.
 */
class ARCCORE_COMMON_EXPORT NativeStream
{
  friend Arcane::Accelerator::RunQueue;
  friend Arcane::Accelerator::RunCommand;
  friend Arcane::Accelerator::Cuda::CudaRunQueueStream;
  friend Arcane::Accelerator::Hip::HipRunQueueStream;
  friend Arcane::Accelerator::Sycl::SyclRunQueueStream;
  friend Arcane::Accelerator::Impl::CudaUtils;
  friend Arcane::Accelerator::Impl::HipUtils;
  friend Arcane::Accelerator::Impl::SyclUtils;

 public:

  NativeStream() = default;

 private:

  explicit NativeStream(void* ptr)
  : m_native_pointer(ptr)
  {}

 private:

  void* m_native_pointer = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator::Impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
