// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* NativeStream.h                                              (C) 2000-2025 */
/*                                                                           */
/* Type opaque pour encapsuler une 'stream' native.                          */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ACCELERATOR_CORE_PLATFORMSTREAM_H
#define ARCANE_ACCELERATOR_CORE_PLATFORMSTREAM_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/accelerator/core/AcceleratorCoreGlobal.h"

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
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator::Impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Type opaque pour encapsuler une 'stream' native.
 *
 * Cette classe permet de conserver *temporairement* une stream native.
 * Le type exact dépend du runtime: cudaStream_t, hipStream_to ou sycl::queue.
 *
 * Les instances de cette classe ne doivent pas être conservées.
 */
class ARCANE_ACCELERATOR_CORE_EXPORT NativeStream
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
