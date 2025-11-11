// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* RunCommandLocalMemory.h                                     (C) 2000-2025 */
/*                                                                           */
/* Mémoire locale à une RunCommand.                                          */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ACCELERATOR_RUNCOMMANDLOCALMEMORY_H
#define ARCANE_ACCELERATOR_RUNCOMMANDLOCALMEMORY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/accelerator/AcceleratorGlobal.h"

#include "arcane/accelerator/core/RunCommand.h"

#include <iostream>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator::Impl
{
#if defined(ARCANE_COMPILING_CUDA) || defined(ARCANE_COMPILING_HIP)
inline __device__ std::byte* _getAcceleratorSharedMemory()
{
  extern __shared__ Int64 shared_memory_ptr[];
  return reinterpret_cast<std::byte*>(shared_memory_ptr);
}
#endif
} // namespace Arcane::Accelerator::Impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Mémoire locale (__shared__) à une RunCommand.
 *
 * \warning API en cours de définition. Ne pas utiliser en dehors d'Arcane.
 */
template <typename T>
class RunCommandLocalMemory
{
  friend ::Arcane::Impl::HostKernelRemainingArgsHelper;
  friend Impl::KernelRemainingArgsHelper;

 public:

  RunCommandLocalMemory(RunCommand& command, Int32 size)
  : m_size(size)
  {
    command._addSharedMemory(static_cast<Int32>(sizeof(T) * size));
  }

  constexpr ARCCORE_HOST_DEVICE SmallSpan<T> span()
  {
    return { m_ptr, m_size };
  }

 private:

#if defined(ARCANE_COMPILING_CUDA) || defined(ARCANE_COMPILING_HIP)
  ARCCORE_DEVICE void _internalExecWorkItemAtBegin(Int32)
  {
    m_ptr = reinterpret_cast<T*>(Impl::_getAcceleratorSharedMemory());
  }
  ARCCORE_DEVICE void _internalExecWorkItemAtEnd(Int32){};
#endif

#if defined(ARCANE_COMPILING_SYCL)
  // NOT YET SUPPORTED
  void _internalExecWorkItemAtBegin(sycl::nd_item<1>){}
  void _internalExecWorkItemAtEnd(sycl::nd_item<1>) {}
#endif

  void _internalHostExecWorkItemAtBegin()
  {
    m_ptr = new T[m_size];
  }
  void _internalHostExecWorkItemAtEnd()
  {
    delete m_ptr;
  }

 private:

  T* m_ptr = nullptr;
  Int32 m_size = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
