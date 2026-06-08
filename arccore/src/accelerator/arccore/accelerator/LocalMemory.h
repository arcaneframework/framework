// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* LocalMemory.h                                               (C) 2000-2025 */
/*                                                                           */
/* Local memory for a RunCommand.                                            */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_ACCELERATOR_LOCALMEMORY_H
#define ARCCORE_ACCELERATOR_LOCALMEMORY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/Span.h"

#include "arccore/common/accelerator/RunCommand.h"
#include "arccore/accelerator/AcceleratorUtils.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator::Impl
{
#if defined(ARCCORE_COMPILING_CUDA_OR_HIP)
inline __device__ std::byte* _getAcceleratorSharedMemory()
{
  extern __shared__ Int64 shared_memory_ptr[];
  return reinterpret_cast<std::byte*>(shared_memory_ptr);
}
#endif

class LocalMemoryKernelRemainingArg;

} // namespace Arcane::Accelerator::Impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Local memory (__shared__) for a RunCommand.
 *
 * \warning API is currently under definition. Do not use outside of Arcane.
 */
template <typename T, Int32 Extent>
class LocalMemory
{
  friend Impl::LocalMemoryKernelRemainingArg;

 public:

  static_assert(std::is_trivially_copyable_v<T>, "type T is not trivially copiable");

 public:

  using SpanType = SmallSpan<T, Extent>;
  using RemainingArgHandlerType = Impl::LocalMemoryKernelRemainingArg;

 public:

  LocalMemory(RunCommand& command, Int32 size)
  : m_size(size)
  {
    _addShareMemory(command);
  }

  explicit LocalMemory(RunCommand& command) requires(Extent != DynExtent)
  {
    _addShareMemory(command);
  }

  constexpr ARCCORE_HOST_DEVICE SmallSpan<T, Extent> span()
  {
    return { m_ptr, m_size.size() };
  }

 private:

  T* m_ptr = nullptr;
  // TODO: the offset is not used, we could remove the offset by passing it
  //! Offset from the beginning of the __shared__ memory
  Int32 m_offset = 0;
  //! Number of elements in the array
  [[no_unique_address]] ::Arcane::Impl::ExtentStorage<Int32, Extent> m_size;

 protected:

  void _addShareMemory(RunCommand& command)
  {
    m_offset = command._addSharedMemory(static_cast<Int32>(sizeof(T) * m_size.size()));
  }
};

} // namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator::Impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Handler for LocalMemory called at the beginning and end of kernel
 * execution.
 */
class LocalMemoryKernelRemainingArg
{
 public:

  template <typename T, Int32 Extent> static void
  execWorkItemAtBeginForHost(LocalMemory<T, Extent>& local_memory)
  {
    local_memory.m_ptr = new T[local_memory.m_size.size()];
  }
  template <typename T, Int32 Extent> static void
  execWorkItemAtEndForHost(LocalMemory<T, Extent>& local_memory)
  {
    delete[] local_memory.m_ptr;
  }

  template <typename T, Int32 Extent> static bool
  isNeedBarrier(const LocalMemory<T, Extent>&)
  {
    return false;
  }

#if defined(ARCCORE_COMPILING_CUDA_OR_HIP)
  template <typename T, Int32 Extent> static ARCCORE_DEVICE void
  execWorkItemAtBeginForCudaHip(LocalMemory<T, Extent>& local_memory, Int32)
  {
    std::byte* begin = Impl::_getAcceleratorSharedMemory() + local_memory.m_offset;
    local_memory.m_ptr = reinterpret_cast<T*>(begin);
  }
  template <typename T, Int32 Extent> static ARCCORE_DEVICE void
  execWorkItemAtEndForCudaHip(LocalMemory<T, Extent>&, Int32)
  {
  }
#endif

#if defined(ARCCORE_COMPILING_SYCL)
  template <typename T, Int32 Extent> static void
  execWorkItemAtBeginForSycl(LocalMemory<T, Extent>& local_memory,
                             sycl::nd_item<1>,
                             SmallSpan<std::byte> shm_view)
  {
    std::byte* begin = shm_view.ptrAt(local_memory.m_offset);
    local_memory.m_ptr = reinterpret_cast<T*>(begin);
  }
  template <typename T, Int32 Extent> static void
  execWorkItemAtEndForSycl(LocalMemory<T, Extent>&,
                           sycl::nd_item<1>,
                           SmallSpan<std::byte>)
  {
  }
#endif
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator::Impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
