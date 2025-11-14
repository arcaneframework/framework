// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* LocalMemory.h                                               (C) 2000-2025 */
/*                                                                           */
/* Mémoire locale à une RunCommand.                                          */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ACCELERATOR_LOCALMEMORY_H
#define ARCANE_ACCELERATOR_LOCALMEMORY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/accelerator/AcceleratorGlobal.h"

#include "arcane/accelerator/core/RunCommand.h"

#include "arccore/base/Span.h"

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
template <typename T, Int32 Extent>
class LocalMemory
{
  friend ::Arcane::Impl::HostKernelRemainingArgsHelper;
  friend Impl::KernelRemainingArgsHelper;

 public:

  static_assert(std::is_trivially_copyable_v<T>, "type T is not trivially copiable");

 public:

  using SpanType = SmallSpan<T, Extent>;

 public:

  LocalMemory(RunCommand& command, Int32 size)
  : m_size(size)
  {
    _addShareMemory(command);
  }

  LocalMemory(RunCommand& command) requires(Extent != DynExtent)
  {
    _addShareMemory(command);
  }

  constexpr ARCCORE_HOST_DEVICE SmallSpan<T, Extent> span()
  {
    return { m_ptr, m_size.size() };
  }

 private:

#if defined(ARCANE_COMPILING_CUDA) || defined(ARCANE_COMPILING_HIP)
  ARCCORE_DEVICE void _internalExecWorkItemAtBegin(Int32)
  {
    std::byte* begin = Impl::_getAcceleratorSharedMemory() + m_offset;
    m_ptr = reinterpret_cast<T*>(begin);
  }
  ARCCORE_DEVICE void _internalExecWorkItemAtEnd(Int32){};
#endif

#if defined(ARCANE_COMPILING_SYCL)
  void _internalExecWorkItemAtBegin(sycl::nd_item<1>, SmallSpan<std::byte> shm_view)
  {
    std::byte* begin = shm_view.ptrAt(m_offset);
    m_ptr = reinterpret_cast<T*>(begin);
  }
  void _internalExecWorkItemAtEnd(sycl::nd_item<1>, SmallSpan<std::byte>) {}
#endif

  void _internalHostExecWorkItemAtBegin()
  {
    m_ptr = new T[m_size.size()];
  }
  void _internalHostExecWorkItemAtEnd()
  {
    delete[] m_ptr;
  }

 private:

  T* m_ptr = nullptr;
  // TODO: l'offset n'est utilisé on pourrait supprimer l'offset en le passant
  //! Offset depuis le début de la mémoire __shared__
  Int32 m_offset = 0;
  //! Nombre d'éléments du tableau
  [[no_unique_address]] ::Arcane::Impl::ExtentStorage<Int32, Extent> m_size;

 protected:

  void _addShareMemory(RunCommand& command)
  {
    m_offset = command._addSharedMemory(static_cast<Int32>(sizeof(T) * m_size.size()));
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
