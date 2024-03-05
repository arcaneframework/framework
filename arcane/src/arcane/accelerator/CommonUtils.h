// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CommonUtils.h                                               (C) 2000-2024 */
/*                                                                           */
/* Fonctions/Classes utilitaires communes à tout les runtimes.               */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ACCELERATOR_COMMONUTILS_H
#define ARCANE_ACCELERATOR_COMMONUTILS_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArrayView.h"

#include "arcane/accelerator/AcceleratorGlobal.h"
#include "arcane/accelerator/core/RunQueue.h"

#if defined(ARCANE_COMPILING_HIP)
#include "arcane/accelerator/hip/HipAccelerator.h"
#include <hip/hip_runtime.h>
#include <rocprim/rocprim.hpp>
#endif
#if defined(ARCANE_COMPILING_CUDA)
#include "arcane/accelerator/cuda/CudaAccelerator.h"
#include <cub/cub.cuh>
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator::impl
{

#if defined(ARCANE_COMPILING_CUDA)
class ARCANE_ACCELERATOR_EXPORT CudaUtils
{
 public:

  static cudaStream_t toNativeStream(RunQueue* queue);
};
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#if defined(ARCANE_COMPILING_HIP)
class ARCANE_ACCELERATOR_EXPORT HipUtils
{
 public:

  static hipStream_t toNativeStream(RunQueue* queue);
};
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Gère l'allocation interne sur le device.
 */
class GenericDeviceStorage
{
 public:

  ~GenericDeviceStorage()
  {
    deallocate();
  }

 public:

  void* address() { return m_ptr; }
  size_t size() const { return m_size; }
  void* allocate(size_t new_size)
  {
    if (new_size < m_size)
      return m_ptr;
    deallocate();
#if defined(ARCANE_COMPILING_CUDA)
    ARCANE_CHECK_CUDA(::cudaMalloc(&m_ptr, new_size));
#endif
#if defined(ARCANE_COMPILING_HIP)
    ARCANE_CHECK_HIP(::hipMallocManaged(&m_ptr, new_size, hipMemAttachGlobal));
#endif
    ARCANE_CHECK_PTR(m_ptr);
    m_size = new_size;
    return m_ptr;
  }

  void deallocate()
  {
    if (!m_ptr)
      return;
#if defined(ARCANE_COMPILING_CUDA)
    ARCANE_CHECK_CUDA_NOTHROW(::cudaFree(m_ptr));
#endif
#if defined(ARCANE_COMPILING_HIP)
    ARCANE_CHECK_HIP_NOTHROW(::hipFree(m_ptr));
#endif
    m_ptr = nullptr;
    m_size = 0;
  }

  Span<const std::byte> bytes() const
  {
    return { reinterpret_cast<const std::byte*>(m_ptr), static_cast<Int64>(m_size) };
  }

 private:

  void* m_ptr = nullptr;
  size_t m_size = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Gère l'allocation interne sur le device.
 */
class ARCANE_ACCELERATOR_EXPORT DeviceStorageBase
{
 protected:

  GenericDeviceStorage m_storage;

 protected:

  //! Copie l'instance dans \a dest_ptr
  void _copyToAsync(Span<std::byte> destination, Span<const std::byte> source, RunQueue* queue);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Gère l'allocation interne sur le device pour un type donné.
 */
template <typename DataType>
class DeviceStorage
: public DeviceStorageBase
{
 public:

  ~DeviceStorage()
  {
  }

  DataType* address() { return reinterpret_cast<DataType*>(m_storage.address()); }
  size_t size() const { return m_storage.size(); }
  DataType* allocate()
  {
    m_storage.allocate(sizeof(DataType));
    return address();
  }
  void deallocate() { m_storage.deallocate(); }

  //! Copie l'instance dans \a dest_ptr
  void copyToAsync(SmallSpan<DataType> dest_ptr, RunQueue* queue)
  {
    _copyToAsync(asWritableBytes(dest_ptr), m_storage.bytes(), queue);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Itérateur sur un index.
 *
 * Permet d'itérer entre deux entiers.
 */
class IndexIterator
{
 public:

  using value_type = Int32;
  using iterator_category = std::random_access_iterator_tag;
  using reference = value_type&;
  using difference_type = ptrdiff_t;

 public:

  IndexIterator() = default;
  ARCCORE_HOST_DEVICE explicit IndexIterator(Int32 v)
  : m_value(v)
  {}

 public:

  ARCCORE_HOST_DEVICE IndexIterator& operator++()
  {
    ++m_value;
    return (*this);
  }
  ARCCORE_HOST_DEVICE IndexIterator operator+(Int32 x) const
  {
    return IndexIterator(m_value + x);
  }
  ARCCORE_HOST_DEVICE IndexIterator operator-(Int32 x) const
  {
    return IndexIterator(m_value - x);
  }
  ARCCORE_HOST_DEVICE Int32 operator*() const { return m_value; }
  ARCCORE_HOST_DEVICE Int32 operator[](Int32 x) const { return m_value + x; }

 private:

  Int32 m_value = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
