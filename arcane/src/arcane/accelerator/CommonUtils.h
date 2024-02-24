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

  ~GenericDeviceStorage() ARCANE_NOEXCEPT_FALSE
  {
    deallocate();
  }

 public:

  void* address() { return m_ptr; }
  size_t size() const { return m_size; }
  void allocate(size_t new_size)
  {
    if (new_size < m_size)
      return;
    deallocate();
#if defined(ARCANE_COMPILING_CUDA)
    ARCANE_CHECK_CUDA(::cudaMalloc(&m_ptr, new_size));
#endif
#if defined(ARCANE_COMPILING_HIP)
    ARCANE_CHECK_HIP(::hipMalloc(&m_ptr, new_size));
#endif
    m_size = new_size;
  }

  void deallocate()
  {
    if (!m_ptr)
      return;
#if defined(ARCANE_COMPILING_CUDA)
    ARCANE_CHECK_CUDA(::cudaFree(m_ptr));
#endif
#if defined(ARCANE_COMPILING_HIP)
    ARCANE_CHECK_HIP(::hipFree(m_ptr));
#endif
    m_ptr = nullptr;
    m_size = 0;
  }

 private:

  void* m_ptr = nullptr;
  size_t m_size = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Gère l'allocation interne sur le device pour un type donné.
 */
template <typename DataType>
class DeviceStorage
{
 public:

  ~DeviceStorage() ARCANE_NOEXCEPT_FALSE
  {
  }

  DataType* address() { return reinterpret_cast<DataType*>(m_storage.address()); }
  size_t size() const { return m_storage.size(); }
  void allocate() { m_storage.allocate(sizeof(DataType)); }
  void deallocate() { m_storage.deallocate(); }

 private:

  GenericDeviceStorage m_storage;
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
