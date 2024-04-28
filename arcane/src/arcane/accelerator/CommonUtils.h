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
#if defined(ARCANE_COMPILING_SYCL)
#include "arcane/accelerator/sycl/SyclAccelerator.h"
#include <sycl/sycl.hpp>
#if defined(__INTEL_LLVM_COMPILER)
#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#endif
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator::impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

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

#if defined(ARCANE_COMPILING_SYCL)
class ARCANE_ACCELERATOR_EXPORT SyclUtils
{
 public:

  static sycl::queue toNativeStream(RunQueue* queue);
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

//! Opérateur de Scan/Reduce pour les sommes
template <typename DataType>
class SumOperator
{
 public:

  constexpr ARCCORE_HOST_DEVICE DataType operator()(const DataType& a, const DataType& b) const
  {
    return a + b;
  }
  static DataType defaultValue() { return {}; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Opérateur de Scan/Reduce pour le minimum
template <typename DataType>
class MinOperator
{
 public:

  constexpr ARCCORE_HOST_DEVICE DataType operator()(const DataType& a, const DataType& b) const
  {
    return (a < b) ? a : b;
  }
  static DataType defaultValue() { return std::numeric_limits<DataType>::max(); }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Opérateur de Scan/Reduce pour le maximum
template <typename DataType>
class MaxOperator
{
 public:

  constexpr ARCCORE_HOST_DEVICE DataType operator()(const DataType& a, const DataType& b) const
  {
    return (a < b) ? b : a;
  }
  static DataType defaultValue() { return std::numeric_limits<DataType>::lowest(); }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Itérateur sur une lambda pour récupérer une valeur via un index.
 */
template <typename DataType, typename GetterLambda>
class GetterLambdaIterator
{
 public:

  using value_type = DataType;
  using iterator_category = std::random_access_iterator_tag;
  using reference = DataType&;
  using difference_type = ptrdiff_t;
  using pointer = void;
  using ThatClass = GetterLambdaIterator<DataType, GetterLambda>;

 public:

  ARCCORE_HOST_DEVICE GetterLambdaIterator(const GetterLambda& s)
  : m_lambda(s)
  {}
  ARCCORE_HOST_DEVICE explicit GetterLambdaIterator(const GetterLambda& s, Int32 v)
  : m_index(v)
  , m_lambda(s)
  {}

 public:

  ARCCORE_HOST_DEVICE ThatClass& operator++()
  {
    ++m_index;
    return (*this);
  }
  ARCCORE_HOST_DEVICE ThatClass& operator+=(Int32 x)
  {
    m_index += x;
    return (*this);
  }
  ARCCORE_HOST_DEVICE friend ThatClass operator+(const ThatClass& iter, Int32 x)
  {
    return ThatClass(iter.m_lambda, iter.m_index + x);
  }
  ARCCORE_HOST_DEVICE friend ThatClass operator+(Int32 x, const ThatClass& iter)
  {
    return ThatClass(iter.m_lambda, iter.m_index + x);
  }
  ARCCORE_HOST_DEVICE friend bool operator<(const ThatClass& iter1, const ThatClass& iter2)
  {
    return iter1.m_index < iter2.m_index;
  }

  ARCCORE_HOST_DEVICE ThatClass operator-(Int32 x) const
  {
    return ThatClass(m_lambda, m_index - x);
  }
  ARCCORE_HOST_DEVICE Int32 operator-(const ThatClass& x) const
  {
    return m_index - x.m_index;
  }
  ARCCORE_HOST_DEVICE value_type operator*() const
  {
    return m_lambda(m_index);
  }
  ARCCORE_HOST_DEVICE value_type operator[](Int32 x) const { return m_lambda(m_index + x); }
  ARCCORE_HOST_DEVICE friend bool operator!=(const ThatClass& a, const ThatClass& b)
  {
    return a.m_index != b.m_index;
  }
  ARCCORE_HOST_DEVICE friend bool operator==(const ThatClass& a, const ThatClass& b)
  {
    return a.m_index == b.m_index;
  }

 private:

  Int32 m_index = 0;
  GetterLambda m_lambda;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
