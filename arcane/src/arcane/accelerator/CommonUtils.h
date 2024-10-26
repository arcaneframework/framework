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

#include "arcane/utils/Array.h"

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

  static cudaStream_t toNativeStream(const RunQueue* queue);
  static cudaStream_t toNativeStream(const RunQueue& queue);
};
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#if defined(ARCANE_COMPILING_HIP)
class ARCANE_ACCELERATOR_EXPORT HipUtils
{
 public:

  static hipStream_t toNativeStream(const RunQueue* queue);
  static hipStream_t toNativeStream(const RunQueue& queue);
};
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#if defined(ARCANE_COMPILING_SYCL)
class ARCANE_ACCELERATOR_EXPORT SyclUtils
{
 public:

  static sycl::queue toNativeStream(const RunQueue* queue);
  static sycl::queue toNativeStream(const RunQueue& queue);
};
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Gère l'allocation interne sur le device.
 */
class ARCANE_ACCELERATOR_EXPORT GenericDeviceStorage
{
 public:

  GenericDeviceStorage();
  ~GenericDeviceStorage()
  {
    deallocate();
  }

 public:

  void* address() { return m_storage.data(); }
  size_t size() const { return m_storage.largeSize(); }
  void* allocate(size_t new_size)
  {
    m_storage.resize(new_size);
    return m_storage.data();
  }

  void deallocate()
  {
    m_storage.clear();
  }

  Span<const std::byte> bytes() const
  {
    return m_storage.span();
  }

 private:

  UniqueArray<std::byte> m_storage;
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
  void _copyToAsync(Span<std::byte> destination, Span<const std::byte> source, const RunQueue& queue);
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

  DataType* address() { return reinterpret_cast<DataType*>(m_storage.address()); }
  size_t size() const { return m_storage.size(); }
  DataType* allocate()
  {
    m_storage.allocate(sizeof(DataType));
    return address();
  }
  void deallocate() { m_storage.deallocate(); }

  //! Copie l'instance dans \a dest_ptr
  void copyToAsync(SmallSpan<DataType> dest_ptr, const RunQueue& queue)
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
  using pointer = void;

  using ThatClass = IndexIterator;

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
  ARCCORE_HOST_DEVICE friend ThatClass operator+(Int32 x, const ThatClass& iter)
  {
    return ThatClass(iter.m_value + x);
  }
  ARCCORE_HOST_DEVICE IndexIterator operator-(Int32 x) const
  {
    return IndexIterator(m_value - x);
  }
  ARCCORE_HOST_DEVICE Int32 operator-(const ThatClass& x) const
  {
    return m_value - x.m_value;
  }
  ARCCORE_HOST_DEVICE Int32 operator*() const { return m_value; }
  ARCCORE_HOST_DEVICE Int32 operator[](Int32 x) const { return m_value + x; }
  ARCCORE_HOST_DEVICE friend bool operator==(const ThatClass& a, const ThatClass& b)
  {
    return a.m_value == b.m_value;
  }
  ARCCORE_HOST_DEVICE friend bool operator<(const ThatClass& iter1, const ThatClass& iter2)
  {
    return iter1.m_value < iter2.m_value;
  }

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
#if defined(ARCANE_COMPILING_SYCL)
  static sycl::plus<DataType> syclFunctor() { return {}; }
#endif
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
#if defined(ARCANE_COMPILING_SYCL)
  static sycl::minimum<DataType> syclFunctor() { return {}; }
#endif
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
#if defined(ARCANE_COMPILING_SYCL)
  static sycl::maximum<DataType> syclFunctor() { return {}; }
#endif
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
/*!
 * \brief Itérateur sur une lambda pour positionner une valeur via un index.
 *
 * Le positionnement se fait via Setter::operator=().
 */
template <typename SetterLambda>
class SetterLambdaIterator
{
 public:

  //! Permet de positionner un élément de l'itérateur de sortie
  class Setter
  {
   public:

    ARCCORE_HOST_DEVICE explicit Setter(const SetterLambda& s, Int32 output_index)
    : m_output_index(output_index)
    , m_lambda(s)
    {}
    ARCCORE_HOST_DEVICE void operator=(Int32 input_index)
    {
      m_lambda(input_index, m_output_index);
    }
    Int32 m_output_index = 0;
    SetterLambda m_lambda;
  };

  using value_type = Int32;
  using iterator_category = std::random_access_iterator_tag;
  using reference = Setter;
  using difference_type = ptrdiff_t;
  using pointer = void;

  using ThatClass = SetterLambdaIterator<SetterLambda>;

 public:

  ARCCORE_HOST_DEVICE SetterLambdaIterator(const SetterLambda& s)
  : m_lambda(s)
  {}
  ARCCORE_HOST_DEVICE explicit SetterLambdaIterator(const SetterLambda& s, Int32 v)
  : m_index(v)
  , m_lambda(s)
  {}

 public:

  ARCCORE_HOST_DEVICE SetterLambdaIterator<SetterLambda>& operator++()
  {
    ++m_index;
    return (*this);
  }
  ARCCORE_HOST_DEVICE SetterLambdaIterator<SetterLambda>& operator--()
  {
    --m_index;
    return (*this);
  }
  ARCCORE_HOST_DEVICE reference operator*() const
  {
    return Setter(m_lambda, m_index);
  }
  ARCCORE_HOST_DEVICE reference operator[](Int32 x) const { return Setter(m_lambda, m_index + x); }
  ARCCORE_HOST_DEVICE friend ThatClass operator+(Int32 x, const ThatClass& iter)
  {
    return ThatClass(iter.m_lambda, iter.m_index + x);
  }
  ARCCORE_HOST_DEVICE friend ThatClass operator+(const ThatClass& iter, Int32 x)
  {
    return ThatClass(iter.m_lambda, iter.m_index + x);
  }
  ARCCORE_HOST_DEVICE Int32 operator-(const ThatClass& x) const
  {
    return m_index - x.m_index;
  }
  ARCCORE_HOST_DEVICE friend bool operator<(const ThatClass& iter1, const ThatClass& iter2)
  {
    return iter1.m_index < iter2.m_index;
  }

 private:

  Int32 m_index = 0;
  SetterLambda m_lambda;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator
{
template <typename DataType> using ScannerSumOperator = impl::SumOperator<DataType>;
template <typename DataType> using ScannerMaxOperator = impl::MaxOperator<DataType>;
template <typename DataType> using ScannerMinOperator = impl::MinOperator<DataType>;
} // namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
