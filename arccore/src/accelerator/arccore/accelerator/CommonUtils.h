// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CommonUtils.h                                               (C) 2000-2025 */
/*                                                                           */
/* Fonctions/Classes utilitaires communes à tout les runtimes.               */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_ACCELERATOR_COMMONUTILS_H
#define ARCCORE_ACCELERATOR_COMMONUTILS_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/accelerator/AcceleratorGlobal.h"

#include "arccore/common/accelerator/RunQueue.h"
#include "arccore/common/Array.h"

#if defined(ARCCORE_COMPILING_HIP)
#include "arccore/accelerator_native/HipAccelerator.h"
#include <rocprim/rocprim.hpp>
#endif
#if defined(ARCCORE_COMPILING_CUDA)
#include "arccore/accelerator_native/CudaAccelerator.h"
#include <cub/cub.cuh>
#endif
#if defined(ARCCORE_COMPILING_SYCL)
#include "arccore/accelerator_native/SyclAccelerator.h"
#if defined(ARCCORE_HAS_ONEDPL)
#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#endif
#if defined(__ADAPTIVECPP__)
#include <AdaptiveCpp/algorithms/algorithm.hpp>
#endif
#endif

#include "arccore/accelerator/AcceleratorUtils.h"

// A définir si on souhaite utiliser LambdaStorage
// #ifdef ARCCORE_USE_LAMBDA_STORAGE

#include <cstring>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator::impl
{
/*!
 * \brief Classe pour gérer la conservation d'une lambda dans un itérateur.
 *
 * Actuellement (C++20), on ne peut pas conserver une lambda dans un
 * itérateur car il manque deux choses: un constructeur par défaut et
 * un opérateur de recopie. Cette classe permet de supporter cela à condition
 * que les deux poins suivants soient respectés:
 * - instances capturées par la lambda sont trivialement copiables et donc
 *   la lambda l'est.
 * - les instances utilisant le constructeur par défaut ne sont pas utilisées
 *   (ce qui est le cas des itérateurs car ils ne sont pas valides s'ils sont
 *   construits avec le constructeur par défaut.
 *
 * A noter que l'alignement de cette classe doit être au moins celui de la
 * lambda associée.
 *
 * Cette classe n'est indispensable que pour SYCL avec oneAPI car il est
 * nécessite que les itérateurs aient le concept std::random_access_iterator.
 * Cependant elle devrait aussi fonctionner avec CUDA et ROCM. A tester
 * l'effet sur les performances.
 */
template <typename LambdaType>
class alignas(LambdaType) LambdaStorage
{
  static constexpr size_t SizeofLambda = sizeof(LambdaType);

 public:

  LambdaStorage() = default;
  ARCCORE_HOST_DEVICE LambdaStorage(const LambdaType& v)
  {
    std::memcpy(bytes, &v, SizeofLambda);
  }
  //! Convertie la classe en la lambda.
  ARCCORE_HOST_DEVICE operator const LambdaType&() const { return *reinterpret_cast<const LambdaType*>(&bytes); }

 private:

  char bytes[SizeofLambda];
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Gère l'allocation interne sur le device.
 */
class ARCCORE_ACCELERATOR_EXPORT GenericDeviceStorage
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
class ARCCORE_ACCELERATOR_EXPORT DeviceStorageBase
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
template <typename DataType, Int32 N = 1>
class DeviceStorage
: public DeviceStorageBase
{
 public:

  DataType* address() { return reinterpret_cast<DataType*>(m_storage.address()); }
  size_t size() const { return m_storage.size(); }
  DataType* allocate()
  {
    m_storage.allocate(sizeof(DataType) * N);
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
#if defined(ARCCORE_COMPILING_SYCL)
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
#if defined(ARCCORE_COMPILING_SYCL)
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
#if defined(ARCCORE_COMPILING_SYCL)
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

 private:

#ifdef ARCCORE_USE_LAMBDA_STORAGE
  using StorageType = LambdaStorage<SetterLambda>;
#else
  using StorageType = SetterLambda;
#endif

 public:

  SetterLambdaIterator() = default;
  ARCCORE_HOST_DEVICE SetterLambdaIterator(const SetterLambda& s)
  : m_lambda(s)
  {}
  ARCCORE_HOST_DEVICE SetterLambdaIterator(const SetterLambda& s, Int32 v)
  : m_index(v)
  , m_lambda(s)
  {}

 private:

#ifdef ARCCORE_USE_LAMBDA_STORAGE
  ARCCORE_HOST_DEVICE SetterLambdaIterator(const LambdaStorage<SetterLambda>& s, Int32 v)
  : m_index(v)
  , m_lambda(s)
  {}
#endif

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
  ARCCORE_HOST_DEVICE friend ThatClass operator-(const ThatClass& iter, Int32 x)
  {
    return ThatClass(iter.m_lambda, iter.m_index - x);
  }
  ARCCORE_HOST_DEVICE friend Int32 operator-(const ThatClass& iter1, const ThatClass& iter2)
  {
    return iter1.m_index - iter2.m_index;
  }
  ARCCORE_HOST_DEVICE friend bool operator<(const ThatClass& iter1, const ThatClass& iter2)
  {
    return iter1.m_index < iter2.m_index;
  }

 private:

  Int32 m_index = 0;
  StorageType m_lambda;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename DataType> using ScannerSumOperator = impl::SumOperator<DataType>;
template <typename DataType> using ScannerMaxOperator = impl::MaxOperator<DataType>;
template <typename DataType> using ScannerMinOperator = impl::MinOperator<DataType>;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifdef ARCCORE_USE_LAMBDA_STORAGE
#undef ARCCORE_USE_LAMBDA_STORAGE
#endif

#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
