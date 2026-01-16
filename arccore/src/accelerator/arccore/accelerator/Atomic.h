// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Atomic.h                                                    (C) 2000-2026 */
/*                                                                           */
/* Opérations atomiques.                                                     */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_ACCELERATOR_ATOMIC_H
#define ARCCORE_ACCELERATOR_ATOMIC_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/accelerator/AcceleratorGlobal.h"

#if defined(ARCCORE_DEVICE_TARGET_CUDA) || defined(ARCCORE_DEVICE_TARGET_HIP)
#include "arccore/accelerator/CommonCudaHipAtomicImpl.h"
#endif
#include "arccore/accelerator/AcceleratorUtils.h"

#include <atomic>
#include <concepts>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator
{
//! Liste des types supportant les opérations atomiques.
template <typename T>
concept AcceleratorAtomicConcept = std::same_as<T, Real> || std::same_as<T, Int32> || std::same_as<T, Int64>;
} // namespace Arcane::Accelerator

namespace Arcane::Accelerator::Impl
{

template <enum eAtomicOperation Operation>
class HostAtomic;
template <enum eAtomicOperation Operation>
class SyclAtomic;

template <>
class HostAtomic<eAtomicOperation::Add>
{
 public:

  template <AcceleratorAtomicConcept DataType> static DataType
  apply(DataType* ptr, DataType value)
  {
    std::atomic_ref<DataType> v(*ptr);
    return v.fetch_add(value);
  }
};

template <>
class HostAtomic<eAtomicOperation::Max>
{
 public:

  template <AcceleratorAtomicConcept DataType> static DataType
  apply(DataType* ptr, DataType value)
  {
    std::atomic_ref<DataType> v(*ptr);
    DataType prev_value = v;
    while (prev_value < value && !v.compare_exchange_weak(prev_value, value)) {
    }
    return prev_value;
  }
};

template <>
class HostAtomic<eAtomicOperation::Min>
{
 public:

  template <AcceleratorAtomicConcept DataType> static DataType
  apply(DataType* ptr, DataType value)
  {
    std::atomic_ref<DataType> v(*ptr);
    DataType prev_value = v;
    while (prev_value > value && !v.compare_exchange_weak(prev_value, value)) {
    }
    return prev_value;
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#if defined(ARCANE_COMPILING_SYCL)

template <>
class SyclAtomic<eAtomicOperation::Add>
{
 public:

  template <AcceleratorAtomicConcept DataType> static DataType
  apply(DataType* ptr, DataType value)
  {
    sycl::atomic_ref<DataType, sycl::memory_order::relaxed, sycl::memory_scope::device> v(*ptr);
    return v.fetch_add(value);
  }
};

template <>
class SyclAtomic<eAtomicOperation::Max>
{
 public:

  template <AcceleratorAtomicConcept DataType> static DataType
  apply(DataType* ptr, DataType value)
  {
    sycl::atomic_ref<DataType, sycl::memory_order::relaxed, sycl::memory_scope::device> v(*ptr);
    return v.fetch_max(value);
  }
};

template <>
class SyclAtomic<eAtomicOperation::Min>
{
 public:

  template <AcceleratorAtomicConcept DataType> static DataType
  apply(DataType* ptr, DataType value)
  {
    sycl::atomic_ref<DataType, sycl::memory_order::relaxed, sycl::memory_scope::device> v(*ptr);
    return v.fetch_min(value);
  }
};

#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class AtomicImpl
{
 public:

  template <AcceleratorAtomicConcept DataType, enum eAtomicOperation Operation>
  ARCCORE_HOST_DEVICE static inline DataType
  doAtomic(DataType* ptr, DataType value)
  {
#if defined(ARCCORE_DEVICE_TARGET_CUDA) || defined(ARCCORE_DEVICE_TARGET_HIP)
    return Impl::CommonCudaHipAtomic<DataType, Operation>::apply(ptr, value);
#elif defined(ARCCORE_DEVICE_TARGET_SYCL)
    return SyclAtomic<Operation>::apply(ptr, value);
#else
    return HostAtomic<Operation>::apply(ptr, value);
#endif
  }

  template <AcceleratorAtomicConcept DataType, enum eAtomicOperation Operation>
  ARCCORE_HOST_DEVICE static inline DataType
  doAtomic(const DataViewGetterSetter<DataType>& view, DataType value)
  {
    return doAtomic<DataType, Operation>(view._address(), value);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator::Impl

namespace Arcane::Accelerator
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Applique l'opération atomique \a Operation à la valeur à l'adresse \a ptr avec la valeur \a value.
 *
 * \retval l'ancienne valeur avant ajout.
 */
template <enum eAtomicOperation Operation, AcceleratorAtomicConcept DataType, typename ValueType>
ARCCORE_HOST_DEVICE inline DataType
doAtomic(DataType* ptr, ValueType value)
requires(std::convertible_to<ValueType, DataType>)
{
  DataType v = value;
  return Impl::AtomicImpl::doAtomic<DataType, Operation>(ptr, v);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Applique l'opération atomique \a Operation à la vue \a view avec la valeur \a value.
 *
 * \retval l'ancienne valeur avant ajout.
 */
template <enum eAtomicOperation Operation, AcceleratorAtomicConcept DataType, typename ValueType>
ARCCORE_HOST_DEVICE inline DataType
doAtomic(const DataViewGetterSetter<DataType>& view, ValueType value)
requires(std::convertible_to<ValueType, DataType>)
{
  DataType v = value;
  return Impl::AtomicImpl::doAtomic<DataType, Operation>(view, v);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Ajoute atomiquement à la vue \a view avec la valeur \a value.
 *
 * \retval l'ancienne valeur avant ajout.
 */
template <AcceleratorAtomicConcept DataType, typename ValueType>
ARCCORE_HOST_DEVICE inline DataType
doAtomicAdd(const DataViewGetterSetter<DataType>& view, ValueType value)
requires(std::convertible_to<ValueType, DataType>)
{
  DataType v = value;
  return Impl::AtomicImpl::doAtomic<DataType, eAtomicOperation::Add>(view, v);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Ajoute atomiquement la valeur à l'adresse \a ptr avec la valeur \a value.
 *
 * \retval l'ancienne valeur avant ajout.
 */
template <AcceleratorAtomicConcept DataType, typename ValueType>
ARCCORE_HOST_DEVICE inline DataType
doAtomicAdd(DataType* ptr, ValueType value)
requires(std::convertible_to<ValueType, DataType>)
{
  DataType v = value;
  return Impl::AtomicImpl::doAtomic<DataType, eAtomicOperation::Add>(ptr, v);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
