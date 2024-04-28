// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Atomic.h                                                    (C) 2000-2024 */
/*                                                                           */
/* Opérations atomiques.                                                     */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ACCELERATOR_ATOMIC_H
#define ARCANE_ACCELERATOR_ATOMIC_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcaneCxx20.h"

#if defined(ARCCORE_DEVICE_TARGET_CUDA) || defined(ARCCORE_DEVICE_TARGET_HIP)
#include "arcane/accelerator/CommonCudaHipAtomicImpl.h"
#endif

#include <atomic>
#include <concepts>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator
{
//! Liste des types supportant les opérations atomiques.
template <typename T>
concept AcceleratorAtomicConcept = std::same_as<T, Real> || std::same_as<T, Int32> || std::same_as<T, Int64>;
}

namespace Arcane::Accelerator::impl
{

template <enum eAtomicOperation Operation>
class HostAtomic;
template <enum eAtomicOperation Operation>
class SyclAtomic;

template <>
class HostAtomic<eAtomicOperation::Add>
{
 public:

  template <AcceleratorAtomicConcept DataType> static void
  apply(DataType* ptr, DataType value)
  {
    std::atomic_ref<DataType> v(*ptr);
    v.fetch_add(value);
  }
};

template <>
class HostAtomic<eAtomicOperation::Max>
{
 public:

  template <AcceleratorAtomicConcept DataType> static void
  apply(DataType* ptr, DataType value)
  {
    std::atomic_ref<DataType> v(*ptr);
    DataType prev_value = v;
    while (prev_value < value && !v.compare_exchange_weak(prev_value, value)) {
    }
  }
};

template <>
class HostAtomic<eAtomicOperation::Min>
{
 public:

  template <AcceleratorAtomicConcept DataType> static void
  apply(DataType* ptr, DataType value)
  {
    std::atomic_ref<DataType> v(*ptr);
    DataType prev_value = v;
    while (prev_value > value && !v.compare_exchange_weak(prev_value, value)) {
    }
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#if defined(ARCANE_COMPILING_SYCL)

template <>
class SyclAtomic<eAtomicOperation::Add>
{
 public:

  template <AcceleratorAtomicConcept DataType> static void
  apply(DataType* ptr, DataType value)
  {
    sycl::atomic_ref<DataType, sycl::memory_order::relaxed, sycl::memory_scope::device> v(*ptr);
    v.fetch_add(value);
  }
};

template <>
class SyclAtomic<eAtomicOperation::Max>
{
 public:

  template <AcceleratorAtomicConcept DataType> static void
  apply(DataType* ptr, DataType value)
  {
    sycl::atomic_ref<DataType, sycl::memory_order::relaxed, sycl::memory_scope::device> v(*ptr);
    v.fetch_max(value);
  }
};

template <>
class SyclAtomic<eAtomicOperation::Min>
{
 public:

  template <AcceleratorAtomicConcept DataType> static void
  apply(DataType* ptr, DataType value)
  {
    sycl::atomic_ref<DataType, sycl::memory_order::relaxed, sycl::memory_scope::device> v(*ptr);
    v.fetch_min(value);
  }
};

#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class AtomicImpl
{
 public:

  template <AcceleratorAtomicConcept DataType, enum eAtomicOperation Operation>
  ARCCORE_HOST_DEVICE static inline void
  doAtomic(DataType* ptr, DataType value)
  {
#if defined(ARCCORE_DEVICE_TARGET_CUDA) || defined(ARCCORE_DEVICE_TARGET_HIP)
    impl::CommonCudaHipAtomic<DataType, Operation>::apply(ptr, value);
#elif defined(ARCCORE_DEVICE_TARGET_SYCL)
    SyclAtomic<Operation>::apply(ptr, value);
#else
    HostAtomic<Operation>::apply(ptr, value);
#endif
  }

  template <AcceleratorAtomicConcept DataType, enum eAtomicOperation Operation>
  ARCCORE_HOST_DEVICE static inline void
  doAtomic(const DataViewGetterSetter<DataType>& view, DataType value)
  {
    doAtomic<DataType, Operation>(view._address(), value);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator::impl

namespace Arcane::Accelerator
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Applique l'opération atomique \a Operation à la valeur à l'adresse \a ptr avec la valeur \a value
template <enum eAtomicOperation Operation, AcceleratorAtomicConcept DataType, typename ValueType>
ARCCORE_HOST_DEVICE inline void
doAtomic(DataType* ptr, ValueType value)
requires(std::convertible_to<ValueType, DataType>)
{
  DataType v = value;
  impl::AtomicImpl::doAtomic<DataType, Operation>(ptr, v);
}

//! Applique l'opération atomique \a Operation à la vue \a view avec la valeur \a value
template <enum eAtomicOperation Operation, AcceleratorAtomicConcept DataType, typename ValueType>
ARCCORE_HOST_DEVICE inline void
doAtomic(const DataViewGetterSetter<DataType>& view, ValueType value)
requires(std::convertible_to<ValueType, DataType>)
{
  DataType v = value;
  impl::AtomicImpl::doAtomic<DataType, Operation>(view, v);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
