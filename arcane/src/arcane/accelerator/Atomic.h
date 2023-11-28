// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Atomic.h                                                    (C) 2000-2023 */
/*                                                                           */
/* Opérations atomiques.                                                     */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ACCELERATOR_ATOMIC_H
#define ARCANE_ACCELERATOR_ATOMIC_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcaneCxx20.h"

#ifdef ARCCORE_DEVICE_CODE
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

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class AtomicImpl
{
 public:

  template <AcceleratorAtomicConcept DataType>
  ARCCORE_HOST_DEVICE static inline void
  doAtomicAdd(DataType* ptr, DataType value_to_add)
  {
#ifdef ARCCORE_DEVICE_CODE
    impl::CommonCudaHipAtomic<DataType, eAtomicOperation::Add>::apply(ptr, value_to_add);
#else
    std::atomic_ref<DataType> v(*ptr);
    v.fetch_add(value_to_add);
#endif
  }

  template <AcceleratorAtomicConcept DataType>
  ARCCORE_HOST_DEVICE static inline void
  doAtomicAdd(const DataViewGetterSetter<DataType>& view, DataType value_to_add)
  {
    doAtomicAdd(view._address(), value_to_add);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator::impl

namespace Arcane::Accelerator
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Ajoute de manière atomique \a value_to_add à l'objet à l'adresse \a ptr
template <AcceleratorAtomicConcept DataType, typename ValueType>
ARCCORE_HOST_DEVICE inline void
atomicAdd(DataType* ptr, ValueType value_to_add)
requires(std::convertible_to<ValueType, DataType>)
{
  DataType v = value_to_add;
  impl::AtomicImpl::doAtomicAdd(ptr, v);
}

//! Ajoute de manière atomique \a value_to_add à la vue \a view
template <AcceleratorAtomicConcept DataType, typename ValueType>
ARCCORE_HOST_DEVICE inline void
atomicAdd(const DataViewGetterSetter<DataType>& view, ValueType value_to_add)
requires(std::convertible_to<ValueType, DataType>)
{
  DataType v = value_to_add;
  impl::AtomicImpl::doAtomicAdd(view, v);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
