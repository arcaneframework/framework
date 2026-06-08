// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* FixedArray.h                                                (C) 2000-2025 */
/*                                                                           */
/* Fixed-size 1D array.                                                      */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_FIXEDARRAY_H
#define ARCCORE_BASE_FIXEDARRAY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ArrayView.h"

#include <array>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup Collection
 *
 * \brief Fixed-size 1D array.
 *
 * This class is similar to std::array with the following differences:
 *
 * - the number of elements is an 'Int32'.
 * - elements are initialized with the default constructor
 * - in 'Check' mode, checks for array overflows
 *
 * This class also provides conversions to ArrayView, ConstArrayView
 * and SmallSpan.
 */
template <typename T, Int32 NbElement>
class FixedArray final
{
  static_assert(NbElement >= 0, "NbElement has to positive");

 public:

  using value_type = T;
  using size_type = Int32;
  using difference_type = std::ptrdiff_t;
  using reference = value_type&;
  using const_reference = const value_type&;
  using pointer = value_type*;
  using const_pointer = const value_type*;

  using iterator = typename std::array<T, NbElement>::iterator;
  using const_iterator = typename std::array<T, NbElement>::const_iterator;

 public:

  //! Creates an array by initializing elements with the default constructor of \a T
  constexpr FixedArray()
  : m_value({})
  {}
  //! Creates an array by initializing elements with \a x
  constexpr FixedArray(std::array<T, NbElement> x)
  : m_value(std::move(x))
  {}
  //! Copies \a x into the instance
  constexpr FixedArray<T,NbElement>& operator=(std::array<T, NbElement> x)
  {
    m_value = std::move(x);
    return *this;
  }

 public:

  //! Value of the i-th element
  constexpr ARCCORE_HOST_DEVICE T& operator[](Int32 index)
  {
    ARCCORE_CHECK_AT(index, NbElement);
    return m_value[index];
  }
  //! Value of the i-th element
  constexpr ARCCORE_HOST_DEVICE const T& operator[](Int32 index) const
  {
    ARCCORE_CHECK_AT(index, NbElement);
    return m_value[index];
  }
  //! Modifiable view of the array
  constexpr ARCCORE_HOST_DEVICE SmallSpan<T, NbElement> span() { return { m_value.data(), NbElement }; }
  //! Non-modifiable view of the array
  constexpr ARCCORE_HOST_DEVICE SmallSpan<const T, NbElement> span() const { return { m_value.data(), NbElement }; }
  //! Modifiable view of the array
  constexpr ARCCORE_HOST_DEVICE ArrayView<T> view() { return { NbElement, m_value.data() }; }
  //! Non-modifiable view of the array
  constexpr ARCCORE_HOST_DEVICE ConstArrayView<T> view() const { return { NbElement, m_value.data() }; }
  constexpr ARCCORE_HOST_DEVICE const T* data() const { return m_value.data(); }
  constexpr ARCCORE_HOST_DEVICE T* data() { return m_value.data(); }

  //! Number of elements in the array
  static constexpr Int32 size() { return NbElement; }

 public:

  //! Iterator to the beginning of the array
  constexpr iterator begin() { return m_value.begin(); }
  //! Iterator to the end of the array
  constexpr iterator end() { return m_value.end(); }
  //! Constant iterator to the beginning of the array
  constexpr const_iterator begin() const { return m_value.begin(); }
  //! Constant iterator to the end of the array
  constexpr const_iterator end() const { return m_value.end(); }

 private:

  std::array<T, NbElement> m_value;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
