// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Array.h                                                     (C) 2000-2018 */
/*                                                                           */
/* 1D Array.                                                                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_ARRAY_H
#define ARCANE_UTILS_ARRAY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/collections/Array.h"
#include "arcane/utils/UtilsTypes.h"
#include "arcane/utils/StdHeader.h"
#include "arcane/utils/Iostream.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename T>
class ArrayFullAccessorT
{
 public:

  ArrayFullAccessorT(Array<T>& v)
  : m_array(&v)
  {}
  ~ArrayFullAccessorT() {}

 public:

  T operator[](Integer i) const { return m_array->item(i); }
  T& operator[](Integer i) { return (*m_array)[i]; }
  Integer size() const { return m_array->size(); }
  void resize(Integer s) { m_array->resize(s); }
  void add(T v) { m_array->add(v); }

 private:

  Array<T>* m_array;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Applies padding at the end of the array \a ids.
 *
 * This method fills the elements of \a ids after the last value
 * so that \a ids has a valid number of elements that is a multiple of the size
 * of a Simd vector.
 *
 * \a ids must use the AlignedMemoryAllocator::Simd() allocator.
 * The padding is done using the value of the last element
 * valid element of \a ids.
 *
 * For example, if ids.size()==5 and the Simd vector size is 8,
 * then ids[5], ids[6], and ids[7] are filled with the value of ids[4].
 */
//@{
extern ARCANE_UTILS_EXPORT void
applySimdPadding(Array<Int32>& ids);

extern ARCANE_UTILS_EXPORT void
applySimdPadding(Array<Int16>& ids);

extern ARCANE_UTILS_EXPORT void
applySimdPadding(Array<Int64>& ids);

extern ARCANE_UTILS_EXPORT void
applySimdPadding(Array<Real>& ids);
//@}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
