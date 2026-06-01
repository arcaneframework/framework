// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MemoryUtils.h                                               (C) 2000-2025 */
/*                                                                           */
/* Memory management utility functions.                                      */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_MEMORYUTILS_H
#define ARCANE_UTILS_MEMORYUTILS_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/UtilsTypes.h"

#include "arccore/common/MemoryAllocationArgs.h"
#include "arccore/common/MemoryUtils.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::MemoryUtils
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace impl
{
  //! Calculates an appropriate capacity for a size \a size
  extern "C++" ARCANE_UTILS_EXPORT Int64
  computeCapacity(Int64 size);
} // namespace impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Resizes an array by adding a memory reserve.
 *
 * The array \a array is resized only if \a new_size is greater than the current size of the array or if \a force_resize is true.
 *
 * If the array is resized, an additional capacity is reserved to prevent reallocating every time.
 *
 * \retval 2 if reallocation occurred via reserve()
 * \retval 1 if resizing occurred without reallocation.
 * \retval 0 if no operation took place.
 */
template <typename DataType> inline Int32
checkResizeArrayWithCapacity(Array<DataType>& array, Int64 new_size, bool force_resize)
{
  Int32 ret_value = 0;
  Int64 s = array.largeSize();
  if (new_size > s || force_resize) {
    ret_value = 1;
    if (new_size > array.capacity()) {
      array.reserve(impl::computeCapacity(new_size));
      ret_value = 2;
    }
    array.resize(new_size);
  }
  return ret_value;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Resizes an array by adding a memory reserve.
 *
 * This call is equivalent to checkResizeArrayWithCapacity(array, new_size, false).
 */
template <typename DataType> inline Int32
checkResizeArrayWithCapacity(Array<DataType>& array, Int64 new_size)
{
  return checkResizeArrayWithCapacity(array, new_size, false);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::MemoryUtils

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
