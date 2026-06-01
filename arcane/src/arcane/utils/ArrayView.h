// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ArrayView.h                                                 (C) 2000-2025 */
/*                                                                           */
/* Types defining C array views.                                             */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_ARRAYVIEW_H
#define ARCANE_UTILS_ARRAYVIEW_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ArrayView.h"
#include "arccore/base/Span.h"

#include "arcane/utils/UtilsTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Applies padding at the end of the array \a ids.
 *
 * This method fills the elements of \a ids after the last value
 * so that \a ids has a valid number of elements that is a multiple of the size
 * of a Simd vector.
 *
 * The array associated with the view must have enough allocated memory
 * to fill the padding elements, otherwise it leads to an
 * array overflow.
 *
 * The padding is done using the value of the last element
 * of \a ids.
 *
 * For example, if ids.size()==5 and the Simd vector size is 8,
 * then ids[5], ids[6], and ids[7] are filled with the value of ids[4].
 */
//@{
extern ARCANE_UTILS_EXPORT void
applySimdPadding(ArrayView<Int32> ids);

extern ARCANE_UTILS_EXPORT void
applySimdPadding(ArrayView<Int16> ids);

extern ARCANE_UTILS_EXPORT void
applySimdPadding(ArrayView<Int64> ids);

extern ARCANE_UTILS_EXPORT void
applySimdPadding(ArrayView<Real> ids);
//@}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
