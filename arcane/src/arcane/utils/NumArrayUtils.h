// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* NumArrayUtils.h                                             (C) 2000-2025 */
/*                                                                           */
/* Utility functions for NumArray.                                           */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_NUMARRAYUTILS_H
#define ARCANE_UTILS_NUMARRAYUTILS_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/UtilsTypes.h"
#include "arcane/utils/MDDim.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::NumArrayUtils
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Fills \a v with the values from \a input.
 *
 * \a v will be resized to the number of values contained in the file.
 */
extern "C++" ARCANE_UTILS_EXPORT void
readFromText(NumArray<double, MDDim1>& v, std::istream& input);

/*!
 * \brief Fills \a v with the values from \a input.
 *
 * \a v will be resized to the number of values contained in the file.
 */
extern "C++" ARCANE_UTILS_EXPORT void
readFromText(NumArray<Int32, MDDim1>& v, std::istream& input);

/*!
 * \brief Fills \a v with the values from \a input.
 *
 * \a v will be resized to the number of values contained in the file.
 */
extern "C++" ARCANE_UTILS_EXPORT void
readFromText(NumArray<Int64, MDDim1>& v, std::istream& input);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::NumArrayUtils

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
