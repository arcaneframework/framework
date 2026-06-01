// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Convert.h                                                   (C) 2000-2025 */
/*                                                                           */
/* Functions to convert one type to another.                                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_CONVERT_H
#define ARCANE_UTILS_CONVERT_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/UtilsTypes.h"
#include "arccore/base/Convert.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Convert
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Converts a byte array to its hexadecimal representation.
 *
 * Each byte of \a input is converted into two hexadecimal characters,
 * belonging to [0-9a-f].
 */
extern ARCANE_UTILS_EXPORT String
toHexaString(ByteConstArrayView input);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Converts a byte array to its hexadecimal representation.
 *
 * Each byte of \a input is converted into two hexadecimal characters,
 * belonging to [0-9a-f].
 */
extern ARCANE_UTILS_EXPORT String
toHexaString(Span<const std::byte> input);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Converts a real number to its hexadecimal representation.
 *
 * Each byte of \a input is converted into two hexadecimal characters,
 * belonging to [0-9a-f].
 */
extern ARCANE_UTILS_EXPORT String
toHexaString(Real input);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Converts a 64-bit integer to its hexadecimal representation.
 *
 * Each byte of \a input is converted into two hexadecimal characters,
 * belonging to [0-9a-f].
 * The \a output array must have at least 16 elements.
 */
extern ARCANE_UTILS_EXPORT void
toHexaString(Int64 input, Span<Byte> output);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Convert

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
