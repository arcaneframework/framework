// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* StringUtils.h                                               (C) 2000-2025 */
/*                                                                           */
/* Utility functions for strings.                                            */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_STRINGUTILS_H
#define ARCCORE_BASE_STRINGUTILS_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/String.h"

#include <vector>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::StringUtils
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Returns the conversion of the instance into UTF-16BE encoding.
 *
 * The returned vector does not contain a null terminator.
 */
extern "C++" ARCCORE_BASE_EXPORT std::vector<UChar>
asUtf16BE(const String& str);

/*!
 * \brief Returns the conversion of \a str to std::wstring.
 *
 * This function is only supported for the Win32 platform.
 */
extern "C++" ARCCORE_BASE_EXPORT std::wstring
convertToStdWString(const String& str);

/*!
 * \brief Converts \a wstr into a String.
 *
 * This function is only supported for the Win32 platform.
 */
extern "C++" ARCCORE_BASE_EXPORT String
convertToArcaneString(const std::wstring_view& wstr);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::StringUtils

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore::StringUtils
{
using Arcane::StringUtils::asUtf16BE;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
