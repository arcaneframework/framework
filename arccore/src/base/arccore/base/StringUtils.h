// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* StringUtils.h                                               (C) 2000-2025 */
/*                                                                           */
/* Fonctions utilitaires sur les chaînes de caractères.                      */
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
 * \brief Retourne la conversion de l'instance dans l'encodage UTF-16BE.
 *
 * Le vecteur retourné ne contient pas de zéro terminal.
 */
extern "C++" ARCCORE_BASE_EXPORT std::vector<UChar>
asUtf16BE(const String& str);

/*!
 * \brief Retourne la conversion de \a str en std::wstring.
 *
 * Cette fonction n'est supportée que pour la plateforme Win32.
 */
extern "C++" ARCCORE_BASE_EXPORT std::wstring
convertToStdWString(const String& str);

/*!
 * \brief Convertie \a wstr en une String.
 *
 * Cette fonction n'est supportée que pour la plateforme Win32.
 */
extern "C++" ARCCORE_BASE_EXPORT String
convertToArcaneString(const std::wstring_view& wstr);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore::StringUtils

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore::StringUtils
{
using Arcane::StringUtils::asUtf16BE;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
