// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Convert.h                                                   (C) 2000-2025 */
/*                                                                           */
/* Fonctions pour convertir un type en un autre.                             */
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
 * \brief Converti un tableau d'octet en sa représentation hexadécimale.
 *
 * Chaque octet de \a input est converti en deux caractères hexadécimaux,
 * appartenant à [0-9a-f].
 */
extern ARCANE_UTILS_EXPORT String
toHexaString(ByteConstArrayView input);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Converti un tableau d'octet en sa représentation hexadécimale.
 *
 * Chaque octet de \a input est converti en deux caractères hexadécimaux,
 * appartenant à [0-9a-f].
 */
extern ARCANE_UTILS_EXPORT String
toHexaString(Span<const std::byte> input);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Converti un réel en sa représentation hexadécimale.
 *
 * Chaque octet de \a input est converti en deux caractères hexadécimaux,
 * appartenant à [0-9a-f].
 */
extern ARCANE_UTILS_EXPORT String
toHexaString(Real input);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Converti un entier 64 bits sa représentation hexadécimale.
 *
 * Chaque octet de \a input est converti en deux caractères hexadécimaux,
 * appartenant à [0-9a-f].
 * Le tableau \a output doit avoir au moins 16 éléments.
 */
extern ARCANE_UTILS_EXPORT void
toHexaString(Int64 input, Span<Byte> output);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Convert

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

