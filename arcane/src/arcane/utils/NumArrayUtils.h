// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* NumArrayUtils.h                                             (C) 2000-2025 */
/*                                                                           */
/* Fonctions utilitaires pour NumArray.                                      */
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
 * \brief Remplit \a v avec les valeurs de \a input.
 *
 * \a v sera redimensionné aux nombre de valeurs contenues dans le fichier.
 */
extern "C++" ARCANE_UTILS_EXPORT void
readFromText(NumArray<double, MDDim1>& v, std::istream& input);

/*!
 * \brief Remplit \a v avec les valeurs de \a input.
 *
 * \a v sera redimensionné aux nombre de valeurs contenues dans le fichier.
 */
extern "C++" ARCANE_UTILS_EXPORT void
readFromText(NumArray<Int32, MDDim1>& v, std::istream& input);

/*!
 * \brief Remplit \a v avec les valeurs de \a input.
 *
 * \a v sera redimensionné aux nombre de valeurs contenues dans le fichier.
 */
extern "C++" ARCANE_UTILS_EXPORT void
readFromText(NumArray<Int64, MDDim1>& v, std::istream& input);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::NumArrayUtils

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
