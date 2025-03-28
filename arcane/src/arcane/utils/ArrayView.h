// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ArrayView.h                                                 (C) 2000-2025 */
/*                                                                           */
/* Types définissant les vues de tableaux C.                                 */
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
 * \brief Applique à \a ids un remplissage en fin de tableau.
 *
 * Cette méthode remplit les éléments de \a ids après la dernière valeur
 * pour que \a ids ait un nombre d'éléments valide multiple de la taille
 * d'un vecteur Simd.
 *
 * Le tableau associé à la vue doit avoir sufisamment de mémoire allouée
 * pour remplir les éléments de padding sinon cela conduit à un
 * débordement de tableau.
 *
 * Le remplissage se fait avec comme valeur celle du dernier élément
 * valide de \a ids.
 *
 * Par exemple, si ids.size()==5 et que la taille de vecteur Simd est de 8,
 * alors ids[5], ids[6] et ids[7] sont remplis avec la valeur de ids[4].
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
