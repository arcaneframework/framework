// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MathApfloat.h                                               (C) 2000-2020 */
/*                                                                           */
/* Fonctions mathématiques diverses pour le type apfloat.                    */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_MATHAPFLOAT_H
#define ARCANE_UTILS_MATHAPFLOAT_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include <apfloat.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::math
{
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Logarithme népérien de \a v.
 */
inline apfloat
log(apfloat v)
{
#ifdef ARCANE_CHECK_MATH
  if (v==0.0 || v<0.0)
    arcaneMathError(Convert::toDouble(v),"log");
#endif
  return ::log(v);
}

/*!
 * \brief Arondir \a v à l'entier immédiatement inférieur.
 */
inline apfloat
floor(apfloat v)
{
  return ::floor(v);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Exponentielle de \a v.
 */
inline apfloat
exp(apfloat v)
{
  return ::exp(v);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Racine carrée de \a v.
 */
inline apfloat
sqrt(apfloat v)
{
#ifdef ARCANE_CHECK_MATH
  if (v<0.)
    arcaneMathError(Convert::toDouble(v),"sqrt");
#endif
  return ::sqrt(v);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Fonction puissance.
 *
 * Calcul \a x à la puissance \a y.
 *
 * \pre x>=0 ou y entier
 */
inline apfloat
pow(apfloat x,apfloat y)
{
#ifdef ARCANE_CHECK_MATH
  // Arguments invalides si x est négatif et y non entier
  if (x<0.0 && ::floor(y)!=y)
    arcaneMathError(Convert::toDouble(x),Convert::toDouble(y),"pow");
#endif
  return ::pow(x,y);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Retourne le minimum de deux réels.
 * \ingroup GroupMathUtils
 */
inline apfloat
min(apfloat a,apfloat b)
{
  return ( (a<b) ? a : b );
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Retourne le maximum de deux réels.
 * \ingroup GroupMathUtils
 */
inline apfloat
max(apfloat a,apfloat b)
{
  return ( (a<b) ? b : a );
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Retourne la valeur absolue d'un réel.
 * \ingroup GroupMathUtils
 */
inline apfloat
abs(apfloat a)
{
  return ::abs(a);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::math

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
