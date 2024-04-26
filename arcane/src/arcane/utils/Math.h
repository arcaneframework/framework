// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Math.h                                                      (C) 2000-2024 */
/*                                                                           */
/* Fonctions mathématiques diverses.                                         */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_MATH_H
#define ARCANE_UTILS_MATH_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/UtilsTypes.h"
#include "arcane/utils/Convert.h"

#include <cmath>
#include <cstdlib>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
  \brief Espace de nom pour les fonctions mathématiques.
 
  Cet espace de nom contient toutes les fonctions mathématiques utilisées
  par le code.
*/
namespace Arcane::math
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Logarithme népérien de \a v.
 */
ARCCORE_HOST_DEVICE inline double
log(double v)
{
#ifdef ARCANE_CHECK_MATH
  if (v==0.0 || v<0.0)
    arcaneMathError(v,"log");
#endif
  return std::log(v);
}

/*!
 * \brief Logarithme népérien de \a v.
 */
ARCCORE_HOST_DEVICE inline long double
log(long double v)
{
#ifdef ARCANE_CHECK_MATH
  if (v==0.0 || v<0.0)
    arcaneMathError(v,"log");
#endif
  return std::log(v);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Logarithme décimal de \a v.
 */
ARCCORE_HOST_DEVICE inline double
log10(double v)
{
#ifdef ARCANE_CHECK_MATH
  if (v==0.0 || v<0.0)
    arcaneMathError(v,"log");
#endif
  return std::log10(v);
}

/*!
 * \brief Logarithme décimal de \a v.
 */
ARCCORE_HOST_DEVICE inline long double
log10(long double v)
{
#ifdef ARCANE_CHECK_MATH
  if (v==0.0 || v<0.0)
    arcaneMathError(v,"log");
#endif
  return std::log10(v);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Arondir \a v à l'entier immédiatement inférieur.
 */
ARCCORE_HOST_DEVICE inline double
floor(double v)
{
  return std::floor(v);
}

/*!
 * \brief Arondir \a v à l'entier immédiatement inférieur.
 */
ARCCORE_HOST_DEVICE inline long double
floor(long double v)
{
  return std::floor(v);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Exponentielle de \a v.
 */
ARCCORE_HOST_DEVICE inline double
exp(double v)
{
  return std::exp(v);
}
/*!
 * \brief Exponentielle de \a v.
 */
ARCCORE_HOST_DEVICE inline long double
exp(long double v)
{
  return std::exp(v);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Racine carrée de \a v.
 */
ARCCORE_HOST_DEVICE inline double
sqrt(double v)
{
#ifdef ARCANE_CHECK_MATH
  if (v<0.)
    arcaneMathError(v,"sqrt");
#endif
  return std::sqrt(v);
}
/*!
 * \brief Racine carrée de \a v.
 */
ARCCORE_HOST_DEVICE inline long double
sqrt(long double v)
{
#ifdef ARCANE_CHECK_MATH
  if (v<0.)
    arcaneMathError(v,"sqrt");
#endif
  return std::sqrt(v);
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
ARCCORE_HOST_DEVICE inline double
pow(double x,double y)
{
#ifdef ARCANE_CHECK_MATH
  // Arguments invalides si x est négatif et y non entier
  if (x<0.0 && ::floor(y)!=y)
    arcaneMathError(x,y,"pow");
#endif
  return std::pow(x,y);
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
ARCCORE_HOST_DEVICE inline long double
pow(long double x,long double y)
{
#ifdef ARCANE_CHECK_MATH
  // Arguments invalides si x est négatif et y non entier
  if (x<0.0 && ::floorl(y)!=y)
    arcaneMathError(x,y,"pow");
#endif
  return std::pow(x,y);
}
/*!
 * \brief Fonction puissance.
 *
 * Calcul \a x à la puissance \a y.
 *
 * \pre x>=0 ou y entier
 */
ARCCORE_HOST_DEVICE inline long double
pow(double x,long double y)
{
#ifdef ARCANE_CHECK_MATH
  // Arguments invalides si x est négatif et y non entier
  if (x<0.0 && ::floorl(y)!=y)
    arcaneMathError(x,y,"pow");
#endif
  return std::pow(x,y);
}
/*!
 * \brief Fonction puissance.
 *
 * Calcul \a x à la puissance \a y.
 *
 * \pre x>=0 ou y entier
 */
ARCCORE_HOST_DEVICE inline long double
pow(long double x,double y)
{
#ifdef ARCANE_CHECK_MATH
  // Arguments invalides si x est négatif et y non entier
  if (x<0.0 && ::floor(y)!=y)
    arcaneMathError(x,y,"pow");
#endif
  return std::pow(x,y);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Retourne le minimum de deux éléments.
 *
 * \ingroup GroupMathUtils
 *
 * Utilise l'opérateur < pour déterminer le minimum.
 */
template<class T> ARCCORE_HOST_DEVICE inline T
min(const T& a,const T& b)
{
  return ( (a<b) ? a : b );
}
/*!
 * \brief Retourne le minimum de deux réels.
 * \ingroup GroupMathUtils
 */
ARCCORE_HOST_DEVICE inline long double
min(long double a,long double b)
{
  return ( (a<b) ? a : b );
}
/*!
 * \brief Retourne le minimum de deux réels.
 * \ingroup GroupMathUtils
 */
ARCCORE_HOST_DEVICE inline long double
min(double a,long double b)
{
  return ( (a<b) ? a : b );
}
/*!
 * \brief Retourne le minimum de deux réels.
 * \ingroup GroupMathUtils
 */
ARCCORE_HOST_DEVICE inline long double
min(long double a,double b)
{
  return ( (a<b) ? a : b );
}
/*!
 * \brief Retourne le minimum de deux réels.
 * \ingroup GroupMathUtils
 */
ARCCORE_HOST_DEVICE inline double
min(double a,double b)
{
  return ( (a<b) ? a : b );
}
/*!
 * \brief Retourne le minimum de deux réels.
 * \ingroup GroupMathUtils
 */
ARCCORE_HOST_DEVICE inline float
min(float a,float b)
{
  return ( (a<b) ? a : b );
}
/*!
 * \brief Retourne le minimum de deux entiers.
 * \ingroup GroupMathUtils
 */
ARCCORE_HOST_DEVICE inline int
min(int a,int b)
{
  return ( (a<b) ? a : b );
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Retourne le maximum de deux éléments.
 *
 * \ingroup GroupMathUtils
 *
 * Utilise l'opérateur < pour déterminer le maximum.
 */
template<class T> ARCCORE_HOST_DEVICE inline T
max(const T& a,const T& b)
{
  return ( (a<b) ? b : a );
}
/*!
 * \brief Retourne le maximum de deux réels.
 * \ingroup GroupMathUtils
 */
ARCCORE_HOST_DEVICE inline long double
max(long double a,long double b)
{
  return ( (a<b) ? b : a );
}
/*!
 * \brief Retourne le maximum de deux réels.
 * \ingroup GroupMathUtils
 */
ARCCORE_HOST_DEVICE inline long double
max(double a,long double b)
{
  return ( (a<b) ? b : a );
}
/*!
 * \brief Retourne le maximum de deux réels.
 * \ingroup GroupMathUtils
 */
ARCCORE_HOST_DEVICE inline long double
max(long double a,double b)
{
  return ( (a<b) ? b : a );
}
/*!
 * \brief Retourne le maximum de deux entiers.
 * \ingroup GroupMathUtils
 */
ARCCORE_HOST_DEVICE inline unsigned long
max(unsigned long a,unsigned long b)
{
  return ( (a<b) ? b : a );
}
/*!
 * \brief Retourne le maximum de deux réels.
 * \ingroup GroupMathUtils
 */
ARCCORE_HOST_DEVICE inline double
max(double a,double b)
{
  return ( (a<b) ? b : a );
}
/*!
 * \brief Retourne le maximum de deux réels.
 * \ingroup GroupMathUtils
 */
ARCCORE_HOST_DEVICE inline float
max(float a,float b)
{
  return ( (a<b) ? b : a );
}
/*!
 * \brief Retourne le maximum de deux Int16
 * \ingroup GroupMathUtils
 */
ARCCORE_HOST_DEVICE inline Int16
max(Int16 a,Int16 b)
{
  return ( (a<b) ? b : a );
}
/*!
 * \brief Retourne le maximum de deux Int32
 * \ingroup GroupMathUtils
 */
ARCCORE_HOST_DEVICE inline Int32
max(Int32 a,Int32 b)
{
  return ( (a<b) ? b : a );
}
/*!
 * \brief Retourne le maximum de deux Int32
 * \ingroup GroupMathUtils
 */
ARCCORE_HOST_DEVICE inline Int64
max(Int32 a,Int64 b)
{
  return ( (a<b) ? b : a );
}
/*!
 * \brief Retourne le maximum de deux Int64
 * \ingroup GroupMathUtils
 */
ARCCORE_HOST_DEVICE inline Int64
max(Int64 a,Int32 b)
{
  return ( (a<b) ? b : a );
}
/*!
 * \brief Retourne le maximum de deux Int64
 * \ingroup GroupMathUtils
 */
ARCCORE_HOST_DEVICE inline Int64
max(Int64 a,Int64 b)
{
  return ( (a<b) ? b : a );
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Retourne la valeur absolue d'un réel.
 * \ingroup GroupMathUtils
 */
ARCCORE_HOST_DEVICE inline long double
abs(long double a)
{
  return std::abs(a);
}
/*!
 * \brief Retourne la valeur absolue d'un réel.
 * \ingroup GroupMathUtils
 */
ARCCORE_HOST_DEVICE inline double
abs(double a)
{
  return std::abs(a);
}
/*!
 * \brief Retourne la valeur absolue d'un réel.
 * \ingroup GroupMathUtils
 */
ARCCORE_HOST_DEVICE inline float
abs(float a)
{
  return std::abs(a);
}

/*!
 * \brief Retourne la valeur absolue d'un 'int'.
 * \ingroup GroupMathUtils
 */
ARCCORE_HOST_DEVICE inline short
abs(short a)
{
  return (a>0) ? a : (short)(-a);
}

/*!
 * \brief Retourne la valeur absolue d'un 'int'.
 * \ingroup GroupMathUtils
 */
ARCCORE_HOST_DEVICE inline int
abs(int a)
{
  return (a>0) ? a : (-a);
}

/*!
 * \brief Retourne la valeur absolue d'un 'long'.
 * \ingroup GroupMathUtils
 */
ARCCORE_HOST_DEVICE inline long
abs(long a)
{
  return (a>0L) ? a : (-a);
}

/*!
 * \brief Retourne la valeur absolue d'un 'long'.
 * \ingroup GroupMathUtils
 */
ARCCORE_HOST_DEVICE inline long long
abs(long long a)
{
  return (a>0LL) ? a : (-a);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Tronque la précision du réel \a v à \a nb_digit chiffres significatifs.
 *
 * Pour un réel double précision en IEEE 754, le nombre de bits significatif
 * est de 15 ou 16 suivant la valeur. Il est à noter qu'il n'est possible
 * de manière simple et rapide de tronquer la précision à une valeur donnée.
 * C'est pourquoi \a nb_digit représente un nombre de chiffre approximatif.
 * Notamment, il n'est pas possible de descendre en dessous de 8 chiffres
 * significatifs.
 *
 * Si \a nb_digit est inférieur ou égal à zéro ou supérieur à 15, c'est
 * la valeur \a v qui est retourné.
 */
extern ARCANE_UTILS_EXPORT double
truncateDouble(double v,Integer nb_digit);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Tronque la précision du tableau de réels \a values à
 * \a nb_digit chiffres significatifs.
 *
 * En sortie, chaque élément de \a values est modifié comme après appel
 * à truncateDouble(double v,Integer nb_digit).
 */
extern ARCANE_UTILS_EXPORT void
truncateDouble(ArrayView<double> values,Integer nb_digit);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::math

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifdef ARCANE_REAL_USE_APFLOAT
#include "arcane/utils/MathApfloat.h"
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
