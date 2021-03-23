/*---------------------------------------------------------------------------*/
/* Math.h                                                      (C) 2000-2006 */
/*                                                                           */
/* Fonctions mathématiques diverses.                                         */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_MATH_H
#define ARCANE_UTILS_MATH_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcaneGlobal.h"
#include <math.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
  \brief Espace de nom pour les fonctions mathématiques.
 
  Cet espace de nom contient toutes les fonctions mathématiques utilisées
  par le code.
*/
namespace math
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Logarithme népérien de \a v.
 */
inline double
log(double v)
{
#ifdef ARCANE_CHECK
  if (v==0.0 || v<0.0)
    arcaneMathError(v,"log");
#endif
  return ::log(v);
}

/*!
 * \brief Logarithme népérien de \a v.
 */
inline long double
log(long double v)
{
#ifdef ARCANE_CHECK
  if (v==0.0 || v<0.0)
    arcaneMathError(v,"log");
#endif
  return ::logl(v);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Logarithme décimal de \a v.
 */
inline double
log10(double v)
{
#ifdef ARCANE_CHECK
  if (v==0.0 || v<0.0)
    arcaneMathError(v,"log");
#endif
  return ::log10(v);
}

/*!
 * \brief Logarithme décimal de \a v.
 */
inline long double
log10(long double v)
{
#ifdef ARCANE_CHECK
  if (v==0.0 || v<0.0)
    arcaneMathError(v,"log");
#endif
  return ::log10l(v);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Arondir \a v à l'entier immédiatement inférieur.
 */
inline double
floor(double v)
{
  return ::floor(v);
}

/*!
 * \brief Arondir \a v à l'entier immédiatement inférieur.
 */
inline long double
floor(long double v)
{
  return ::floorl(v);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Exponentielle de \a v.
 */
inline double
exp(double v)
{
  return ::exp(v);
}
/*!
 * \brief Exponentielle de \a v.
 */
inline long double
exp(long double v)
{
  return ::expl(v);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Racine carrée de \a v.
 */
inline double
sqrt(double v)
{
#ifdef ARCANE_CHECK
  if (v<0.)
    arcaneMathError(v,"sqrt");
#endif
  return ::sqrt(v);
}
/*!
 * \brief Racine carrée de \a v.
 */
inline long double
sqrt(long double v)
{
#ifdef ARCANE_CHECK
  if (v<0.)
    arcaneMathError(v,"sqrt");
#endif
  return ::sqrtl(v);
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
inline double
pow(double x,double y)
{
#ifdef ARCANE_CHECK
  // Arguments invalides si x est négatif et y non entier
  if (x<0.0 && ::floor(y)!=y)
    arcaneMathError(x,y,"pow");
#endif
  return ::pow(x,y);
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
inline long double
pow(long double x,long double y)
{
#ifdef ARCANE_CHECK
  // Arguments invalides si x est négatif et y non entier
  if (x<0.0 && ::floorl(y)!=y)
    arcaneMathError(x,y,"pow");
#endif
  return ::powl(x,y);
}
/*!
 * \brief Fonction puissance.
 *
 * Calcul \a x à la puissance \a y.
 *
 * \pre x>=0 ou y entier
 */
inline long double
pow(double x,long double y)
{
#ifdef ARCANE_CHECK
  // Arguments invalides si x est négatif et y non entier
  if (x<0.0 && ::floorl(y)!=y)
    arcaneMathError(x,y,"pow");
#endif
  return ::powl(x,y);
}
/*!
 * \brief Fonction puissance.
 *
 * Calcul \a x à la puissance \a y.
 *
 * \pre x>=0 ou y entier
 */
inline long double
pow(long double x,double y)
{
#ifdef ARCANE_CHECK
  // Arguments invalides si x est négatif et y non entier
  if (x<0.0 && ::floor(y)!=y)
    arcaneMathError(x,y,"pow");
#endif
  return ::powl(x,y);
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
template<class T> inline T
min(const T& a,const T& b)
{
  return ( (a<b) ? a : b );
}
/*!
 * \brief Retourne le minimum de deux réels.
 * \ingroup GroupMathUtils
 */
inline long double
min(long double a,long double b)
{
  return ( (a<b) ? a : b );
}
/*!
 * \brief Retourne le minimum de deux réels.
 * \ingroup GroupMathUtils
 */
inline long double
min(double a,long double b)
{
  return ( (a<b) ? a : b );
}
/*!
 * \brief Retourne le minimum de deux réels.
 * \ingroup GroupMathUtils
 */
inline long double
min(long double a,double b)
{
  return ( (a<b) ? a : b );
}
/*!
 * \brief Retourne le minimum de deux réels.
 * \ingroup GroupMathUtils
 */
inline double
min(double a,double b)
{
  return ( (a<b) ? a : b );
}
/*!
 * \brief Retourne le minimum de deux réels.
 * \ingroup GroupMathUtils
 */
inline float
min(float a,float b)
{
  return ( (a<b) ? a : b );
}
/*!
 * \brief Retourne le minimum de deux entiers.
 * \ingroup GroupMathUtils
 */
inline int
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
template<class T> inline T
max(const T& a,const T& b)
{
  return ( (a<b) ? b : a );
}
/*!
 * \brief Retourne le maximum de deux réels.
 * \ingroup GroupMathUtils
 */
inline long double
max(long double a,long double b)
{
  return ( (a<b) ? b : a );
}
/*!
 * \brief Retourne le maximum de deux réels.
 * \ingroup GroupMathUtils
 */
inline long double
max(double a,long double b)
{
  return ( (a<b) ? b : a );
}
/*!
 * \brief Retourne le maximum de deux réels.
 * \ingroup GroupMathUtils
 */
inline long double
max(long double a,double b)
{
  return ( (a<b) ? b : a );
}
/*!
 * \brief Retourne le maximum de deux entiers.
 * \ingroup GroupMathUtils
 */
inline long
max(unsigned long a,unsigned long b)
{
  return ( (a<b) ? b : a );
}
/*!
 * \brief Retourne le maximum de deux réels.
 * \ingroup GroupMathUtils
 */
inline double
max(double a,double b)
{
  return ( (a<b) ? b : a );
}
/*!
 * \brief Retourne le maximum de deux réels.
 * \ingroup GroupMathUtils
 */
inline float
max(float a,float b)
{
  return ( (a<b) ? b : a );
}
/*!
 * \brief Retourne le maximum de deux Int32
 * \ingroup GroupMathUtils
 */
inline Int32
max(Int32 a,Int32 b)
{
  return ( (a<b) ? b : a );
}
/*!
 * \brief Retourne le maximum de deux Int32
 * \ingroup GroupMathUtils
 */
inline Int64
max(Int32 a,Int64 b)
{
  return ( (a<b) ? b : a );
}
/*!
 * \brief Retourne le maximum de deux Int64
 * \ingroup GroupMathUtils
 */
inline Int64
max(Int64 a,Int32 b)
{
  return ( (a<b) ? b : a );
}
/*!
 * \brief Retourne le maximum de deux Int64
 * \ingroup GroupMathUtils
 */
inline Int64
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
inline long double
abs(long double a)
{
  return ::fabsl(a);
  //return (a>0.L) ? a : (-a);
}
/*!
 * \brief Retourne la valeur absolue d'un réel.
 * \ingroup GroupMathUtils
 */
inline double
abs(double a)
{
  return ::fabs(a);
}
/*!
 * \brief Retourne la valeur absolue d'un réel.
 * \ingroup GroupMathUtils
 */
inline float
abs(float a)
{
  return ::fabsf(a);
  //return (a>0.F) ? a : (-a);
}
/*!
 * \brief Retourne la valeur absolue d'un 'int'.
 * \ingroup GroupMathUtils
 */
inline int
abs(int a)
{
  return (a>0) ? a : (-a);
}

/*!
 * \brief Retourne la valeur absolue d'un 'long'.
 * \ingroup GroupMathUtils
 */
inline long
abs(long a)
{
  return (a>0L) ? a : (-a);
}

/*!
 * \brief Retourne la valeur absolue d'un 'long'.
 * \ingroup GroupMathUtils
 */
inline long long
abs(long long a)
{
  return (a>0LL) ? a : (-a);
}

} // namespace math

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifdef ARCANE_REAL_USE_APFLOAT
#include "arcane/utils/MathApfloat.h"
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

