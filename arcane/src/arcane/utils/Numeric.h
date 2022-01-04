// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Numeric.h                                                   (C) 2000-2020 */
/*                                                                           */
/* Définitions des constantes numériques.                                    */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_DATATYPE_NUMERIC_H
#define ARCANE_DATATYPE_NUMERIC_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Limits.h"
#include "arcane/utils/Math.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Opérations de comparaisons pour un type numérique T
 *
 * Cette classe définit juste un opérateur de comparaison pour le
 * type 'T' paramètre template. Il existe deux types de comparaisons:
 * - les comparaisons exactes (isEqual());
 * - les comparaisons approximatives (isNearlyEqual()).
 *
 * Les deux types de comparaisons sont identiques, sauf pour les
 * types flottants ou équivalents. Dans ce cas, la comparaison exacte
 * compare bit à bit les deux valeurs et la comparaison approximative
 * considère que deux nombres sont égaux si leur différence relative est
 * inférieure à un epsilon près.
 */
template<class T>
class TypeEqualT
{
 public:
  /*!
   * \brief Compare \a a à zéro.
   * \retval true si \a a vaut zéro à un epsilon près,
   * \retval false sinon.
   */
  constexpr ARCCORE_HOST_DEVICE static bool isNearlyZero (const T& a)
  {
    return (a==T());
  }

  /*!
   * \brief Compare \a a à zéro.
   * \retval true si \a a vaut exactement zéro,
   * \retval false sinon.
   */
  constexpr ARCCORE_HOST_DEVICE static bool isZero (const T& a)
  {
    return (a==T());
  }

  /*!
   * \brief Compare \a a à \a b.
   * \retval true si \a a et \b sont égaux à un epsilon près,
   * \retval false sinon.
   */
  constexpr ARCCORE_HOST_DEVICE static bool isNearlyEqual(const T& a,const T& b)
  {
    return (a==b);
  }

  /*!
   * \brief Compare \a a à \a b.
   * \retval true si \a a et \b sont égaux à un epsilon près,
   * \retval false sinon.
   */
  constexpr ARCCORE_HOST_DEVICE static bool isNearlyEqualWithEpsilon(const T& a,const T& b,const T&)
  {
    return (a==b);
  }

  /*!
   * \brief Compare \a a à \a b.
   * \retval true si \a a et \b sont exactements égaux,
   * \retval false sinon.
   */
  constexpr ARCCORE_HOST_DEVICE static bool isEqual(const T& a,const T& b)
  {
    return (a==b);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Définit l'opérateur == pour les flottants.
 *
 * \note A terme, devrait utiliser pour l'epsilon la classe 'numeric_limits'
 * de la STL quand cela sera implémenté.
 */
template<class T>
class FloatEqualT
{
 private:
  constexpr ARCCORE_HOST_DEVICE static T nepsilon() { return FloatInfo<T>::nearlyEpsilon(); }
 public:
  constexpr ARCCORE_HOST_DEVICE static bool isNearlyZero(T a)
  {
    return ( (a<0.) ? a>-nepsilon() : a<nepsilon() );
  }
  
  /*!
   * \brief Compare \a a à zéro à \a epsilon près.
   * 
   * \a epsilon doit être positif.
   *
   * \retval true si abs(a)<epilon
   * \retval false sinon
   */
  constexpr ARCCORE_HOST_DEVICE static bool isNearlyZeroWithEpsilon(T a,T epsilon)
  {
    return ( (a<0.) ? a>-epsilon : a<epsilon );
  }
  
  /*! \brief Compare \a avec \a b*epsilon.
   * \warning b doit être positif. */
  ARCCORE_HOST_DEVICE static bool isNearlyZero(T a,T b)
  {
    return ( (a<0.) ? a>-(b*nepsilon()) : a<(b*nepsilon()) );
  }

  constexpr ARCCORE_HOST_DEVICE static bool isTrueZero(T a) { return (a==FloatInfo<T>::zero()); }
  constexpr ARCCORE_HOST_DEVICE static bool isZero(T a) { return (a==FloatInfo<T>::zero()); }
  constexpr ARCCORE_HOST_DEVICE static bool isNearlyEqual(T a,T b)
  {
    T s = math::abs(a) + math::abs(b);
    T d = a - b;
    return (d==FloatInfo<T>::zero()) ? true : isNearlyZero(d/s);
  }
  constexpr ARCCORE_HOST_DEVICE static bool isNearlyEqualWithEpsilon(T a,T b,T epsilon)
  {
    T s = math::abs(a) + math::abs(b);
    T d = a - b;
    return (d==FloatInfo<T>::zero()) ? true : isNearlyZeroWithEpsilon(d/s,epsilon);
  }
  constexpr ARCCORE_HOST_DEVICE static bool isEqual(T a,T b)
  {
    return a==b;
  }
};
/*!
 * \internal
 * \brief spécialisation de TypeEqualT pour le type <tt>float</tt>.
 */
template<>
class TypeEqualT<float>
: public FloatEqualT<float>
{};

/*!
 * \internal
 * \brief spécialisation de TypeEqualT pour le type <tt>double</tt>.
 */
template<>
class TypeEqualT<double>
: public FloatEqualT<double>
{};

/*!
 * \internal
 * \brief spécialisation de TypeEqualT pour le type <tt>long double</tt>.
 */
template<>
class TypeEqualT<long double>
: public FloatEqualT<long double>
{};

#ifdef ARCANE_REAL_NOT_BUILTIN
/*!
 * \internal
 * \brief spécialisation de TypeEqualT pour le type <tt>Real</tt>.
 */
template<>
class TypeEqualT<Real>
: public FloatEqualT<Real>
{};
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace math
{

/*!
 * \brief Teste si deux valeurs sont à un peu près égales.
 * Pour les types entiers, cette fonction est équivalente à IsEqual().
 * Dans le cas de types réels, les deux nombres sont considérés comme égaux
 * si et seulement si la valeur absolue de leur différence relative est
 * inférieure à un epsilon donné. Cet
 * epsilon est égal à float_info<_Type>::nearlyEpsilon().
 * \retval true si les deux valeurs sont égales,
 * \retval false sinon.
 */
template<class _Type> constexpr ARCCORE_HOST_DEVICE inline bool
isNearlyEqual(const _Type& a,const _Type& b)
{
  return TypeEqualT<_Type>::isNearlyEqual(a,b);
}

//! Surcharge pour les reels
constexpr ARCCORE_HOST_DEVICE inline bool
isNearlyEqual(Real a,Real b)
{
  return TypeEqualT<Real>::isNearlyEqual(a,b);
}

/*!
 * \brief Teste si deux valeurs sont à un peu près égales.
 * Pour les types entiers, cette fonction est équivalente à IsEqual().
 * Dans le cas de types réels, les deux nombres sont considérés comme égaux
 * si et seulement si la valeur absolue de leur différence relative est
 * inférieure à \a epsilon.
 *
 * \retval true si les deux valeurs sont égales,
 * \retval false sinon.
 */
template<class _Type> constexpr ARCCORE_HOST_DEVICE inline bool
isNearlyEqualWithEpsilon(const _Type& a,const _Type& b,const _Type& epsilon)
{
  return TypeEqualT<_Type>::isNearlyEqualWithEpsilon(a,b,epsilon);
}

//! Surcharge pour les reels
ARCCORE_HOST_DEVICE constexpr inline bool
isNearlyEqualWithEpsilon(Real a,Real b,Real epsilon)
{
  return TypeEqualT<Real>::isNearlyEqualWithEpsilon(a,b,epsilon);
}

/*!
 * \brief Teste l'égalité bit à bit entre deux valeurs.
 * \retval true si les deux valeurs sont égales,
 * \retval false sinon.
 */
template<class _Type> constexpr ARCCORE_HOST_DEVICE inline bool
isEqual(const _Type& a,const _Type& b)
{
  return TypeEqualT<_Type>::isEqual(a,b);
}

//! Surcharge pour les reels
ARCCORE_HOST_DEVICE constexpr inline bool
isEqual(Real a,Real b)
{
  return TypeEqualT<Real>::isEqual(a,b);
}

/*!
 * \brief Teste si une valeur est à peu près égale à zéro à un epsilon près.
 *
 * Pour les types entiers, cette fonction est équivalente à IsZero().
 * Dans le cas de types réels, la valeur est considérée comme égale à
 * zéro si et seulement si sa valeur absolue est inférieure à un epsilon
 * donné par la fonction float_info<_Type>::nearlyEpsilon().
 * \retval true si les deux valeurs sont égales,
 * \retval false sinon.
 */
template<class _Type> constexpr ARCCORE_HOST_DEVICE inline bool
isNearlyZeroWithEpsilon(const _Type& a,const _Type& epsilon)
{
  return TypeEqualT<_Type>::isNearlyZeroWithEpsilon(a,epsilon);
}

/*!
 * \brief Teste si une valeur est à peu près égale à zéro à l'epsilon standard près.
 *
 * L'epsilon standard est celui retourné par FloatInfo<_Type>::nearlyEpsilon().
 *
 * \sa isNearlyZero(const _Type& a,const _Type& epsilon).
 */
template<class _Type> constexpr ARCCORE_HOST_DEVICE inline bool
isNearlyZero(const _Type& a)
{
  return TypeEqualT<_Type>::isNearlyZero(a);
}

/*!
 * \brief Teste si une valeur est exactement égale à zéro.
 * \retval true si \a vaut zéro,
 * \retval false sinon.
 */
template<class _Type> constexpr ARCCORE_HOST_DEVICE inline bool
isZero(const _Type& a)
{
  return TypeEqualT<_Type>::isZero(a);
}
} // namespace math

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
