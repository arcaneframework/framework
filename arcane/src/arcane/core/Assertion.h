// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Assertion.h                                                 (C) 2000-2025 */
/*                                                                           */
/* Ensemble d'assertions utilisées pour les tests unitaires.                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ASSERTION_H
#define ARCANE_CORE_ASSERTION_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceInfo.h"
#include "arcane/utils/Numeric.h"
#include "arcane/core/ArcaneTypes.h"
#include "arcane/core/ArcaneException.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class IParallelMng;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \file Assertion.h
 *
 * Ce fichier contient les assertions utilisées pour les tests unitaires.
 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe de base pour assertions dans les tests unitaires.
 */
class ARCANE_CORE_EXPORT Assertion
{
 private:

  void _checkAssertion(bool is_error,const TraceInfo& where,
                       const String& expected, const String& actual, IParallelMng* pm);

 public:

  void fail(const TraceInfo& where)
  {
    throw AssertionException(where);
  }

  //! Lance une exception AssertException si \a condition est faux.
 void assertTrue(const TraceInfo& where, bool condition, IParallelMng* pm = nullptr)
  {
    bool is_error = (!condition);
    _checkAssertion(is_error, where, "true", "false", pm);
  }

  //! Lance une exception AssertException si \a condition est vrai.
  void assertFalse(const TraceInfo& where, bool condition, IParallelMng* pm = nullptr)
  {
    bool is_error = (condition);
    _checkAssertion(is_error, where, "false", "true", pm);
  }

  /*!
   * Surcharge pour les chaînes de caractères. Cela permet de comparer des
   * String avec des 'const char*' par exemple.
   */
  void assertEqual(const TraceInfo& where, const String& expected,
                   const String& actual, IParallelMng* pm = nullptr)
  {
	  bool is_error = (expected != actual);
    _checkAssertion(is_error,where,expected,actual,pm);
  }

  template<typename T>
  void assertEqual(const TraceInfo& where, const T& expected, const T& actual, IParallelMng* pm = nullptr)
  {
	  // utilisation de l'operateur == pour les types numeriques et non !=
	  bool is_error = (! (expected == actual));
    _checkAssertion(is_error,where,String::fromNumber(expected),String::fromNumber(actual),pm);
  }

  template<typename T>
  void assertNearlyEqual(const TraceInfo& where, const T& expected,
                         const T& actual, IParallelMng* pm = nullptr)
  {
	  bool is_error = (!math::isNearlyEqual(expected,actual));
    _checkAssertion(is_error,where,String::fromNumber(expected),String::fromNumber(actual),pm);
  }

  template<typename T>
  void assertNearlyZero(const TraceInfo& where, const T& actual, IParallelMng* pm = nullptr)
  {
	  bool is_error = (!math::isNearlyZero(actual));
    _checkAssertion(is_error, where, "0", String::fromNumber(actual),pm);
  }

  template<typename T>
  void assertNearlyEqualWithEpsilon(const TraceInfo& where, const T& expected,
                                    const T& actual,const T& epsilon, IParallelMng* pm = nullptr)
  {
	  bool is_error = (!math::isNearlyEqualWithEpsilon(expected,actual,epsilon));
    _checkAssertion(is_error, where, String::fromNumber(expected), String::fromNumber(actual), pm);
  }

  template<typename T>
  void assertNearlyZeroWithEpsilon(const TraceInfo& where, const T& actual,
                                   const T& epsilon, IParallelMng* pm = nullptr)
  {
	  bool is_error = (!math::isNearlyZeroWithEpsilon(actual,epsilon));
    _checkAssertion(is_error, where, "0", String::fromNumber(actual), pm);
  }
};

#define FAIL fail(A_FUNCINFO)

/*!
 * \brief Vérifie que \a condition est vrai.
 */
#define ASSERT_TRUE(condition) \
assertTrue(A_FUNCINFO, condition)

/*!
 * \brief Vérifie en parallèle que \a condition est vrai.
 */
#define PARALLEL_ASSERT_TRUE(condition, parallel_mng) \
  assertTrue(A_FUNCINFO, condition, parallel_mng)

/*!
 * \brief Vérifie que \a condition est faux.
 */
#define ASSERT_FALSE(condition) \
assertFalse(A_FUNCINFO, condition)

/*!
 * \brief Vérifie que \a condition est faux.
 */
#define PARALLEL_ASSERT_FALSE(condition, parallel_mng)  \
  assertFalse(A_FUNCINFO, condition, parallel_mng)

/*!
 * \brief Vérifie que \a expected et \a actual sont égaux.
 * La comparaison se fait via l'opérator==() du type.
 */
#define ASSERT_EQUAL(expected, actual) \
assertEqual(A_FUNCINFO, expected, actual)

/*!
 * \brief Vérifie que \a expected et \a actual sont égaux.
 * La comparaison se fait via l'opérator==() du type.
 */
#define PARALLEL_ASSERT_EQUAL(expected, actual, parallel_mng)  \
  assertEqual(A_FUNCINFO, expected, actual, parallel_mng)

/*!
 * \brief Vérifie que \a expected et \a actual sont presques égaux.
 * \sa math::isNearlyEqual()
 */
#define ASSERT_NEARLY_EQUAL(expected, actual) \
assertNearlyEqual(A_FUNCINFO, expected, actual)

/*!
 * \brief Vérifie que \a expected et \a actual sont presques égaux.
 * \sa math::isNearlyEqual()
 */
#define PARALLEL_ASSERT_NEARLY_EQUAL(expected, actual, parallel_mng) \
  assertNearlyEqual(A_FUNCINFO, expected, actual, parallel_mng)

/*!
 * \brief Vérifie que \a actual est presque égal à zéro
 * à l'epsilon standard
 * \sa math::isNearlyZero()
 */
#define ASSERT_NEARLY_ZERO(actual) \
assertNearlyZero(A_FUNCINFO, actual)

/*!
 * \brief Vérifie que \a actual est presque égal à zéro
 * à l'epsilon standard
 * \sa math::isNearlyZero()
 */
#define PARALLEL_ASSERT_NEARLY_ZERO(actual, parallel_mng) \
  assertNearlyZero(A_FUNCINFO, actual, parallel_mng)

/*!
 * \brief Vérifie que \a expected et \a actual sont presques égaux.
 * \sa math::isNearlyEqualWithEpsilon()
 */
#define ASSERT_NEARLY_EQUAL_EPSILON(expected, actual, epsilon) \
assertNearlyEqualWithEpsilon(A_FUNCINFO, expected, actual, epsilon)

/*!
 * \brief Vérifie que \a expected et \a actual sont presques égaux.
 * \sa math::isNearlyEqualWithEpsilon()
 */
#define PARALLEL_ASSERT_NEARLY_EQUAL_EPSILON(expected, actual, epsilon, parallel_mng) \
  assertNearlyEqualWithEpsilon(A_FUNCINFO, expected, actual, epsilon, parallel_mng)

/*!
 * \brief Vérifie que \a actual est presque égal à zéro
 * à l'epsilon spécifié \a epsilon
 * \sa math::isNearlyZero()
 */
#define ASSERT_NEARLY_ZERO_EPSILON(actual,epsilon) \
assertNearlyZeroWithEpsilon(A_FUNCINFO, actual, epsilon)

/*!
 * \brief Vérifie que \a actual est presque égal à zéro
 * à l'epsilon spécifié \a epsilon
 * \sa math::isNearlyZero()
 */
#define PARALLEL_ASSERT_NEARLY_ZERO_EPSILON(actual,epsilon, parallel_mng)  \
  assertNearlyZeroWithEpsilon(A_FUNCINFO, actual, epsilon, parallel_mng)

/*!
 * \brief Vérifie que \a expected et \a actual sont égaux.
 * La comparaison se fait via l'opérator==() du type.
 * \deprecated Utiliser ASSERT_EQUAL() (sans le S)
 */
#define ASSERT_EQUALS(expected, actual) \
assertEqual(A_FUNCINFO, expected, actual)

/*!
 * \brief Vérifie que \a expected et \a actual sont presques égaux.
 * \sa math::isNearlyEqual()
 * \deprecated Utiliser ASSERT_NEARLY_EQUAL() (sans le S)
 */
#define ASSERT_NEARLY_EQUALS(expected, actual) \
assertNearlyEqual(A_FUNCINFO, expected, actual)

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
