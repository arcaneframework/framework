// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#include <gtest/gtest.h>

#include "arcane/utils/ValueConvert.h"
#include "arcane/utils/internal/ValueConvertInternal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arcane;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename T> void _checkBad(const String& s)
{
  T x = {};
  bool is_bad = builtInGetValue(x, s);
  std::cout << "CheckBad S=" << s << " X=" << x << " is_bad?=" << is_bad << "\n";
  ASSERT_TRUE(is_bad);
}

void _checkDouble(const String& s, double expected_x)
{
  Real x = 0;
  bool is_bad = builtInGetValue(x, s);
  std::cout << "S=" << s << " X=" << x << " is_bad?=" << is_bad << "\n";
  ASSERT_FALSE(is_bad);
  ASSERT_EQ(x, expected_x);
}

void _checkNaN(const String& s)
{
  Real x = 0;
  bool is_bad = builtInGetValue(x, s);
  std::cout << "S=" << s << " X=" << x << " is_bad?=" << is_bad << "\n";
  ASSERT_FALSE(is_bad);
  ASSERT_TRUE(std::isnan(x));
}

/*!
 * \brief Vérifie que \a value et \a expected_value sont strictements identiques.
 *
 * \return true si la valeur attendu n'est pas NaN.
 */
void _checkValidDouble(double value, double expected_value)
{
  // Pour NaN, on ne peut pas faire la comparaison.
  if (std::isnan(expected_value)) {
    ASSERT_TRUE(std::isnan(value)) << "value " << value << " is not 'nan'";
    return;
  }
  ASSERT_EQ(value, expected_value);
}

void _checkReal2(const String& s, Real2 expected_v)
{
  Real2 v = {};
  bool is_bad = builtInGetValue(v, s);
  std::cout << "S=" << s << " Real2=" << v << " is_bad?=" << is_bad << "\n";
  ASSERT_FALSE(is_bad) << "Can not convert '" << s << "' to Real2";
  bool do_test = false;
  _checkValidDouble(v.x, expected_v.x);
  _checkValidDouble(v.y, expected_v.y);
  if (do_test) {
    ASSERT_EQ(v, expected_v);
  }
}

void _checkReal3(const String& s, Real3 expected_v)
{
  Real3 v = {};
  bool is_bad = builtInGetValue(v, s);
  std::cout << "S=" << s << " Real3=" << v << " is_bad?=" << is_bad << "\n";
  ASSERT_FALSE(is_bad) << "Can not convert '" << s << "' to Real3";
  bool do_test = false;
  _checkValidDouble(v.x, expected_v.x);
  _checkValidDouble(v.y, expected_v.y);
  _checkValidDouble(v.z, expected_v.z);
  if (do_test) {
    ASSERT_EQ(v, expected_v);
  }
}

void _testDoubleConvert(bool use_from_chars)
{

  {
    // TODO: tester les autres conversions
    String s = "25e3";
    Int32 x = 0;
    bool is_bad = builtInGetValue(x, s);
    std::cout << "S=" << s << " X=" << x << " is_bad?=" << is_bad << "\n";
    ASSERT_TRUE(is_bad);
  }
  // Avec la version 'from_chars', convertir une chaîne vide est une erreur
  // mais pas avec la version historique.
  if (use_from_chars)
    _checkBad<double>("");
  _checkDouble("-0x1.81e03f705857bp-16", -2.3e-05);
  _checkDouble("0x1.81e03f705857bp-16", 2.3e-05);
  if (!use_from_chars)
    _checkDouble("+1.23e42", 1.23e42);
  _checkBad<double>("d2");
  _checkBad<double>("2.3w");

  {
    Real inf_x = std::numeric_limits<Real>::infinity();
    _checkDouble("inf", inf_x);
    _checkDouble("INF", inf_x);
    _checkDouble("infinity", inf_x);
    _checkDouble("INFINITY", inf_x);
  }
  {
    Real minus_inf_x = -std::numeric_limits<Real>::infinity();
    _checkDouble("-inf", minus_inf_x);
    _checkDouble("-INF", minus_inf_x);
    _checkDouble("-infinity", minus_inf_x);
    _checkDouble("-INFINITY", minus_inf_x);
  }

  {
    _checkNaN("nan");
    _checkNaN("NAN");
    _checkNaN("NaN");
    _checkNaN("nAN");
  }
  {
    String s3 = "23123.132e123";
    Real total = 0.0;
    Int32 nb_iter = 1000000 * 10;
    nb_iter = 1;
    for (Int32 i = 0; i < nb_iter; ++i) {
      Real v = {};
      builtInGetValue(v, s3);
      total += v;
    }
    std::cout << "Total=" << total << "\n";
  }
}

void _testReal2Convert(bool use_same_that_real)
{
  impl::arcaneSetUseSameValueConvertForAllReal(use_same_that_real);
  Real v_nan = std::numeric_limits<double>::quiet_NaN();
  _checkReal2("2.3e1 -1.2", Real2(2.3e1, -1.2));
  if (use_same_that_real)
    _checkReal2("-1.3 nan", Real2(-1.3, v_nan));
  _checkBad<Real2>("2.3 1.2w");
  _checkBad<Real2>("2.3x");
  _checkBad<Real2>(" y2.3 1.2");
}

void _testReal3Convert(bool use_same_that_real)
{
  impl::arcaneSetUseSameValueConvertForAllReal(use_same_that_real);
  Real v_nan = std::numeric_limits<double>::quiet_NaN();
  Real v_inf = std::numeric_limits<double>::infinity();
  _checkReal3("2.3e1 -1.2 1.5", Real3(2.3e1, -1.2, 1.5));
  if (use_same_that_real) {
    _checkReal3("-1.3 nan 4.6", Real3(-1.3, v_nan, 4.6));
    _checkReal3("1.3 4.2 inf", Real3(1.3, 4.2, v_inf));
    _checkReal3("-2.1 -1.5 1.0e5", Real3(-2.1, -1.5, 1.0e5));
    //_checkReal3("-2.1 -1.5 +1.0e5", Real3(-2.1, -1.5, 1.0e5));
  }
  _checkBad<Real3>("2.3 1.2w");
  _checkBad<Real3>("2.3x");
  _checkBad<Real3>("2.3 1.2");
  _checkBad<Real3>("2.3 -1.2ee2 4.5");
  _checkBad<Real3>("z2.3 -1.2e2 -2323.3");
  _checkBad<Real3>("2.3 -1.2e2 -2323.3x");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(ValueConvert, Basic)
{
  std::cout << "TEST_ValueConvert Basic\n";
  impl::arcaneSetValueConvertVerbosity(1);

#if defined(ARCANE_HAS_CXX20)
  impl::arcaneSetIsValueConvertUseFromChars(true);
  _testDoubleConvert(true);
  _testReal2Convert(true);
  _testReal2Convert(false);
  _testReal3Convert(true);
  _testReal3Convert(false);
#endif

  impl::arcaneSetIsValueConvertUseFromChars(false);
  _testDoubleConvert(false);
  _testReal2Convert(true);
  _testReal2Convert(false);
  _testReal3Convert(true);
  _testReal3Convert(false);
}

TEST(ValueConvert, TryParse)
{
  {
    String s2;
    auto v = Convert::Type<Int32>::tryParse(s2);
    ASSERT_FALSE(v.has_value());
  }

  {
    String s2;
    auto v = Convert::Type<Int32>::tryParseIfNotEmpty(s2, 4);
    ASSERT_TRUE(v.has_value());
    ASSERT_EQ(v, 4);
  }

  {
    String s2("2.3");
    auto v = Convert::Type<Real>::tryParse(s2);
    ASSERT_EQ(v, 2.3);
  }

  {
    String s2("2.3w");
    auto v = Convert::Type<Real>::tryParse(s2);
    ASSERT_FALSE(v.has_value());
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
