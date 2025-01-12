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

namespace
{

void _checkBadDouble(const String& s)
{
  Real x = 0;
  bool is_bad = builtInGetValue(x, s);
  std::cout << "S=" << s << " X=" << x << " is_bad?=" << is_bad << "\n";
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
    _checkBadDouble("");
  _checkDouble("-0x1.81e03f705857bp-16", -2.3e-05);
  _checkDouble("0x1.81e03f705857bp-16", 2.3e-05);

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

} // namespace

TEST(ValueConvert, Basic)
{
  std::cout << "TEST_ValueConvert Basic\n";
  impl::arcaneSetValueConvertVerbosity(1);

#if defined(ARCANE_HAS_CXX20)
  impl::arcaneSetIsValueConvertUseFromChars(true);
  _testDoubleConvert(true);
#endif

  impl::arcaneSetIsValueConvertUseFromChars(false);
  _testDoubleConvert(false);
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
