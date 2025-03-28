// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#include <gtest/gtest.h>

#include "arcane/utils/ArcaneCxx20.h"

#include <atomic>
#include <charconv>
#include <iostream>
#include <limits>
#include <cmath>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arcane;

template <class T>
concept integral = std::is_integral_v<T>;

TEST(TestCxx20,Atomic)
{
  Int32 x = 25;
  std::atomic_ref<Int32> ax(x);
  ax.fetch_add(32);
  ASSERT_EQ(x,57);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{
template<integral DataType> DataType _testAdd(DataType a,DataType b)
{
  return a+b;
}
}

TEST(TestCxx20,Concept)
{
  Int32 a = 12;
  Int32 b = -48;
  ASSERT_EQ(_testAdd(a,b),(a+b));
}


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{

double _doTestDouble(double value)
{
  std::cout << "ValueDirectStream=" << value << "\n";
  std::ostringstream ostr;
  ostr << value << "\n";
  std::string str0 = ostr.str();
  std::cout << "O_STR=" << str0;

  {
    std::istringstream istr(str0);
    double v = -1.0;
    std::cout << "IS_GOOD?=" << istr.good() << "\n";
    istr >> std::ws >> v;
    std::cout << "IS_GOOD?=" << istr.good() << "\n";
    std::cout << "V=" << v << "\n";
    char* str_end = nullptr;
    double v2 = std::strtod(str0.data(), &str_end);
    std::cout << "ReadWith 'strtod' =" << v2 << "\n";
  }

  double result = {};
  {
    auto [ptr, ec] = std::from_chars(str0.data(), str0.data() + str0.length(), result);
    if (ec == std::errc())
      std::cout << "Result: " << result << ", ptr -> " << (ptr - str0.data()) << '\n';
    else if (ec == std::errc::invalid_argument)
      std::cout << "This is not a number.\n";
    else if (ec == std::errc::result_out_of_range)
      std::cout << "This number is larger than an int.\n";
  }
  return result;
}

} // namespace

TEST(TestFromChars, Real)
{
  std::cout << "TEST_ValueConvert 'Real' \n";
  double d_inf = std::numeric_limits<double>::infinity();
  double d_nan = std::numeric_limits<double>::quiet_NaN();
  std::cout << "Infinity=" << d_inf << "\n";
  std::cout << "NaN=" << d_nan << "\n";

  double d_t0 = 1.2345;
  std::cout << "** Test: " << d_t0 << "\n";
  double r_t0 = _doTestDouble(d_t0);
  ASSERT_EQ(r_t0, d_t0);

  std::cout << "** Test: Infinity\n";
  double r_inf = _doTestDouble(d_inf);
  ASSERT_EQ(r_inf, d_inf);

  std::cout << "** Test: NaN\n";
  double r_nan = _doTestDouble(d_nan);
  ASSERT_TRUE(std::isnan(r_nan));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
