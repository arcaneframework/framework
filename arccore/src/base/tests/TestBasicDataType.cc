// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#include <gtest/gtest.h>

#include "arccore/base/BasicDataType.h"
#include "arccore/base/String.h"
#include "arccore/base/BFloat16.h"
#include "arccore/base/Float16.h"
#include "arccore/base/Float128.h"
#include "arccore/base/Int128.h"

#include <string>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arccore;

namespace
{
void _check(eBasicDataType basic_type, int value, int nb_byte, const char* const_char_name)
{
  const String name = const_char_name;
  ASSERT_EQ((int)basic_type, value);
  ASSERT_EQ(basicDataTypeSize(basic_type), nb_byte);
  String returned_name = basicDataTypeName(basic_type);
  ASSERT_EQ(returned_name, name);
  {
    std::ostringstream ostr;
    ostr << basic_type;
    String s2(ostr.str());
    ASSERT_EQ(name, s2);
  }
  {
    std::string str(name.toStdStringView());
    std::istringstream istr(str);
    eBasicDataType expected_type = eBasicDataType::Unknown;
    istr >> expected_type;
    ASSERT_FALSE(istr.bad());
  }
  {
    bool has_error = false;
    eBasicDataType dt = basicDataTypeFromName(const_char_name, has_error);
    ASSERT_FALSE(has_error);
    ASSERT_EQ(basic_type, dt);
    [[maybe_unused]] eBasicDataType dt2 = basicDataTypeFromName("Test1", has_error);
    ASSERT_TRUE(has_error);
  }
  {
    eBasicDataType dt = basicDataTypeFromName(const_char_name);
    ASSERT_EQ(basic_type, dt);
  }
}

} // namespace

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(BasicDataType,Misc)
{
  ASSERT_EQ(NB_BASIC_DATA_TYPE,12);

  _check(eBasicDataType::Unknown,0,0,"Unknown");
  _check(eBasicDataType::Byte,1,1,"Byte");
  _check(eBasicDataType::Float16,2,2,"Float16");
  _check(eBasicDataType::Float32,3,4,"Float32");
  _check(eBasicDataType::Float64,4,8,"Float64");
  _check(eBasicDataType::Float128,5,16,"Float128");
  _check(eBasicDataType::Int16,6,2,"Int16");
  _check(eBasicDataType::Int32,7,4,"Int32");
  _check(eBasicDataType::Int64,8,8,"Int64");
  _check(eBasicDataType::Int128,9,16,"Int128");
  _check(eBasicDataType::BFloat16,10,2,"BFloat16");
  _check(eBasicDataType::Int8,11,1,"Int8");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{
class RefValue
{
 public:

  float original_value = 0.0f;
  uint16_t raw_value = 0;
  float converted_value = 0.0f;

 public:

  static void checkBFloat16(const RefValue& v)
  {
    checkConvertBFloat16(v);
    checkDirectBFloat16(v);
  }

  static void checkConvertBFloat16(const RefValue& v)
  {
    uint16_t x = Arccore::impl::convertBFloat16ToUint16Impl(v.original_value);
    float cv = Arccore::impl::convertToBFloat16Impl(x);
    std::cout << "ConvertBF16: V=" << v.original_value << " expected_raw=" << v.raw_value << " expected_cv=" << v.converted_value << "\n";
    EXPECT_FLOAT_EQ(x, v.raw_value);
    EXPECT_FLOAT_EQ(cv, v.converted_value);
  }

  static void checkDirectBFloat16(const RefValue& v)
  {
    BFloat16 bf(v.original_value);
    float cv = bf;
    std::cout << "DirectBF16: V=" << v.original_value << " expected_cv=" << v.converted_value << "\n";
    EXPECT_FLOAT_EQ(cv, v.converted_value);
  }

  static void checkFloat16(const RefValue& v)
  {
    checkConvertFloat16(v);
    checkDirectFloat16(v);
  }

  static void checkConvertFloat16(const RefValue& v)
  {
    uint16_t x = Arccore::impl::convertFloat16ToUint16Impl(v.original_value);
    float cv = Arccore::impl::convertToFloat16Impl(x);
    std::cout << "ConvertF16: V=" << v.original_value << " expected_raw=" << v.raw_value << " expected_cv=" << v.converted_value << "\n";
    EXPECT_FLOAT_EQ(x, v.raw_value);
    EXPECT_FLOAT_EQ(cv, v.converted_value);
  }

  static void checkDirectFloat16(const RefValue& v)
  {
    Float16 bf(v.original_value);
    float cv = bf;
    std::cout << "DirectF16: V=" << v.original_value << " expected_cv=" << v.converted_value << "\n";
    EXPECT_FLOAT_EQ(cv, v.converted_value);
  }
};

} // namespace

TEST(BasicDataType, BFloat16)
{
  BFloat16 x(2.3f);
  std::cout << "BF16_X=" << x << "\n";
  x = 1.4f;
  std::cout << "BF16_X=" << x << "\n";

  uint16_t x2 = Arccore::impl::convertBFloat16ToUint16Impl(2.3f);
  std::cout << "BF16_X2=" << x2 << "\n";
  float x2_f = Arccore::impl::convertToBFloat16Impl(x2);
  std::cout << "BF16_X2F=" << x2_f << "\n";

  RefValue::checkBFloat16({ 2.3f, 16403, 2.296875f });
  RefValue::checkBFloat16({ 1.4f, 16307, 1.3984375f });
  RefValue::checkBFloat16({ -1.2e-2f, 48197, -0.012023926f });

  BFloat16 bf1(4.5f);
  BFloat16 bf2(-1.2f);
  BFloat16 bf3(4.5f);
  ASSERT_EQ(bf3, bf1);
  ASSERT_TRUE(bf2 < bf1);
  ASSERT_FALSE(bf1 < bf2);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

TEST(BasicDataType, Float16)
{
  Float16 x(2.3f);
  std::cout << "F16_X=" << x << "\n";
  x = 1.4f;
  std::cout << "F16_X=" << x << "\n";

  uint16_t x2 = Arccore::impl::convertFloat16ToUint16Impl(2.3f);
  std::cout << "F16_X2=" << x2 << "\n";
  float x2_f = Arccore::impl::convertToFloat16Impl(x2);
  std::cout << "F16_X2F=" << x2_f << "\n";

  RefValue::checkFloat16({ 2.3f, 16538, 2.3007812f });
  RefValue::checkFloat16({ 1.4f, 15770, 1.4003906f });
  RefValue::checkFloat16({ -1.2e-2f, 41509, -0.012001038f });

  Float16 bf1(4.5f);
  Float16 bf2(-1.2f);
  Float16 bf3(4.5f);
  ASSERT_EQ(bf3, bf1);
  ASSERT_TRUE(bf2 < bf1);
  ASSERT_FALSE(bf1 < bf2);
}

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wlanguage-extension-token"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"

TEST(BasicDataType, Float128)
{
  Float128 a = 1.0;
  Float128 b = 2.0;
  Float128 c = a + b;
  double c_as_double = static_cast<double>(c);
  std::cout << "F128=" << c_as_double << "\n";
  ASSERT_EQ(sizeof(Float128), 16);
}

TEST(BasicDataType, Int128)
{
  Int128 a = 1;
  Int128 b = 2;
  Int128 c = a + b;
  std::cout << "I128=" << static_cast<int64_t>(c) << "\n";
  std::cout << "sizeof(Int128) = " << sizeof(Int128) << "\n";
  std::cout << "sizeof(intmax_t) = " << sizeof(intmax_t) << "\n";
  ASSERT_EQ(sizeof(Int128), 16);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
