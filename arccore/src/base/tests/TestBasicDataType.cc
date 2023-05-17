// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#include <gtest/gtest.h>

#include "arccore/base/BasicDataType.h"
#include "arccore/base/String.h"

#include <string>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arccore;

namespace
{
void _check(eBasicDataType basic_type,int value,int nb_byte,const char* const_char_name)
{
  const String name = const_char_name;
  ASSERT_EQ((int)basic_type,value);
  ASSERT_EQ(basicDataTypeSize(basic_type),nb_byte);
  String returned_name = basicDataTypeName(basic_type);
  ASSERT_EQ(returned_name,name);
  {
    std::ostringstream ostr;
    ostr << basic_type;
    String s2(ostr.str());
    ASSERT_EQ(name,s2);
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
    eBasicDataType dt = basicDataTypeFromName(const_char_name,has_error);
    ASSERT_FALSE(has_error);
    ASSERT_EQ(basic_type,dt);
    eBasicDataType dt2 = basicDataTypeFromName("Test1",has_error);
    ASSERT_TRUE(has_error);
  }
  {
    eBasicDataType dt = basicDataTypeFromName(const_char_name);
    ASSERT_EQ(basic_type,dt);
  }
}

}

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
