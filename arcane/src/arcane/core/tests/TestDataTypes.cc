// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#include <gtest/gtest.h>

#include "arcane/utils/NumericTypes.h"
#include "arcane/utils/String.h"
#include "arcane/utils/ArgumentException.h"

#include "arcane/core/datatype/DataTypes.h"

#include "arccore/serialize/ISerializer.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arcane;

#define TEST_DATA_TYPE(a) \
  ASSERT_EQ(String(dataTypeName(DT_##a)), #a); \
  ASSERT_EQ(dataTypeFromName(#a), DT_##a); \
  ASSERT_EQ(dataTypeSize(DT_##a), sizeof(a))

TEST(ArcaneDataTypes, Misc)
{
  TEST_DATA_TYPE(Byte);
  TEST_DATA_TYPE(Real);
  TEST_DATA_TYPE(Int16);
  TEST_DATA_TYPE(Int32);
  TEST_DATA_TYPE(Int64);
  TEST_DATA_TYPE(Real2);
  TEST_DATA_TYPE(Real3);
  TEST_DATA_TYPE(Real2x2);
  TEST_DATA_TYPE(Real3x3);
  TEST_DATA_TYPE(BFloat16);
  TEST_DATA_TYPE(Float16);
  TEST_DATA_TYPE(Float32);
  TEST_DATA_TYPE(Int8);

  ASSERT_EQ(String(dataTypeName(DT_Unknown)), "Unknown");
  ASSERT_EQ(dataTypeFromName("Unknown"), DT_Unknown);

  ASSERT_EQ(String(dataTypeName(DT_String)), "String");
  ASSERT_EQ(dataTypeFromName("String"), DT_String);
  EXPECT_THROW(dataTypeSize(DT_String), ArgumentException);

  EXPECT_THROW(dataTypeSize((eDataType)50), ArgumentException);
  ASSERT_EQ(dataTypeFromName("Toto"), DT_Unknown);

  ASSERT_EQ((int)eDataType::DT_Byte, (int)Arccore::ISerializer::DT_Byte);
  ASSERT_EQ((int)eDataType::DT_Real, (int)Arccore::ISerializer::DT_Real);
  ASSERT_EQ((int)eDataType::DT_Int16, (int)Arccore::ISerializer::DT_Int16);
  ASSERT_EQ((int)eDataType::DT_Int32, (int)Arccore::ISerializer::DT_Int32);
  ASSERT_EQ((int)eDataType::DT_Int64, (int)Arccore::ISerializer::DT_Int64);
  ASSERT_EQ((int)eDataType::DT_Float32, (int)Arccore::ISerializer::DT_Float32);
  ASSERT_EQ((int)eDataType::DT_Float16, (int)Arccore::ISerializer::DT_Float16);
  ASSERT_EQ((int)eDataType::DT_BFloat16, (int)Arccore::ISerializer::DT_BFloat16);
  ASSERT_EQ((int)eDataType::DT_Int8, (int)Arccore::ISerializer::DT_Int8);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
