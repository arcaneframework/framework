﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* PropertiesUnitTest.cc                                       (C) 2000-2021 */
/*                                                                           */
/* Service de test des propriétés.                                           */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/OStringStream.h"
#include "arcane/utils/ScopedPtr.h"
#include "arcane/utils/MD5HashAlgorithm.h"

#include "arcane/BasicUnitTest.h"
#include "arcane/FactoryService.h"
#include "arcane/Timer.h"
#include "arcane/Properties.h"
#include "arcane/SerializeBuffer.h"
#include "arcane/IPropertyMng.h"

#include "arcane/tests/ArcaneTestGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANETEST_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arcane;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Service de test des propriétés.
 */
class PropertiesUnitTest
: public BasicUnitTest
{
 public:

  explicit PropertiesUnitTest(const ServiceBuildInfo& cb);
  ~PropertiesUnitTest();

 public:

  void initializeTest() override {}
  void executeTest() override;

 private:

  template <typename DataType> void
  _compare(ConstArrayView<DataType> v1,ConstArrayView<DataType> v2)
  {
    Integer s1 = v1.size();
    Integer s2 = v2.size();
    if (s1!=s2)
      ARCANE_FATAL("Bad size");
    for( Integer i=0; i<s1; ++i )
      if (v1[i]!=v2[i])
        ARCANE_FATAL("Bad value");
  }

  void _executeTest(Properties& p,Properties& p2);
  String _getHash(Span<const Byte> bytes);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_CASE_OPTIONS_NOAXL_FACTORY(PropertiesUnitTest,IUnitTest,PropertiesUnitTest);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

PropertiesUnitTest::
PropertiesUnitTest(const ServiceBuildInfo& mb)
: BasicUnitTest(mb)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

PropertiesUnitTest::
~PropertiesUnitTest()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void PropertiesUnitTest::
executeTest()
{
  IPropertyMng* pm = subDomain()->propertyMng();
  Properties p(pm,"TEST");
  Properties p2(pm,"TEST2");
  Properties p3(p2,"SUB_TEST");
  _executeTest(p,p2);

  {
    SerializeBuffer sb;
    sb.setMode(ISerializer::ModeReserve);
    pm->serialize(&sb);
    sb.allocateBuffer();
    sb.setMode(ISerializer::ModePut);
    pm->serialize(&sb);
    {
      Span<Byte> bytes = sb.globalBuffer();
      info() << "size=" << bytes.size();
    }

    sb.setMode(ISerializer::ModeGet);
    pm->serialize(&sb);
  }
  {
    info() << "TEST: PROPERTY_MNG: PREPARE_FOR_DUMP";
    // Vérifie que deux sauvegardes donnent le même hash
    // et fait de même après une relecture.
    UniqueArray<Byte> saved_bytes;
    pm->writeTo(saved_bytes);
    String hash1 = _getHash(saved_bytes);
    UniqueArray<Byte> saved_bytes2;
    pm->writeTo(saved_bytes2);
    String hash2 = _getHash(saved_bytes2);
    if (hash1!=hash2)
      ARCANE_FATAL("Bad hash for properties hash1={0} hash2={1}",hash1,hash2);
    info() << "TEST: PROPERTY_MNG: READ_FROM_DUMP";
    pm->readFrom(saved_bytes);
    UniqueArray<Byte> saved_bytes3;
    pm->writeTo(saved_bytes3);
    String hash3 = _getHash(saved_bytes3);
    if (hash1!=hash3)
      ARCANE_FATAL("Bad hash for properties hash1={0} hash3={1}",hash1,hash3);
  }

  {
    OStringStream o;
    pm->print(o());
    info() << "Properties\n" << o.str();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String PropertiesUnitTest::
_getHash(Span<const Byte> bytes)
{
  MD5HashAlgorithm hash_algo;
  UniqueArray<Byte> output_hash;
  hash_algo.computeHash64(bytes,output_hash);
  String hash_string = Convert::toHexaString(output_hash);
  info() << "PROPERTY_HASH n=" << bytes.size() << " =" << hash_string;
  return hash_string;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#define DO_TEST(CppDataType,DataType,Value)     \
  {\
  info() << "Testing property " #DataType;\
    CppDataType x = Value;                               \
    CppDataType xget;                               \
    p.set##DataType(#DataType,x);                      \
    if (p.get##DataType(#DataType)!=x)                         \
      ARCANE_FATAL("Bad value"); \
    p.get(#DataType,xget);                         \
    if (xget!=x)                         \
      ARCANE_FATAL("Bad value 2");  \
  }

#define DO_TEST_ARRAY(DataType,Value)\
  {\
  info() << "Testing array property " #DataType;\
    DataType x = Value; \
    DataType x2;                               \
    p.set(#DataType,x);                      \
    p.get(#DataType,x2);                      \
    _compare(x2.constView(),x.constView());             \
    p.set(#DataType,x);                      \
    p.get(#DataType,x2);                      \
    _compare(x2.constView(),x.constView());             \
  }

#define DO_COMPARE(DataType)\
  {\
  info() << "Comparing property " #DataType;\
  if (p.get##DataType(#DataType)!=p2.get##DataType(#DataType))          \
    ARCANE_FATAL("Bad compare for type " #DataType); \
  }

#define DO_COMPARE_ARRAY(DataType)\
  {\
    info() << "Comparing array property " #DataType;  \
    DataType x1;                               \
    DataType x2;                               \
    p.get(#DataType,x1);                      \
    p2.get(#DataType,x2);                      \
    _compare(x1.constView(),x2.constView());             \
  }

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void PropertiesUnitTest::
_executeTest(Properties& p,Properties& p2)
{
  DO_TEST(Int32,Int32,25);
  DO_TEST(String,String,"TITI");
  DO_TEST(Int64,Int64,82);
  DO_TEST(Real,Real,23.2);
  DO_TEST(bool,Bool,true);

  UniqueArray<Int32> i32x;
  i32x.add(2);
  i32x.add(5);

  UniqueArray<Int64> i64x;
  i64x.add(29);
  i64x.add(35);

  UniqueArray<String> strx;
  strx.add("TITI");
  strx.add("TOTO");
  strx.add("TUTU");

  UniqueArray<Real> realx;
  realx.add(2.7);
  realx.add(-1.3);
  realx.add(7.9);
  realx.add(-122.12424);

  DO_TEST_ARRAY(UniqueArray<Int32>,i32x);
  DO_TEST_ARRAY(UniqueArray<Int64>,i64x);
  DO_TEST_ARRAY(UniqueArray<String>,strx);
  DO_TEST_ARRAY(UniqueArray<Real>,realx);

  {
    OStringStream o;
    p.print(o());
    info() << o.str();
  }

  SerializeBuffer sb;
  sb.setMode(ISerializer::ModeReserve);
  p.serialize(&sb);
  sb.allocateBuffer();
  sb.setMode(ISerializer::ModePut);
  p.serialize(&sb);

  sb.setMode(ISerializer::ModeGet);
  p2.serialize(&sb);

  {
    OStringStream o;
    p2.print(o());
    info() << "P2=" << o.str();
  }

  DO_COMPARE(Int32);
  DO_COMPARE(String);
  DO_COMPARE(Int64);
  DO_COMPARE(Real);
  DO_COMPARE(Bool);

  DO_COMPARE_ARRAY(Int32UniqueArray);
  DO_COMPARE_ARRAY(StringUniqueArray);
  DO_COMPARE_ARRAY(Int64UniqueArray);
  DO_COMPARE_ARRAY(RealUniqueArray);
  DO_COMPARE_ARRAY(BoolUniqueArray);


}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANETEST_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
