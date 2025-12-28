// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* JSONUnitTest.cc                                             (C) 2000-2025 */
/*                                                                           */
/* Test du lecteur/ecrivain JSON.                                            */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/JSONWriter.h"
#include "arcane/utils/JSONReader.h"

#include "arcane/BasicUnitTest.h"
#include "arcane/FactoryService.h"
#include "arcane/IIOMng.h"

#include "arcane/tests/ArcaneTestGlobal.h"

#define RAPIDJSON_HAS_STDSTRING 1
#include "arccore/common/internal/json/rapidjson/writer.h"
#include "arccore/common/internal/json/rapidjson/document.h"
#include "arccore/common/internal/json/rapidjson/stringbuffer.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace ArcaneTest
{
using namespace Arcane;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Service de test du lecteur/ecrivain JSON.
 */
class JSONUnitTest
: public BasicUnitTest
{
 public:

  explicit JSONUnitTest(const ServiceBuildInfo& cb);
  ~JSONUnitTest();

 public:

  void initializeTest() override {}
  void executeTest() override;

 private:

  void _testJSON_1();
  void _testJSON_2();
  void _testJSON_Read_1();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_CASE_OPTIONS_NOAXL_FACTORY(JSONUnitTest,IUnitTest,JSONUnitTest);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

JSONUnitTest::
JSONUnitTest(const ServiceBuildInfo& mb)
: BasicUnitTest(mb)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

JSONUnitTest::
~JSONUnitTest()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void JSONUnitTest::
executeTest()
{
  _testJSON_1();
  _testJSON_2();
  _testJSON_Read_1();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void JSONUnitTest::
_testJSON_1()
{
  info() << "TEST_JSON 1!";
  //using namespace rapidjson;
  // 1. Parse a JSON string into DOM.
  const char* json = "{\"project\":\"rapidjson\",\"stars\":10, \"double\":\"0x1.999999999999ap-4\"}";
  rapidjson::Document d;
  d.Parse(json);
  if (d.HasParseError()){
    info() << "ERROR: " << d.GetParseError();
  }

  // 2. Modify it by DOM.
  rapidjson::Value& s = d["stars"];
  s.SetInt(s.GetInt() + 1);
  //Value& s2 = d["double"];
  //info() << "JSON: Double" << s2.GetDouble();
  s.SetInt(s.GetInt() + 1);
  // 3. Stringify the DOM
  rapidjson::StringBuffer buffer;
  rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
  d.Accept(writer);
  // Output {"project":"rapidjson","stars":11}
  info() << "JSON: " << buffer.GetString();

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void JSONUnitTest::
_testJSON_2()
{
  info() << "TEST_JSON 2!";

  JSONWriter serializer;
  Int32UniqueArray int_values = { 2, 5, -3 };
  RealUniqueArray real_values = { 4.0, 1.0, -9.2, 3.4e7, 0.1 };
  Real real_value = -2.3e-5;
  String str1("test1");
  String null_str;
  serializer.beginObject();
  serializer.write("int_values",int_values);
  serializer.write("real_values",real_values);
  serializer.write("int64_value",(Int64)-25424);
  serializer.write("uint64_value",(UInt64)-3131);
  serializer.write("string_value",str1);
  serializer.write("null_string_value",null_str);
  serializer.write("real_hex_value",real_value);
  serializer.write("real_value",real_value);
  serializer.endObject();
  StringView buf = serializer.getBuffer();
  info() << "BUF=" << buf;
  {
    std::ofstream ofile("test1.json");
    ofile << buf;
  }
  
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void JSONUnitTest::
_testJSON_Read_1()
{
  info() << "TEST_JSON_Read_1!";
  String str1("test1");

  IIOMng* io_mng = subDomain()->ioMng();
  UniqueArray<Byte> bytes;
  bool is_bad = io_mng->localRead("test1.json",bytes,false);
  if (is_bad)
    ARCANE_FATAL("Can not read file");

  JSONDocument document;
  document.parse(bytes);

  JSONValue doc_root = document.root();
  JSONKeyValue v1 = doc_root.keyValueChild("toto");
  info() << "IS_V1?=" << v1.null() << " name=" << v1.name() << " values=" << v1.value().valueAsStringView();
  {
    String strv = v1.value().value();
    if (!strv.null())
      ARCANE_FATAL("Value 'v1'  should be null (v={0})", strv);
    String strview = v1.value().valueAsStringView();
    if (!strview.empty())
      ARCANE_FATAL("Value 'v1' should be empty (v={0})", strview);
  }
  JSONKeyValue v2 = doc_root.keyValueChild("real_values");
  info() << "IS_V2?=" << v2.null() << " name=" << v2.name() << " values=" << v2.value().valueAsStringView();
  {
    String strv = v2.value().value();
    if (strv.null())
      ARCANE_FATAL("Value 'v2' should not be null (v={0})", strv);
    String strview = v2.value().valueAsStringView();
    if (strview.empty())
      ARCANE_FATAL("Value 'v2' should not be empty (v={0})", strview);
  }
  {
    JSONValue json_str = doc_root.expectedChild("string_value");
    String strv = json_str.value();
    if (strv!=str1)
      ARCANE_FATAL("Bad value for 'strv' v='{0}' expected='{1}'", strv, str1);
    String strview = json_str.valueAsStringView();
    if (strview!=str1)
      ARCANE_FATAL("Bad value for 'strview' v='{0}' expected='{1}'", strview, str1);
  }
  {
    JSONValue v3 = doc_root.child("int64_value");
    Int64 v = v3.valueAsInt64();
    if (v!=(-25424))
      ARCANE_FATAL("Bad value for Int64 = {0}",v);
  }
  {
    JSONValue v4 = doc_root.child("real_value");
    Real v = v4.valueAsReal();
    Real expected_value = -2.3e-5;
    if (v!=expected_value)
      ARCANE_FATAL("Bad value for Real v={0} expected={1}",v, expected_value);
  }

  for( auto& x : doc_root.keyValueChildren() ){
    info() << "CHILD NAME=" << x.name();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
