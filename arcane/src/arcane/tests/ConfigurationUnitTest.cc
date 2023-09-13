// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ConfigurationUnitTest.cc                                    (C) 2000-2020 */
/*                                                                           */
/* Service de test de la configuration.                                      */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/OStringStream.h"
#include "arcane/utils/ScopedPtr.h"
#include "arcane/utils/JSONReader.h"

#include "arcane/BasicUnitTest.h"
#include "arcane/FactoryService.h"
#include "arcane/IIOMng.h"
#include "arcane/IXmlDocumentHolder.h"
#include "arcane/XmlNode.h"

#include "arcane/IConfigurationSection.h"
#include "arcane/IConfiguration.h"
#include "arcane/IConfigurationMng.h"

#include "arcane/impl/ConfigurationReader.h"

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
class ConfigurationUnitTest
: public BasicUnitTest
{

 public:

  ConfigurationUnitTest(const ServiceBuildInfo& cb);
  ~ConfigurationUnitTest();

 public:

  void initializeTest() override {}
  void executeTest() override;

 private:

  template <typename DataType1,typename DataType2> void
  _checkEqual(const DataType1& v1,const DataType2& v2)
  {
    info() << "Check equal v1=" << v1 << " v2=" << v2;
    if (v1!=v2)
      ARCANE_FATAL("Bad value value={0} expected={1}",v1,v2);
  }
  void _checkValid(IConfiguration* configuration);
  void _executeTestJSON();
  void _executeTestXml();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_CASE_OPTIONS_NOAXL_FACTORY(ConfigurationUnitTest,IUnitTest,ConfigurationUnitTest);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ConfigurationUnitTest::
ConfigurationUnitTest(const ServiceBuildInfo& mb)
: BasicUnitTest(mb)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ConfigurationUnitTest::
~ConfigurationUnitTest()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ConfigurationUnitTest::
_checkValid(IConfiguration* configuration)
{
  IConfigurationSection* cf = configuration->mainSection();

  _checkEqual(cf->valueAsInt32("Section1.Section2.V1",120),250000);
  _checkEqual(cf->valueAsString("GlobalStr","None"),String("Toto"));
  _checkEqual(cf->valueAsString("NoValue","Titi"),String("Titi"));
  _checkEqual(cf->valueAsReal("GlobalReal",2.5),3.5);
  _checkEqual(cf->valueAsInt64("GlobalInt64",0),1234567890123);

  ScopedPtrT<IConfigurationSection> s1(configuration->createSection("Section1"));
  _checkEqual(s1->valueAsInt32("Section2.V1",320),250000);
  _checkEqual(s1->valueAsBool("Bool1",false),true);
  _checkEqual(s1->valueAsBool("Bool2",false),true);
  _checkEqual(s1->valueAsBool("NoBool3",true),true);
  _checkEqual(s1->valueAsBool("NoBool3",false),false);

  ScopedPtrT<IConfigurationSection> s2(configuration->createSection("Section1.Section2"));
  _checkEqual(s2->valueAsInt32("V1",320),250000);

  info() << "Dumping SubDomain configuration:";
  subDomain()->configuration()->dump();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ConfigurationUnitTest::
executeTest()
{
  _executeTestXml();
  _executeTestJSON();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ConfigurationUnitTest::
_executeTestXml()
{
  IApplication* app = subDomain()->application();
  IConfigurationMng* cm = app->configurationMng();

  ScopedPtrT<IConfiguration> configuration(cm->createConfiguration());

  String cval = "<?xml version=\"1.0\"?>\n"
  "<configuration>\n"
  " <add name=\"Section1.Section2.V1\" value=\"250000\" />\n"
  " <add name=\"GlobalStr\" value=\"Toto\" />\n"
  " <add name=\"GlobalReal\" value=\"3.5\" />\n"
  " <add name=\"GlobalInt64\" value=\"1234567890123\" />\n"
  " <add name=\"Section1.Bool1\" value=\"true\" />\n"
  " <add name=\"Section1.Bool2\" value=\"1\" />\n"
  "</configuration>\n";

  IIOMng* io_mng = app->ioMng();
  info() << "XML_FILE is " << cval;

  ScopedPtrT<IXmlDocumentHolder> xml_doc(io_mng->parseXmlString(cval,"None"));
  if (!xml_doc.get())
    ARCANE_FATAL("Invalid XML file");

  XmlNode root_element = xml_doc->documentNode().documentElement();
  {
    ConfigurationReader cr(traceMng(),configuration.get());
    cr.addValuesFromXmlNode(root_element,0);
  }
  _checkValid(configuration.get());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ConfigurationUnitTest::
_executeTestJSON()
{
  IApplication* app = subDomain()->application();
  IConfigurationMng* cm = app->configurationMng();

  ScopedPtrT<IConfiguration> configuration(cm->createConfiguration());

  String cval = "<?xml version=\"1.0\"?>\n"
  "<configuration>\n"
  " <add name=\"Section1.Section2.V1\" value=\"250000\" />\n"
  " <add name=\"GlobalStr\" value=\"Toto\" />\n"
  " <add name=\"GlobalReal\" value=\"3.5\" />\n"
  " <add name=\"GlobalInt64\" value=\"1234567890123\" />\n"
  " <add name=\"Section1.Bool1\" value=\"true\" />\n"
  " <add name=\"Section1.Bool2\" value=\"1\" />\n"
  "</configuration>\n";

  String json_cval = "{\n"
  " \"GlobalStr\" : \"Toto\",\n"
  " \"GlobalReal\" : \"3.5\",\n"
  " \"GlobalInt64\" : \"1234567890123\",\n"
  " \"Section1\" : { \n"
  " \"Section2.V1\" : \"250000\",\n"
  " \"Bool1\" : \"true\",\n"
  " \"Bool2\" : \"1\"\n"
  " } \n"
  "}";

  JSONDocument jdoc;
  jdoc.parse(json_cval.bytes());
  {
    ConfigurationReader cr(traceMng(),configuration.get());
    cr.addValuesFromJSON(jdoc.root(),0);
  }
  _checkValid(configuration.get());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANETEST_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
