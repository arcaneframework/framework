// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* XmlUnitTest.cc                                              (C) 2000-2025 */
/*                                                                           */
/* Test du lecteur/ecrivain Xml.                                             */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/ScopedPtr.h"

#include "arcane/BasicUnitTest.h"
#include "arcane/FactoryService.h"
#include "arcane/Directory.h"
#include "arcane/ISubDomain.h"
#include "arcane/IApplication.h"
#include "arcane/IIOMng.h"
#include "arcane/IXmlDocumentHolder.h"
#include "arcane/IRessourceMng.h"
#include "arcane/Directory.h"
#include "arcane/XmlNode.h"

#include "arcane/tests/ArcaneTestGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace ArcaneTest
{
using namespace Arcane;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Service de test du lecteur/ecrivain Xml.
 */
class XmlUnitTest
: public BasicUnitTest
{
 public:

  explicit XmlUnitTest(const ServiceBuildInfo& sbi);

 public:

  void initializeTest() override;
  void executeTest() override;

 private:

  void _testXml_1();
  void _testXml_2();
  void _testXml_Huge();

 private:

  Directory m_xml_path;

 private:

  void _readDocument(const String& filename);
  void _doDoc(Integer i);
  void _readAndValidateDocument(const String& filename,const String& schema_filename);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_CASE_OPTIONS_NOAXL_FACTORY(XmlUnitTest,IUnitTest,XmlUnitTest);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

XmlUnitTest::
XmlUnitTest(const ServiceBuildInfo& sbi)
: BasicUnitTest(sbi)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void XmlUnitTest::
initializeTest()
{
  String xml_path = platform::getEnvironmentVariable("ARCANE_XML_PATH");
  if (xml_path.null())
    ARCANE_FATAL("Environment variable 'ARCANE_XML_PATH' is not set");
  Directory dir(xml_path);
  m_xml_path = Directory(dir,"xml");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void XmlUnitTest::
executeTest()
{
  _testXml_1();
  _testXml_2();
  _testXml_Huge();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void XmlUnitTest::
_readDocument(const String& filename)
{
  info() << "READ DOCUMENT " << filename;
  IIOMng* io_mng = subDomain()->application()->ioMng();
  ScopedPtrT<IXmlDocumentHolder> doc(io_mng->parseXmlFile(filename));
  if (!doc.get())
    ARCANE_FATAL("Can not read file '{0}'",filename);
  XmlNode root_element = doc->documentNode().documentElement();
  info() << "ROOT name=" << root_element.name() << " value=" << root_element.value();
  const dom::NamedNodeMap& attrs = root_element.domNode().attributes();
  dom::ULong nb_attr = attrs.length();
  info() << "NB_ATTR=" << attrs.length();
  for( dom::ULong i=0; i<nb_attr; ++i ){
    XmlNode a(nullptr,attrs.item(i));
    info() << "ATTR name=" << a.name() << " value=" << a.value();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void XmlUnitTest::
_readAndValidateDocument(const String& filename,const String& schema_filename)
{
  info() << "READ AND VALIDATE DOCUMENT " << filename << " schema=" << schema_filename;
  IIOMng* io_mng = subDomain()->application()->ioMng();
  ScopedPtrT<IXmlDocumentHolder> doc(io_mng->parseXmlFile(filename,schema_filename));
  if (!doc.get())
    ARCANE_FATAL("Can not read file '{0}'",filename);
  XmlNode root_element = doc->documentNode().documentElement();
  info() << "VALIDATE ROOT name=" << root_element.name() << " value=" << root_element.value();
  const dom::NamedNodeMap& attrs = root_element.domNode().attributes();
  dom::ULong nb_attr = attrs.length();
  info() << "NB_ATTR=" << attrs.length();
  for( dom::ULong i=0; i<nb_attr; ++i ){
    XmlNode a(nullptr,attrs.item(i));
    info() << "ATTR name=" << a.name() << " value=" << a.value();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void XmlUnitTest::
_doDoc(Integer i)
{
  info() << "CREATE DOC I=" << i;
  IApplication* app = subDomain()->application();
  ScopedPtrT<IXmlDocumentHolder> doc_holder(app->ressourceMng()->createXmlDocument());
  XmlNode doc = doc_holder->documentNode();
  XmlElement root(doc,"root1");

  XmlElement config(root,"config");
  XmlElement test(config,"test");

  test.createAndAppendElement("t1","Test1");
  config.createAndAppendElement("host","Test2");
  test.createAndAppendElement("iteration1",String::fromNumber(i));
  test.createAndAppendElement("iteration2",String::fromNumber(i+1));
  test.createAndAppendElement("iteration3",String::fromNumber(i+2));
  test.setAttrValue("attr1","value1");
  XmlNode attr = test.attr("attr1");
  if (attr.null())
    ARCANE_FATAL("Unexpected null attribute");
  String attr_value = attr.value();
  if (attr_value!="value1")
    ARCANE_FATAL("Bad value for attribute");
  dom::Element dom_element{test.domNode()};
  String attr_value2 = dom_element.getAttribute("attr1");
  if (attr_value2!=attr_value)
    ARCANE_FATAL("Bad value for attribute (2)");
    
  test.clear();
  test.createAndAppendElement("iteration4",String::fromNumber(i+4));
  test.createAndAppendElement("iteration5","TestIteration5");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void XmlUnitTest::
_testXml_1()
{
  info() << "TEST XML1";
  Directory dir(m_xml_path);
  UniqueArray<String> file_names =
  {
    "isolat1", "isolat2",
    "isolat3", "japancrlf.xml", "att10", "att11",

    "XInclude/docs/fallback2.xml",
    "XInclude/docs/fallback.xml",
    "XInclude/docs/include.xml",
    "XInclude/docs/nodes2.xml",
    "XInclude/docs/nodes.xml",
    "XInclude/docs/recursive.xml",
    "XInclude/docs/tstencoding.xml",
    "XInclude/docs/txtinclude.xml",

    "schemas/import2_0.xml"
  };
  for( const String& s : file_names ){
    info() << "FILE1=" << s;
    String fname(dir.file(s));
    info() << "FILE2=" << fname;
    _readDocument(fname);
  }
  UniqueArray<String> validation_file_names =
  {
    "schemas/import2_0.xml",
    "schemas/import2_0.xsd"
  };
  for( Integer i=0, n=validation_file_names.size(); i<n; i+=2 ){
    String f = validation_file_names[i];
    String v = validation_file_names[i+1];
    String xml_filename(dir.file(f));
    String schema_filename(dir.file(v));
    info() << "FILE xml=" << xml_filename << " schema=" << schema_filename;
    _readAndValidateDocument(xml_filename,schema_filename);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void XmlUnitTest::
_testXml_2()
{
  info() << "TEST XML2";
  for( Integer i=0; i<20; ++i ){
    _doDoc(i);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Test la lecture d'un élément contenant un élément de 50Mo.
 */
void XmlUnitTest::
_testXml_Huge()
{
  info() << "TEST XML_HUGE";
  Directory base_path(subDomain()->exportDirectory());
  String xml_filename = base_path.file("test_huge.xml");
  ofstream ofile(xml_filename.localstr());
  ofile << "<?xml version='1.0' ?>\n";
  ofile << "<root>\n";
  String base_value = "123456789abcdefghijklmnopqrst123456789abcdefghijklmnopqrst\n";
  for (Int32 i = 0; i < 500000; ++i)
    ofile << base_value;
  ofile << "</root>\n";
  ofile.close();

  // Relit le fichier.
  {
    ScopedPtrT<IXmlDocumentHolder> doc(IXmlDocumentHolder::loadFromFile(xml_filename, traceMng()));
  }

  Platform::removeFile(xml_filename);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
