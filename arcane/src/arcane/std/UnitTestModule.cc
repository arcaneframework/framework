// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* UnitTestModule.cc                                           (C) 2000-2024 */
/*                                                                           */
/* Module pour les tests unitaires.                                          */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ScopedPtr.h"

#include "arcane/core/IUnitTest.h"
#include "arcane/core/ISubDomain.h"
#include "arcane/core/IApplication.h"
#include "arcane/core/ITimeLoop.h"
#include "arcane/core/ITimeLoopMng.h"
#include "arcane/core/IXmlDocumentHolder.h"
#include "arcane/core/IIOMng.h"
#include "arcane/core/ArcaneException.h"
#include "arcane/core/TimeLoopEntryPointInfo.h"
#include "arcane/core/Directory.h"
#include "arcane/core/XmlNode.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/DomUtils.h"

#include "arcane/std/UnitTest_axl.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Module maître
 */
class UnitTestModule
: public ArcaneUnitTestObject
{
 public:

  explicit UnitTestModule(const ModuleBuildInfo& cb);

 public:

  static void staticInitialize(ISubDomain* sd);
  VersionInfo versionInfo() const override { return VersionInfo(2, 0, 0); }

 public:

  void unitTestBuild() override;
  void unitTestInit() override;
  void unitTestDoTest() override;
  void unitTestExit() override;

 private:

  ScopedPtrT<IXmlDocumentHolder> m_tests_doc; //!< Traces des tests unitaires
  bool m_success = true; //!< Vrai tant qu'un test unitaire n'a pas retourné d'erreur.

 private:

  void _checkCreateXmlTestDocument();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_MODULE_UNITTEST(UnitTestModule);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

UnitTestModule::
UnitTestModule(const ModuleBuildInfo& mb)
: ArcaneUnitTestObject(mb)
, m_tests_doc(domutils::createXmlDocument())
{
  XmlNode doc = m_tests_doc->documentNode();
  XmlElement root(doc, "unit-tests-results");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void UnitTestModule::
staticInitialize(ISubDomain* sd)
{
  String time_loop_name("UnitTest");
  ITimeLoopMng* tlm = sd->timeLoopMng();
  ITimeLoop* time_loop = tlm->createTimeLoop(time_loop_name);

  {
    List<TimeLoopEntryPointInfo> clist;
    clist.add(TimeLoopEntryPointInfo("UnitTest.UnitTestBuild"));
    time_loop->setEntryPoints(ITimeLoop::WBuild, clist);
  }

  {
    List<TimeLoopEntryPointInfo> clist;
    clist.add(TimeLoopEntryPointInfo("UnitTest.UnitTestInit"));
    time_loop->setEntryPoints(ITimeLoop::WInit, clist);
  }

  {
    List<TimeLoopEntryPointInfo> clist;
    clist.add(TimeLoopEntryPointInfo("UnitTest.UnitTestDoTest"));
    time_loop->setEntryPoints(ITimeLoop::WComputeLoop, clist);
  }

  {
    List<TimeLoopEntryPointInfo> clist;
    clist.add(TimeLoopEntryPointInfo("UnitTest.UnitTestExit"));
    time_loop->setEntryPoints(ITimeLoop::WExit, clist);
  }

  {
    StringList clist;
    clist.add("UnitTest");
    clist.add("ArcanePostProcessing");
    time_loop->setRequiredModulesName(clist);
  }

  tlm->registerTimeLoop(time_loop);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void UnitTestModule::
unitTestBuild()
{
  for (IUnitTest* service : options()->test)
    service->buildInitializeTest();

  for (IXmlUnitTest* service : options()->xmlTest)
    service->buildInitializeTest();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void UnitTestModule::
unitTestInit()
{
  // Initialise au cas où aucun test ne le fait
  m_global_deltat = 1.0;

  for (IUnitTest* service : options()->test)
    service->initializeTest();

  for (IXmlUnitTest* service : options()->xmlTest)
    service->initializeTest();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void UnitTestModule::
unitTestDoTest()
{
  subDomain()->timeLoopMng()->stopComputeLoop(false);

  for (IUnitTest* service : options()->test)
    service->executeTest();

  if (options()->xmlTest.size() > 0) {
    XmlNode xtests = m_tests_doc->documentNode().documentElement();
    for (IXmlUnitTest* service : options()->xmlTest) {
      XmlNode xservice = xtests.createAndAppendElement("service");
      if (!service->executeTest(xservice))
        m_success = false;
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void UnitTestModule::
unitTestExit()
{
  for (IUnitTest* service : options()->test)
    service->finalizeTest();

  for (IXmlUnitTest* service : options()->xmlTest)
    service->finalizeTest();

  if (options()->xmlTest.size() > 0) {
    // ecriture du rapport XML.
    // En parallèle, seul le processeur maitre écrit le fichier
    IParallelMng* pm = subDomain()->parallelMng();
    if (pm->isMasterIO()) {
      Directory listing_dir(subDomain()->listingDirectory());
      String filename(listing_dir.file("unittests.xml"));
      info() << "Output of the report of the unit test in '" << filename << "'";
      subDomain()->ioMng()->writeXmlFile(m_tests_doc.get(), filename);
    }

    // sortie en exception si demandé
    if (!m_success)
      ARCANE_FATAL("Some errors have occured in the unit tests.");
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
