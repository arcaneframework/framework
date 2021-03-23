// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SingletonServiceTestModule.cc                               (C) 2000-2018 */
/*                                                                           */
/* Module de test des services singletons.                                   */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/ISubDomain.h"
#include "arcane/ITimeLoopMng.h"
#include "arcane/BasicService.h"
#include "arcane/FactoryService.h"
#include "arcane/ServiceBuilder.h"
#include "arcane/Configuration.h"

#include "arcane/tests/ArcaneTestGlobal.h"

#include "arcane/tests/SingletonServiceTest_axl.h"
#include "arcane/tests/IServiceInterface.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace ArcaneTest
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arcane;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Module de test de sous-maillage dans Arcane.
 */
class SingletonServiceTestModule
: public ArcaneSingletonServiceTestObject
{
 public:

  SingletonServiceTestModule(const ModuleBuildInfo& cb);
  ~SingletonServiceTestModule() override;

 public:

  VersionInfo versionInfo() const override { return VersionInfo(1,0,0); }

 public:

  void build();
  void init() override;
  void compute() override;

 private:
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class TestSingletonService
: public BasicService
, public IServiceInterface3
, public IServiceInterface4
{
 public:
  TestSingletonService(const ServiceBuildInfo& sbi)
  : BasicService(sbi)
  {
    info() << "INFO: Create test singleton service ptr=" << this;
  }
  ~TestSingletonService() override
  {
    info() << "INFO: Destroy test singleton service ptr=" << this;
  }
 public:
  void* getPointer3() override { return this; }
  void* getPointer4() override { return this; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class TestSingletonService2
: public BasicService
, public IServiceInterface5
{
 public:
  TestSingletonService2(const ServiceBuildInfo& sbi)
  : BasicService(sbi)
  {
    info() << "INFO: Create test singleton service2 ptr=" << this;
  }
  ~TestSingletonService2() override
  {
    info() << "INFO: Destroy test singleton service2 ptr=" << this;
  }
 public:
  void* getPointer5() override { return this; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class TestSingletonService3
: public BasicService
, public IServiceInterface6
{
 public:
  TestSingletonService3(const ServiceBuildInfo& sbi)
  : BasicService(sbi)
  {
    info() << "INFO: Create test singleton service3 ptr=" << this;
  }
  ~TestSingletonService3() override
  {
    info() << "INFO: Destroy test singleton service3 ptr=" << this;
  }
 public:
  void* getPointer6() override { return this; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_DEFINE_STANDARD_MODULE(SingletonServiceTestModule,SingletonServiceTestModule);

ARCANE_REGISTER_SERVICE(TestSingletonService,
                        ServiceProperty("TestSingleton1",ST_SubDomain,SFP_Singleton),
                        ARCANE_SERVICE_INTERFACE(IServiceInterface3),
                        ARCANE_SERVICE_INTERFACE(IServiceInterface4));

ARCANE_REGISTER_SERVICE(TestSingletonService,
                        ServiceProperty("TestSingleton2",ST_SubDomain,SFP_Singleton),
                        ARCANE_SERVICE_INTERFACE(IServiceInterface3));

ARCANE_REGISTER_SERVICE(TestSingletonService2,
                        ServiceProperty("TestSingleton5",ST_SubDomain,SFP_Singleton),
                        ARCANE_SERVICE_INTERFACE(IServiceInterface5));

ARCANE_REGISTER_SERVICE(TestSingletonService3,
                        ServiceProperty("TestSingleton6",ST_SubDomain,SFP_Singleton),
                        ARCANE_SERVICE_INTERFACE(IServiceInterface6));

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/


SingletonServiceTestModule::
SingletonServiceTestModule(const ModuleBuildInfo& mb)
: ArcaneSingletonServiceTestObject(mb)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/


SingletonServiceTestModule::
~SingletonServiceTestModule()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SingletonServiceTestModule::
build()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SingletonServiceTestModule::
init()
{
  ServiceBuilder<IServiceInterface1> sb(subDomain());
  auto msi1 = sb.getSingleton();
  void* ptr1 = msi1->getPointer1();
  info() << "VALUE1=" << msi1->value() << " ptr=1" << ptr1
         << " impl=" << msi1->implementationName();

  ServiceBuilder<IServiceInterface2> sb2(subDomain());
  auto msi2 = sb2.getSingleton();
  void* ptr2 = msi2->getPointer2();
  info() << "VALUE2=" << msi2 << " ptr2=" << ptr2;

  ServiceBuilder<IServiceInterface3> sb3(subDomain());
  auto msi3 = sb3.getSingleton();
  void* ptr3 = msi3->getPointer3();
  info() << "VALUE3=" << msi3 << " ptr3=" << ptr3;

  ServiceBuilder<IServiceInterface4> sb4(subDomain());
  auto msi4 = sb4.getSingleton();
  void* ptr4 = msi4->getPointer4();
  info() << "VALUE4=" << msi4 << " ptr4=" << ptr4;

  if (ptr1!=ptr2)
    ARCANE_FATAL("Different singleton instance ptr1={0} ptr2={1}",ptr1,ptr2);

  if (ptr3!=ptr4)
    ARCANE_FATAL("Different singleton instance ptr3={0} ptr4={1}",ptr3,ptr4);

  IConfiguration* configuration = subDomain()->configuration();
  info() << "SubDomainConfiguration=" << configuration;
  configuration->dump();

  IConfigurationSection* cs = configuration->mainSection();
  Integer v1 = cs->valueAsInteger("TestGlobalConfig1",0);
  if (v1!=267)
    ARCANE_FATAL("Bad config value for TestGlobalConfig1 v={0}",v1);

  Real v2 = cs->valueAsReal("TestGlobalConfig2",0.0);
  if (v2!=4.5)
    ARCANE_FATAL("Bad config value for TestGlobalConfig2 v={0}",v2);

  Real v3 = cs->valueAsReal("TestGlobalConfig3",0.0);
  if (v3!=9.3)
    ARCANE_FATAL("Bad config value for TestGlobalConfig3 v={0}",v3);

  {
    ServiceBuilder<IServiceInterface5> sb(subDomain());
    auto x = sb.getSingleton();
    info() << "ServiceInterface5 = " << x;
  }
  {
    ServiceBuilder<IServiceInterface6> sb(subDomain());
    auto x = sb.getSingleton();
    info() << "ServiceInterface6 = " << x;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SingletonServiceTestModule::
compute()
{
  if (subDomain()->commonVariables().globalIteration()>10)
    subDomain()->timeLoopMng()->stopComputeLoop(true);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace ArcaneTest

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
