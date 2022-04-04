﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CaseOptionsTesterModule.cc                                  (C) 2000-2018 */
/*                                                                           */
/* Module de test des options du jeu de données.                             */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/utils/Real3.h"
#include "arcane/utils/Real2.h"
#include "arcane/utils/Real3x3.h"
#include "arcane/utils/Real2x2.h"
#include "arcane/utils/ValueChecker.h"
#include "arcane/utils/PlatformUtils.h"

#include "arcane/EntryPoint.h"
#include "arcane/ISubDomain.h"
#include "arcane/CaseOptionsMain.h"
#include "arcane/VariableTypes.h"
#include "arcane/MeshVariable.h"
#include "arcane/MathUtils.h"
#include "arcane/ObserverPool.h"
#include "arcane/IMesh.h"
#include "arcane/IItemFamily.h"

#include "arcane/ITimeLoopMng.h"
#include "arcane/ITimeLoop.h"
#include "arcane/ITimeLoopService.h"
#include "arcane/ICaseMng.h"
#include "arcane/IPostProcessorWriter.h"
#include "arcane/TimeLoopEntryPointInfo.h"
#include "arcane/StandardCaseFunction.h"
#include "arcane/AbstractCaseDocumentVisitor.h"
#include "arcane/AbstractService.h"
#include "arcane/ServiceFactory.h"
#include "arcane/ICaseDocument.h"

#include "arcane/tests/TypesCaseOptionsTester.h"

class IComplex1SubInterface
{
 public:
  virtual ~IComplex1SubInterface() = default;
};

class IComplex1Interface
{
 public: virtual ~IComplex1Interface() = default;

 public:
  virtual Arcane::ConstArrayView<Arcane::Real> getSimpleReal2Multi() = 0;
  virtual Arcane::Real getSimpleReal2() = 0;
  virtual Arcane::Integer getSimpleInteger2() =0;
  virtual Arcane::Real3 getSimpleReal32() =0;
  virtual ArcaneTest::TestRealInt getExtendedRealInt2() =0;
  virtual ArcaneTest::TypesCaseOptionsTester::eSimpleEnum getSimpleEnum2() =0;
  virtual IComplex1SubInterface* getComplex1Sub() =0;
  virtual IComplex1SubInterface* getComplex1Subref() =0;
};
class IComplex2Interface {public: virtual ~IComplex2Interface(){}};
class IComplex3Interface {public: virtual ~IComplex3Interface(){}};
class IComplex4Interface {public: virtual ~IComplex4Interface(){}};

ARCANETEST_BEGIN_NAMESPACE
class ICaseOptionTestInterface { public: virtual ~ICaseOptionTestInterface(){} };
ARCANETEST_END_NAMESPACE

#include "arcane/tests/IServiceInterface.h"
#include "arcane/tests/CaseOptionsTester_axl.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE
extern "C++" bool
_caseOptionConvert(const CaseOptionBase& co,const String& str,
                   ArcaneTest::TestRealInt& value)
{
  ARCANE_UNUSED(co);
  ARCANE_UNUSED(str);
  ARCANE_UNUSED(value);
  return false;
}
ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANETEST_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Exemple de module.
 */
class CaseOptionsTesterModule
: public ArcaneCaseOptionsTesterObject
, public TypesCaseOptionsTester
{
 public:

  CaseOptionsTesterModule(const Arcane::ModuleBuilder& cb);
  ~CaseOptionsTesterModule();

 public:

  static void staticInitialize(Arcane::ISubDomain* sd);

 public:
	
  virtual Arcane::VersionInfo versionInfo() const { return Arcane::VersionInfo(0,1,0); }

 public:
	
  virtual void build();
  virtual void init();
  virtual void arcaneLoop();
  
 private:
	
  static void _createTimeLoop(Arcane::ISubDomain* sd);
  void _applyVisitor();

  Arcane::ObserverPool m_observers;

  void _onBeforePhase1();
  void _onBeforePhase2();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_DEFINE_STANDARD_MODULE(CaseOptionsTesterModule,CaseOptionsTester);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{
using namespace Arcane;
class StandardFuncTest
: public StandardCaseFunction
, public IBinaryMathFunctor<Real,Real3,Real>
{
 public:
  StandardFuncTest(const CaseFunctionBuildInfo& bi) : StandardCaseFunction(bi){}
 public:
  virtual IBinaryMathFunctor<Real,Real3,Real>* getFunctorRealReal3ToReal()
  {
    return this;
  }
  virtual Real apply(Real r,Real3 r3)
  {
    return r + r3.normL2();
  }
};

class Visitor1
: public AbstractCaseDocumentVisitor
, public TraceAccessor
{
 public:
  Visitor1(ITraceMng* tm) : TraceAccessor(tm){}
 public:
  void beginVisit(const ICaseOptions* opt) override
  {
    info() << "BeginOpt " << opt->rootTagName() << " {";
  }
  void endVisit(const ICaseOptions* opt) override
  {
    info() << "EndOptList " << opt->rootTagName() << " }";
  }
  void applyVisitor(const CaseOptionSimple* opt) override
  {
    info() << "SimpleOpt " << _getName(opt);
  }
  void applyVisitor(const CaseOptionMultiSimple* opt) override
  {
    info() << "MultiSimple " << _getName(opt);
  }
  void applyVisitor(const CaseOptionMultiExtended* opt) override
  {
    info() << "MultiExtended " << _getName(opt);
  }
  void applyVisitor(const CaseOptionExtended* opt) override
  {
    info() << "Extended " << _getName(opt);
  }
  void applyVisitor(const CaseOptionMultiEnum* opt) override
  {
    info() << "MultiEnum " << _getName(opt);
  }
  void applyVisitor(const CaseOptionEnum* opt) override
  {
    info() << "Enum " << _getName(opt);
  }
  void beginVisit(const CaseOptionServiceImpl* opt) override
  {
    info() << "Begin Service " << opt->name();
  }
  void endVisit(const CaseOptionServiceImpl* opt) override
  {
    info() << "End Service " << opt->name();
  }
  void beginVisit(const CaseOptionMultiServiceImpl* opt,Integer index) override
  {
    info() << "Begin MultiService " << opt->serviceName(index) << " index=" << index;
  }
  void endVisit(const CaseOptionMultiServiceImpl* opt,Integer index) override
  {
    info() << "End MultiService " << opt->serviceName(index) << " index=" << index;
  }
  String _getName(const CaseOptionBase* opt)
  {
    return opt->name() + " en:" + opt->trueName() + " fr:" + opt->translatedName("fr");
  }
};
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CaseOptionsTesterModule::
CaseOptionsTesterModule(const Arcane::ModuleBuildInfo& mb)
: ArcaneCaseOptionsTesterObject(mb)
{
  using namespace Arcane;

  addEntryPoint(this,"CaseOptionBuild",
                &CaseOptionsTesterModule::build,IEntryPoint::WBuild);
  addEntryPoint(this,"CaseOptionInit",
                &CaseOptionsTesterModule::init,IEntryPoint::WInit);
  addEntryPoint(this,"CaseOptionLoop",
                &CaseOptionsTesterModule::arcaneLoop);

  ICaseMng* cm = mb.subDomain()->caseMng();
  {
    CaseFunctionBuildInfo bi(mb.subDomain()->traceMng(),"std_func");
    cm->addFunction(new StandardFuncTest(bi));

    ICaseDocument* doc = cm->caseDocument();
    // 'doc' peut être nul lorsqu'on génère le infos sur tous les
    // modules et service via 'dump_internal'
    if (doc){
      info() << "DefaultCategory1=" << doc->defaultCategory();
      String xd = platform::getEnvironmentVariable("ARCANE_DEFAULT_CATEGORY");
      if (!xd.null())
        doc->setDefaultCategory(xd);
      info() << "DefaultCategory2=" << doc->defaultCategory();
    }
  }

  // Appelle les méthodes _onBeforePhase{1|2} lors des phases de lecture du
  // jeu de données.
  m_observers.addObserver(this,&CaseOptionsTesterModule::_onBeforePhase1,
                          cm->observable(eCaseMngEventType::BeginReadOptionsPhase1));
  m_observers.addObserver(this,&CaseOptionsTesterModule::_onBeforePhase2,
                          cm->observable(eCaseMngEventType::BeginReadOptionsPhase2));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CaseOptionsTesterModule::
~CaseOptionsTesterModule()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseOptionsTesterModule::
_createTimeLoop(Arcane::ISubDomain* sd)
{
  using namespace Arcane;

  String time_loop_name("CaseOptionsTester");

  ITimeLoopMng* tlm = sd->timeLoopMng();
  ITimeLoop* time_loop = tlm->createTimeLoop(time_loop_name);

  {
    List<TimeLoopEntryPointInfo> clist;
    clist.add(TimeLoopEntryPointInfo("CaseOptionsTester.CaseOptionBuild"));
    time_loop->setEntryPoints(ITimeLoop::WBuild,clist);
  }

  {
    List<TimeLoopEntryPointInfo> clist;
    clist.add(TimeLoopEntryPointInfo("CaseOptionsTester.CaseOptionInit"));
    time_loop->setEntryPoints(ITimeLoop::WInit,clist);
  }

  {
    List<TimeLoopEntryPointInfo> clist;
    clist.add(TimeLoopEntryPointInfo("CaseOptionsTester.CaseOptionLoop"));
    time_loop->setEntryPoints(ITimeLoop::WComputeLoop,clist);
  }

  {
    StringList clist;
    clist.add(String("CaseOptionsTester"));
    time_loop->setRequiredModulesName(clist);
  }

  tlm->registerTimeLoop(time_loop);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseOptionsTesterModule::
staticInitialize(Arcane::ISubDomain* sd)
{
  _createTimeLoop(sd);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseOptionsTesterModule::
_onBeforePhase1()
{
  info() << "Event: BeforeReadPhase1";
  options()->simpleRealWithDynamicDefault.setDefaultValue(3.0);
  options()->cellGroupWithDynamicDefault.setDefaultValue("ZG");
  options()->testServiceWithDynamicDefault.setDefaultValue("ServiceTestImpl3");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseOptionsTesterModule::
_onBeforePhase2()
{
  using namespace Arcane;

  info() << "Event: BeforeReadPhase2";
  // Lors de la phase2, le maillage a déjà été lu donc on peut
  // changer dynamiquement les valeurs en récupérant des infos du maillage.

  // Pour test, positionne l'option avec le nom du premier groupe de face.
  ItemGroupCollection face_groups = defaultMesh()->faceFamily()->groups();
  info() << "NbFaceGroup=" << face_groups.count();
  ItemGroup last_group = face_groups.front();
  info() << "Set 'faceGroup' default value v=" << last_group.name();
  options()->faceGroupWithDynamicDefault.setDefaultValue(last_group.name());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseOptionsTesterModule::
arcaneLoop()
{
  using namespace Arcane;

  if (m_global_iteration()>options()->maxIteration.value())
    subDomain()->timeLoopMng()->stopComputeLoop(true);

  info() << "SimpleEnumFunction = " << (int)options()->simpleEnumFunction.value()
         << " changed=" << options()->simpleEnumFunction.hasChangedSinceLastIteration();
  info() << "SimpleRealUnit2 = " << options()->simpleRealUnit2.value()
         << " changed=" << options()->simpleRealUnit2.hasChangedSinceLastIteration();
  for( Integer i=0, is=options()->complex2.size(); i<is; ++i ){
    info() << "Complex2/SimpleReal = "
           << options()->complex2[i].simpleRealC2
           << " changed=" << options()->complex2[i].simpleRealC2.hasChangedSinceLastIteration();
  }
  IStandardFunction* sf = options()->simpleWithStandardFunction.standardFunction();
  if (sf){
    Real r = (Real)m_global_iteration();
    Real v = sf->getFunctorRealReal3ToReal()->apply(r,Real3(1.0,r,3.0));
    info() << "V=" << v;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseOptionsTesterModule::
build()
{
  using namespace Arcane;

  info() << "Entering " << A_FUNCNAME;
  // Vérifie que le module implémente bien l'interface ICaseOptionTestInterface
  ICaseOptionTestInterface* o = this;
  info() << "ICaseOptionTestInterface=" << o;

  if (options()->testId()==4){
    options()->simpleRealWithDefault.setDefaultValue(3.0);
    options()->simpleIntegerWithDefault.setDefaultValue(99);
    UniqueArray<Real> x = { 4.5, 7.9, 12.3, 1.6 };
    options()->simpleRealarrayWithDefault.setDefaultValue(x);
    options()->simpleReal3WithDefault.setDefaultValue(Real3(1.2,2.3,4.5));
    options()->simpleEnumWithDefault.setDefaultValue(TypesCaseOptionsTester::SEEnum4);
    info() << "SetDefaultValue for CellGroup";
    options()->cellGroupWithDynamicDefault.setDefaultValue("ZD");
    for( Integer i=0, n=options()->complex2.size(); i<n; ++i ){
      auto& coi = options()->complex2[i].simpleIntegerC2WithDefault;
      coi.setDefaultValue(25+3*i);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseOptionsTesterModule::
init()
{
  using namespace Arcane;

  info() << "Entering " << A_FUNCNAME;

  m_global_deltat = 1.0;

  info() << " MaxIteration= " << options()->maxIteration;
  info() << " SimpleReal=   " << options()->simpleReal;
  if (options()->simpleRealOptional.hasValidValue())
    info() << " SimpleRealOptional=   " << options()->simpleRealOptional;
  info() << " SimpleRealHexa=   " << Convert::toHexaString(options()->simpleReal);
  info() << " SimpleRealUnit=   " << options()->simpleRealUnit
         << " default_unit=" << options()->simpleRealUnit.defaultPhysicalUnit()
         << " unit=" << options()->simpleRealUnit.physicalUnit();
  info() << " SimpleRealUnit2=   " << options()->simpleRealUnit2
         << " default_unit=" << options()->simpleRealUnit2.defaultPhysicalUnit()
         << " unit=" << options()->simpleRealUnit2.physicalUnit();
  info() << " SimpleInteger=" << options()->simpleInteger;
  info() << " SimpleInteger=" << options()->simpleInteger.name();
  info() << " SimpleBool=   " << options()->simpleBool;
  info() << " SimpleString= " << options()->simpleString;
  info() << " InfinityReal    = " << options()->infinityReal();
  info() << " SimpleEnumWithFunction   = " << options()->simpleEnumFunction.name();
  info() << " Complex1/SimpleReal-2    = " << options()->complex1.simpleReal2;
  info() << " Complex1/SimpleReal3-2   = " << options()->complex1.simpleReal32;
  info() << " Complex1/SimpleInteger-2 = " << options()->complex1.simpleInteger2;
  info() << " Complex1/SimpleReal2-Multi = " << options()->complex1.getSimpleReal2Multi();
  info() << " Complex2[2]/Complex3[1]  = " << options()->complex2[1].complex3[0].element().xpathFullName();
  info() << " Complex2[2]/Complex3[2]  = " << options()->complex2[1].complex3[1].element().xpathFullName();
  info() << " Complex4/simple-real     = " << options()->complex4.simpleReal();
  info() << " Complex5/simple-real     = " << options()->complex5.simpleReal();
  
  info() << " Complex6 is_present?     = " << options()->complex6.isPresent();
  if (options()->complex6.isPresent()){
    info() << " Complex6/SimpleReal    = " << options()->complex6.simpleReal;
  }
  if (options()->complex1.simpleEnum2.isPresent()){
    info() << " Complex1/simpleEnum2   = " << options()->complex1.simpleEnum2;
  }
  else
    info() << " Complex1/simpleEnum2   = (absent)";
    
  {
    const RealArray& opt = options()->simpleRealarrayUnit();
    for( Integer i=0, n=opt.size(); i<n; ++i ){
      info() << " SimpleRealArrayUnit i=" << i << " V=" << opt[i];
    }
  }

  {
    Integer n = options()->simpleRealArrayMulti.size();
    for( Integer i=0; i<n; ++i ){
      info() << " SimpleRealArrayMulti i=" << i << " V=" << options()->simpleRealArrayMulti[i];
    }
  }
  ValueChecker vc(A_FUNCINFO);
  {
    UniqueArray<Real> v0 = { 5.2, 2.3 };
    vc.areEqual(options()->complex1.getSimpleReal2Multi(),v0.constView(),"SimpleReal2Multi");
  }
  if (options()->testId()==4){
    vc.areEqual(options()->simpleRealWithDefault(),3.0,"setDefaultValue() for real");
    vc.areEqual(options()->simpleIntegerWithDefault(),99,"setDefaultValue() for integer");
    RealUniqueArray x = { 4.5, 7.9, 12.3, 1.6 };
    vc.areEqual(options()->simpleRealarrayWithDefault().constView(),x.constView(),"setDefaultValue() for real[]");
    vc.areEqual(options()->simpleReal3WithDefault(),Real3(1.2,2.3,4.5),"setDefaultValue() for real3");
    vc.areEqual((int)(options()->simpleEnumWithDefault()),(int)(TypesCaseOptionsTester::SEEnum4),
                "setDefaultValue() for enumeration");
    vc.areEqual(options()->cellGroupWithDynamicDefault().name(),String("ZD"),
                "setDefaultValue() for Arcane::CellGroup");
    for( Integer i=0, n=options()->complex2.size(); i<n; ++i ){
      vc.areEqual(options()->complex2[i].simpleIntegerC2WithDefault(),25+3*i,
                  "setDefaultValue() for sub complex simple-integer-c2-with-default");
    }
    vc.areEqual(options()->testServiceWithDynamicDefault()->implementationName(),String("ServiceTestImpl3"),
                "service with dynamic default");
  }
  if (options()->testId()==5){
    vc.areEqual(options()->simpleReal3(),Real3(25.1,12.3,1.0),"TestId5 - simple-real3 (with default category");
    UniqueArray<Int32> x = { 3, -4, 5, -6, 7 };   
    vc.areEqual(options()->simpleInt32Array().constView(),x.constView(),"TestId5 - simple-int32-array (with default category");
  }
  _applyVisitor();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseOptionsTesterModule::
_applyVisitor()
{
  using namespace Arcane;

  Visitor1 my_visitor(traceMng());
  CaseOptionsCollection opts = subDomain()->caseMng()->blocks();
  for( CaseOptionsCollection::Enumerator i(opts); ++i; ){
    ICaseOptions* o = *i;
    info() << " OptName=" << o->rootTagName();
    o->visit(&my_visitor);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ServiceTestImplInterface1
: public Arcane::AbstractService
, public IServiceInterface1
{
 public:

  ServiceTestImplInterface1(const ServiceBuildInfo& sbi)
  : AbstractService(sbi)
  {
    info() << "Create ServiceTestImplInterface1 name=" << serviceInfo()->localName();
  }

 public:
  Integer value() override { return 0; }
  void* getPointer1() override { return this; }
  Arccore::String implementationName() const override { return serviceInfo()->localName(); }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ServiceTestImplInterface3
: public Arcane::AbstractService
, public IServiceInterface3
{
 public:

  ServiceTestImplInterface3(const ServiceBuildInfo& sbi)
  : AbstractService(sbi)
  {
    info() << "Create ServiceTestImplInterface3 name=" << serviceInfo()->localName();
  }

 public:
  void* getPointer3() override { return this; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ServiceTestImplInterface4
: public Arcane::AbstractService
, public IServiceInterface1
, public IServiceInterface2
, public IServiceInterface3
, public IServiceInterface4
{
 public:

  ServiceTestImplInterface4(const ServiceBuildInfo& sbi)
  : AbstractService(sbi)
  {
    info() << "Create ServiceTestImplInterface3 name=" << serviceInfo()->localName();
  }

 public:
  void* getPointer1() override { return this; }
  void* getPointer2() override { return this; }
  void* getPointer3() override { return this; }
  void* getPointer4() override { return this; }
  Arccore::Integer value() override { return 3; }
  Arccore::String implementationName() const override { return "ServiceTestImplInterface4"; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE(ServiceTestImplInterface1,
                        Arcane::ServiceProperty("ServiceTestImpl1",Arcane::ST_CaseOption),
                        ARCANE_SERVICE_INTERFACE(IServiceInterface1));

ARCANE_REGISTER_SERVICE(ServiceTestImplInterface1,
                        Arcane::ServiceProperty("ServiceTestImpl2",Arcane::ST_CaseOption),
                        ARCANE_SERVICE_INTERFACE(IServiceInterface1));

ARCANE_REGISTER_SERVICE(ServiceTestImplInterface1,
                        Arcane::ServiceProperty("ServiceTestImpl3",Arcane::ST_CaseOption),
                        ARCANE_SERVICE_INTERFACE(IServiceInterface1));

ARCANE_REGISTER_SERVICE(ServiceTestImplInterface1,
                        Arcane::ServiceProperty("ServiceTestImpl4",Arcane::ST_CaseOption),
                        ARCANE_SERVICE_INTERFACE(IServiceInterface1));

ARCANE_REGISTER_SERVICE(ServiceTestImplInterface3,
                        Arcane::ServiceProperty("ServiceTestInterface3Impl1",Arcane::ST_CaseOption),
                        ARCANE_SERVICE_INTERFACE(IServiceInterface3));

ARCANE_REGISTER_SERVICE(ServiceTestImplInterface3,
                        Arcane::ServiceProperty("ServiceTestInterface3Impl2",Arcane::ST_CaseOption),
                        ARCANE_SERVICE_INTERFACE(IServiceInterface3));

ARCANE_REGISTER_SERVICE(ServiceTestImplInterface4,
                        Arcane::ServiceProperty("ServiceTestInterface4Full",Arcane::ST_CaseOption),
                        ARCANE_SERVICE_INTERFACE(IServiceInterface1),
                        ARCANE_SERVICE_INTERFACE(IServiceInterface2),
                        ARCANE_SERVICE_INTERFACE(IServiceInterface3),
                        ARCANE_SERVICE_INTERFACE(IServiceInterface4));

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANETEST_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
