// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* UtilsUnitTest.cc                                            (C) 2000-2025 */
/*                                                                           */
/* Test des fonctions utilitaires de Arcane.                                 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/UserDataList.h"
#include "arcane/utils/IUserData.h"
#include "arcane/utils/ArgumentException.h"
#include "arcane/utils/AutoDestroyUserData.h"
#include "arcane/utils/HPReal.h"
#include "arcane/utils/IMemoryInfo.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/TraceClassConfig.h"
#include "arcane/utils/FunctorUtils.h"
#include "arcane/utils/ArithmeticException.h"
#include "arcane/utils/ValueConvert.h"
#include "arcane/utils/ITraceMngPolicy.h"
#include "arcane/utils/StringList.h"
#include "arcane/utils/IHashAlgorithm.h"
#include "arcane/utils/ValueChecker.h"

#include "arcane/core/BasicUnitTest.h"
#include "arcane/core/FactoryService.h"
#include "arcane/core/ServiceBuilder.h"

#include <fenv.h>

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
 * \brief Service de test des ItemVector
 */
class UtilsUnitTest
: public BasicUnitTest
{
 public:
  class MyUserData : public IUserData
  {
   public:
    MyUserData() : nb_attach(0), nb_detach(0){}
   public:
    virtual void notifyAttach(){ ++nb_attach; }
    virtual void notifyDetach(){ ++nb_detach; }
   public:
    Int32 nb_attach;
    Int32 nb_detach;
  };
  class MyAutoDestroyTestClass
  {
   public:
    MyAutoDestroyTestClass(Int32* ptr) : m_ptr(ptr){}
    ~MyAutoDestroyTestClass()
    {
      *m_ptr = 1;
    }
   private:
    Int32* m_ptr;
  };
  class MemoryInfoPrinter
  : public IFunctorWithArgumentT<const MemoryInfoChunk&>
  {
   public:
    MemoryInfoPrinter(ITraceMng* tm) : m_trace_mng(tm){}
   public:
    virtual void executeFunctor(const MemoryInfoChunk& chunk);
   private:
    ITraceMng* m_trace_mng;
  };
 public:

  UtilsUnitTest(const ServiceBuildInfo& cb);
  ~UtilsUnitTest();

 public:

  virtual void initializeTest() {}
  virtual void executeTest();

 private:

  void _testStringFormatter();
  void _testString();
  void _testReal3x3();
  void _testUserData();
  void _testHPReal();
  void _testTruncateReal();
  void _printMemoryInfos();
  void _testNullPointer();
  void _testNumeric();
  void _testStackTrace();
  void _testSetClassConfig();
  void _testFloatingException();
  void _testConvertFromString();
  void _testConvertFromStringToInt32Array();
  void _testCommandLine();
  void _testHashAlgorithm();
  void _testDataTypeNames();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_CASE_OPTIONS_NOAXL_FACTORY(UtilsUnitTest,IUnitTest,UtilsUnitTest);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

UtilsUnitTest::
UtilsUnitTest(const ServiceBuildInfo& mb)
: BasicUnitTest(mb)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

UtilsUnitTest::
~UtilsUnitTest()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void UtilsUnitTest::
executeTest()
{
  _testNullPointer();
  _testNumeric();
  _testString();
  _testStringFormatter();
  _testSetClassConfig();
  _testReal3x3();
  _testUserData();
  _testHPReal();
  _testTruncateReal();

  info() << Trace::Color::darkRed()     << "[TEST_COLOR] DARK_RED";
  info() << Trace::Color::darkGreen()   << "[TEST_COLOR] DARK_GREEN";
  info() << Trace::Color::darkYellow()  << "[TEST_COLOR] DARK_YELLOW";
  info() << Trace::Color::darkBlue()    << "[TEST_COLOR] DARK_BLUE";
  info() << Trace::Color::darkMagenta() << "[TEST_COLOR] DARK_MAGENTA";
  info() << Trace::Color::darkCyan()    << "[TEST_COLOR] DARK_CYAN";
  info() << Trace::Color::darkGrey()    << "[TEST_COLOR] DARK_GREY";

  info() << Trace::Color::red()     << "[TEST_COLOR] RED";
  info() << Trace::Color::green()   << "[TEST_COLOR] GREEN";
  info() << Trace::Color::yellow()  << "[TEST_COLOR] YELLOW";
  info() << Trace::Color::blue()    << "[TEST_COLOR] BLUE";
  info() << Trace::Color::magenta() << "[TEST_COLOR] MAGENTA";
  info() << Trace::Color::cyan()    << "[TEST_COLOR] CYAN";
  info() << Trace::Color::grey()    << "[TEST_COLOR] GREY";

  _testConvertFromString();
  _testConvertFromStringToInt32Array();
  _testStackTrace();
  _printMemoryInfos();
  // Teste les FPE en dernier car cela ne fonctionne pas avec valgrind
  // et fait planter le test. En mettant cela en dernier, on peut quand
  // même utiliser valgrind sur les autres tests.
  _testFloatingException();

  _testCommandLine();
  _testHashAlgorithm();
  _testDataTypeNames();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void UtilsUnitTest::
_testNullPointer()
{
  // Vérfie que ARCANE_CHECK_POINTER lance bien une exception de type
  // FatalErrorException en cas de pointeur nul.
  bool is_ok = false;
  try{
    Int32* x = nullptr;
    ARCANE_CHECK_POINTER(x);
  }
  catch(const FatalErrorException& ex)
  {
    is_ok = true;
  }
  if (!is_ok)
    ARCANE_FATAL("Exception not caught");

  // Vérifie la validité de la macro avec une expression et un message.
  IService* x2 = this;
  const BasicUnitTest* bt = ARCANE_CHECK_POINTER(dynamic_cast<const BasicUnitTest*>(x2));
  const BasicUnitTest* bt2 = ARCANE_CHECK_POINTER2(dynamic_cast<const BasicUnitTest*>(x2),"Error");
  info() << "BT=" << bt << "BT2=" << bt2;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void UtilsUnitTest::
_testNumeric()
{
  info() << "** TEST NUMERIC";
  Real small_double = 1e-15;
  if (!math::isNearlyZero(small_double))
    ARCANE_FATAL("Bad nearly zero");
  if (math::isNearlyZeroWithEpsilon(small_double*5000.0,1.0e-12))
    ARCANE_FATAL("Bad nearly zero with epsilon (1)");
  if (!math::isNearlyZeroWithEpsilon(small_double*5000.0,1.0e-11))
    ARCANE_FATAL("Bad nearly zero with epsilon (2)");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void UtilsUnitTest::
_testStringFormatter()
{
  info() << "** TEST STRING FORMATTER";
  int b = 3;
  float c = 5.1f;
  //double d = 7.4;
  String e = "Titi";
  Real3 r3(1.2,1.6,1.7);
  const char* f = A_FUNCINFO . name();
  //info() << (StringFormatter("Try {1} tovers {0}") << b << c).m_format;
  info() << String::format("Function {2} --> Try {1} to {0} name = {3}",
                            b,c,e,f);
  info() << String::format("Function {2} --> Try {1} to {0} name = {3} R={4}",
                           b,c,e,f,r3);
  info() << String::format("Function {2} --> Try {1} to {0} name = {3} R={4} r2={5}",
                           b,c,e,f,r3,r3);
  info() << String::format("Function {2} --> Try {1} to {0} name = {3} R={4} r2={5} r3={6}",
                           b,c,e,f,r3,r3,b);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void UtilsUnitTest::
_testString()
{
  info() << "** TEST STRING";
  String e = "Titi";
  String f = "Toto23";
  if (!f.endsWith("23"))
    ARCANE_FATAL("Bad compare 1");
  if (!f.startsWith("Toto"))
    ARCANE_FATAL("Bad compare 2");
  if (f.startsWith("Toto1"))
    ARCANE_FATAL("Bad compare 3");
  if (f.endsWith("Toto1"))
    ARCANE_FATAL("Bad compare 4");
  if (f.startsWith("Toto234"))
    ARCANE_FATAL("Bad compare 5");
  if (f.endsWith("Toto234"))
    ARCANE_FATAL("Bad compare 6");
  String s2 = f.substring(3);
  if (s2!="o23")
    ARCANE_FATAL("Bad compare 7");
  s2 = f.substring(3,2);
  info() << "S2_8=" << s2;
  if (s2!="o2")
    ARCANE_FATAL("Bad compare 8");
  s2 = f.substring(1,2);
  info() << "S2_9=" << s2;
  if (s2!="ot")
    ARCANE_FATAL("Bad compare 9");

  s2 = f.substring(7,2);
  info() << "S2_10=" << s2;
  if (s2!="")
    ARCANE_FATAL("Bad compare 10");

  s2 = f.substring(2,1);
  info() << "S2_11=" << s2;
  if (s2!="t")
    ARCANE_FATAL("Bad compare 11");

  s2 = f.substring(5,1);
  info() << "S2_12=" << s2;
  if (s2!="3")
    ARCANE_FATAL("Bad compare 12");

  s2 = f.substring(0);
  info() << "S2_13=" << s2;
  if (s2!=f)
    ARCANE_FATAL("Bad compare 13");

  String g = "   \tceci   \tcela ";
  info() << " G=  '" << g << "'";
  String g2 = String::collapseWhiteSpace(g);
  info() << " G2= '" << g2 << "'";
  String g3 = String::replaceWhiteSpace(g);
  info() << " G3= '" << g3 << "'";
  if (g3!="    ceci    cela ")
    ARCANE_FATAL("Bad G3");
  if (g2!="ceci cela")
    ARCANE_FATAL("Bad G2");

  String gnull;
  String gnull2 = String::collapseWhiteSpace(gnull);
  info() << "GNULL2='" << gnull2 << "'";
  if (gnull2!=String())
    ARCANE_FATAL("Bad gnull");

  String gempty("");
  String gempty2 = String::collapseWhiteSpace(gempty);
  info() << "GEMPTY2='" << gempty2 << "'";
  if (gempty2!="")
    ARCANE_FATAL("Bad gempty");

  {
    String k0 = ":Toto::Titi:::Tata::::Tutu:Tete:";
    //String k0 = ":Toto:Titi";
    //String k0 = ":Toto::Titi";
    info() << "ORIGINAL STRING TO STRING = '" << k0 << "'";
    UniqueArray<String> k0_list;
    k0.split(k0_list,':');
    for( Integer i=0, n=k0_list.size(); i<n; ++i ){
      info() << "K i=" << i << " v='" << k0_list[i] << "' is_null?=" << k0_list[i].null();
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void UtilsUnitTest::
_testSetClassConfig()
{
  String x = "<?xml version=\"1.0\"?>\n"
  "<arcane-config>\n"
  "<traces>\n"
  "<trace-class name='UnitTest' info='false' debug='high' />\n"
  "<trace-class name='MyTest' info='false' debug='med' />\n"
  "<trace-class name='MyTest2' info='true' debug='med' print-elapsed-time='true' print-class-name='false'/>\n"
  "<trace-class name='Mpi' info='true' debug='med' />\n"
  "</traces>\n"
  "</arcane-config>\n";
  info() << "SET_CLASS_CONFIG file=" << x;
  auto func = [&](std::pair<String,TraceClassConfig> tcc)
    {
      info() << "CLASS CONFIG=" << tcc.first << " v=" << tcc.second.verboseLevel();
    };
  auto func2 = functor::make(func);
  traceMng()->visitClassConfigs(&func2);
  subDomain()->application()->getTraceMngPolicy()->setClassConfigFromXmlBuffer(traceMng(),x.utf8());
  {
    Trace::Setter mclass(traceMng(),"MyTest");
    // Ne devrait pas s'afficher
    info() << "MY_TEST_PRINT";
  }
  {
    Trace::Setter mclass(traceMng(),"MyTest2");
    info() << "MY_TEST2_PRINT with time and no class name";
  }
  TraceClassConfig tcc = traceMng()->classConfig("MyTest");
  tcc.setFlags(Trace::PF_ElapsedTime|Trace::PF_NoClassName);
  tcc.setActivated(true);
  traceMng()->setClassConfig("MyTest",tcc);
  {
    Trace::Setter mclass(traceMng(),"MyTest");
    info() << "TEST_DIRECT_WITH_TIME and no class name";
  }
  tcc.setFlags(Trace::PF_NoClassName);
  traceMng()->setClassConfig("MyTest",tcc);
  {
    Trace::Setter mclass(traceMng(),"MyTest");
    info() << "TEST_DIRECT_WITH no class name";
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void UtilsUnitTest::
_testReal3x3()
{
  Real values[9];
  for( int i=0; i<9; ++i )
    values[i] = (Real)i;
  // Vérifie que les accesseurs operator[] de Real3x3 sont corrects.
  Real3x3 r = Real3x3::fromLines(values[0],values[1],values[2],
                                 values[3],values[4],values[5],
                                 values[6],values[7],values[8]);
  for( int i=0; i<3; ++i )
    for( int j=0; j<3; ++j )
      if (r[i][j]!=values[(i*3)+j])
        ARCANE_FATAL("Bad value");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void UtilsUnitTest::
_testUserData()
{
  info() << "TEST_USER_DATA";
  info() << "TEST1";
  {
    UserDataList udl;
    MyUserData ud;
    udl.setData("Toto",&ud);
    udl.removeData("Toto");
    info() << "NB_ATTACH=" << ud.nb_attach << " NB_DETACH=" << ud.nb_detach;
    // Vérifie que notifyAttach() et notifyDetach() ont bien été appelé 1 fois.
    if (ud.nb_attach!=1 || ud.nb_detach!=1)
      ARCANE_FATAL("(1) Bad number of call for attach or detach");
  }

  info() << "TEST2";
  {
    MyUserData ud;
    {
      UserDataList udl;
      udl.setData("Toto",&ud);
    }
    // Vérifie que notifyAttach() et notifyDetach() ont bien été appelé 1 fois.
    if (ud.nb_attach!=1 || ud.nb_detach!=1)
      ARCANE_FATAL("(2) Bad number of call for attach or detach");
  }

  info() << "TEST3";
  {
    // Attention a bien declarer \a ud avec \a udl car ce dernier
    // va l'appeler dans son destructeur.
    MyUserData ud;
    UserDataList udl;
    udl.setData("Toto",&ud);
    IUserData* ud2 = udl.data("Toto");
    IUserData* orig_ud = &ud;
    if (ud2!=orig_ud)
      ARCANE_FATAL("Can not retrieve UserData");
  }

  info() << "TEST4";
  // Vérifie qu'on a bien une exception si on ne trouve pas un UserData.
  {
    UserDataList udl;
    bool is_catched = false;
    try{
      IUserData* ud = udl.data("Toto");
      info() << "UD="<< ud;
    }
    catch(ArgumentException& ae){
      is_catched = true;
    }
    if (!is_catched)
      ARCANE_FATAL("Exception not catched");
  }

  info() << "TEST5";
  // Vérifie pas d'exception si demandé
  {
    UserDataList udl;
    IUserData* ud = udl.data("Toto",true);
    if (ud)
      ARCANE_FATAL("UserData is not null");
  }

  info() << "TEST6";
  // Vérifie qu'on a bien une exception si on ne trouve pas un UserData
  // lors d'un remove
  {
    UserDataList udl;
    bool is_catched = false;
    try{
      udl.removeData("Toto");
    }
    catch(ArgumentException& ae){
      is_catched = true;
    }
    if (!is_catched)
      ARCANE_FATAL("(2) Exception not catched");
  }

  info() << "TEST7";
  // Vérifie pas d'exception si demandé
  {
    UserDataList udl;
    udl.removeData("Toto",true);
  }

  info() << "TEST8";
  // Vérifie qu'on ne peut pas ajouter 2 fois pour la même clé.
  {
    MyUserData ud1;
    MyUserData ud2;
    UserDataList udl;
    bool is_catched = false;
    try{
      udl.setData("Toto",&ud1);
      udl.setData("Toto",&ud2);
    }
    catch(ArgumentException& ae){
      is_catched = true;
    }
    if (!is_catched)
      ARCANE_FATAL("(3) Exception not catched");
  }

  info() << "TEST9";
  {
    // Doit être positionné à 1 dans le destructeur de MyAutoDestroyTestClass.
    Int32 value = 0;
    {
      UserDataList udl;
      MyAutoDestroyTestClass* mc = new MyAutoDestroyTestClass(&value);
      udl.setData("Toto",new AutoDestroyUserData<MyAutoDestroyTestClass>(mc));
    }
    if (value!=1)
      ARCANE_FATAL("AutoDestroy failed");
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline Real _getReal(int v)
{
  Real x = ((Real)(v+1)) / 5.0;
  return log10(x);
}

void UtilsUnitTest::
_testHPReal()
{
  info() << "TESTING HPREAL";

  {
    HPReal x1(1.0);
    HPReal x2(2.0);
    HPReal x3 = x1 + x2;
    if (x3.toReal()!=3.0)
      ARCANE_FATAL("Bad value x3='{0} (expected 3.0)",x3);
  }

  /*
   * Pour le test suivant, on fait la somme d'un tableau
   * de réels dans l'ordre croissant des indices puis
   * dans l'ordre décroissant et on vérifie que
   * dans les deux cas on a les mêmes valeurs.
   */

  HPReal r(0.0);
  int nb_value = 10000;

  UniqueArray<Real> values(nb_value);
  for( int i=0; i<nb_value; ++i ){
    values[i] = _getReal(i);
  }
  for( int i=0; i<nb_value; ++i ){
    r = HPReal::accumulate(values[i],r);
  }
  HPReal r2(0.0);
  for( int i=(nb_value-1); i>=0; --i ){
    r2 += values[i];
  }

  Real direct = 0.0;
  for( int i=0; i<nb_value; ++i ){
    direct += values[i];
  }
  Real direct2 = 0.0;
  for( int i=(nb_value-1); i>=0; --i ){
    direct2 += values[i];
  }

  info() << "Direct R  = " << direct << " R2=" << direct2 << " diff=" << (direct2-direct);
  info() << "Value R  = " << r.toReal() << " err=" << r.correction();
  info() << "Value R2 = " << r2.toReal() << " err=" << r2.correction();
  Real diff = r2.toReal() - r.toReal();
  info() << "Value diff = " << diff;
  if (diff!=0.0)
    ARCANE_FATAL("Invalid value v='{0}' (should be 0.0)",diff);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
// Pour TEST:
// MACRO qui permet de n'afficher les messages que si demandé et qui n'évalue pas
// les arguments si cela n'est pas nécessaire.
#if 0
#define CINFO(category, ...) \
    for (bool qt_category_enabled = category; qt_category_enabled; qt_category_enabled = false) \
      info(__VA_ARGS__)
#else
#define CINFO(category, ...) \
  if (category) \
      info(__VA_ARGS__)
#endif

void UtilsUnitTest::
_testTruncateReal()
{
  info() << "Testing real truncation";
  double z = 0.1234567890123456789;
  for( Integer x=1; x<10; ++x ){
    double z2 =  z * (double)x * math::pow(3.0,(double)x);
    for( Integer i=1; i<52; i+=2 ){
      double tz = math::truncateDouble(z2,i);
      info() << " z= " << z2 << " tz(" << i << ")=" << tz << " diff=" << (z2-tz) / (z2);
    }

    for( Integer i=1; i<15; ++i ){
      double tz = math::truncateDouble(z2,i);
      double diff = math::abs((z2-tz) / (z2));
      double diff_log = -math::log10(diff);
      int diff_digit = (int)trunc(diff_log);
      info() << " digit z= " << z2 << " tz(" << i << ")=" << tz
             << " diff=" << diff << " log=" << diff_log << " diff_digit=" << diff_digit;
      // Vérifie que la troncature est au moins du nombre de chiffres
      // souhaité.
      // On ne teste que pour plus de 8 chiffres significatifs car en dessous
      // ce n'est pas précis.
      if (i>=8 && diff_digit>i)
        ARCANE_FATAL("Bad truncate");
    }

  }

  {
    Integer nb = 25;
    UniqueArray<double> ref_values(nb);
    for( Integer k=0; k<nb; ++k ){
      ref_values[k] = z * math::pow(5.0,(double)(k+1));
    }
    for( Integer i=1; i<15; ++i ){
      UniqueArray<double> values(ref_values);
      math::truncateDouble(values,i);
      for( Integer k=0; k<nb; ++k ){
        double rv = math::truncateDouble(ref_values[k],i);
        if (values[k]!=rv)
          ARCANE_FATAL("Bad array truncate i={0} k={1} ref={2} v={3}",
                       i,k,rv,values[k]);
      }
    }
  }

  CINFO(true,4) << "TRUE (if output_level==4) Hello World 4 !!!!";
  CINFO(true,3) << "TRUE Hello World 3 !!!!";
  CINFO(false) << "FALSE Hello World !!!!";

#if __cpp_generic_lambdas
  auto xfunc = [](auto k){ return k * 2; };
  CINFO(true) << "K(3)=" << xfunc(3);
  CINFO(true) << "K(4.5)=" << xfunc(4.5);
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void UtilsUnitTest::
_testFloatingException()
{
  bool has_support = platform::hasFloatingExceptionSupport();
  info() << "PlatformFPE Support?=" << has_support;
  if (!has_support){
    info() << "No FPE test because platform do not support setting FPE";
    return;
  }
  bool is_enabled = platform::isFloatingExceptionEnabled();
  info() << "IS_FloatingPointException enabled?=" << is_enabled;
  if (is_enabled){
    platform::enableFloatingException(false);
    bool is_enabled2 = platform::isFloatingExceptionEnabled();
    info() << "IS_FloatingPointException2 enabled?=" << is_enabled2;
    // Ne doit pas provoquer d'exception flottante
    platform::raiseFloatingException();
  }
  {
    platform::enableFloatingException(true);
    bool is_enabled2 = platform::isFloatingExceptionEnabled();
    info() << "IS_FloatingPointException3 enabled?=" << is_enabled2;
    if (!is_enabled2)
      ARCANE_FATAL("Can not enable FPE");
    // Test si on récupère bien une ArithmeticException.
    bool is_ok = false;
    bool is_ok2 = false;
    try{
      platform::raiseFloatingException();
    }
    catch(const ArithmeticException& ex){
      info() << "'ArithmeticException' catched\n";
      is_ok = true;
      // Regarde si le FPE peut être relancé à l'intérieur.
      try{
        platform::raiseFloatingException();
      }
      catch(const ArithmeticException& ex){
        info() << "'ArithmeticException' catched (nested)\n";
        is_ok2 = true;
      }
    }
    catch(...){
      info() << "Unknown exception catched\n";
    }
    if (!is_ok)
      ARCANE_FATAL("No 'ArithmeticException' catched");
    if (!is_ok2)
      ARCANE_FATAL("No nested 'ArithmeticException' catched");
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void UtilsUnitTest::
_testStackTrace()
{
  String s = platform::getStackTrace();
  info() << "CurrenStackTrace=" << s;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void UtilsUnitTest::
_printMemoryInfos()
{
  info() << "Printing Memory infos";
  IMemoryInfo* mem_info = arcaneGlobalMemoryInfo();
  if (mem_info){
    MemoryInfoPrinter mip(traceMng());
    mem_info->visitAllocatedBlocks(&mip);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void UtilsUnitTest::
_testConvertFromString()
{
  // TODO: tester les autres conversions
  info() << "TEST CONVERT_FROM_STRING";
  String s = "25e3";
  Int32 x = 0;
  bool is_bad = builtInGetValue(x,s);
  info() << "S=" << s << " X=" << x << " is_bad?=" << is_bad;
  if (!is_bad)
    ARCANE_FATAL("Should be bad");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void UtilsUnitTest::
_testConvertFromStringToInt32Array()
{
  // TODO: tester les autres conversions
  info() << "TEST CONVERT_FROM_STRING_TO_INT32ARRAY";
  String s = "3 7 9 12";
  UniqueArray<Int32> x;
  bool is_bad = builtInGetValue(x,s);
  info() << "S='" << s << "' X=" << x << " is_bad?=" << is_bad;
  if (is_bad)
    ARCANE_FATAL("Can not convert to Int32[]");
  if (x.size()!=4)
    ARCANE_FATAL("Bad size");
  if (x[0]!=3 || x[1]!=7 || x[2]!=9 || x[3]!=12)
    ARCANE_FATAL("Bad values");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void UtilsUnitTest::MemoryInfoPrinter::
executeFunctor(const MemoryInfoChunk& chunk)
{
  if (chunk.size()>1024)
    m_trace_mng->info() << "BLOCK=" << chunk.size();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void UtilsUnitTest::
_testCommandLine()
{
  info() << "Test READ_COMMAND_LINE";
  StringList arg_list;
  platform::fillCommandLineArguments(arg_list);
  Int32 nb_arg = arg_list.count();
  info() << "NB_ARG=" << nb_arg;
  // La fonction peut retourner une liste vide si on ne supporte pas la
  // récupération des arguments de la ligne de commande sur la plateforme.
  if (nb_arg>=2){
    String arg_case_file = arg_list[nb_arg-1];
    info() << "ARG_CASE_FILE=" <<  arg_case_file;
    String expected_end = "testUtils-1.arc";
    if (!arg_case_file.endsWith(expected_end))
      ARCANE_FATAL("Bad value '{0}'. should ends with '{1}'",arg_case_file,expected_end);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void UtilsUnitTest::
_testHashAlgorithm()
{
  ValueChecker vc(A_FUNCINFO);
  UniqueArray<String> names = {
    "MD5HashAlgorithm",
    "SHA1HashAlgorithm",
    "SHA3_224HashAlgorithm",
    "SHA3_256HashAlgorithm",
    "SHA3_384HashAlgorithm",
    "SHA3_512HashAlgorithm"
  };
  ServiceBuilder<IHashAlgorithm> builder(subDomain()->application());
  UniqueArray<Int32> test_values(12345);
  for( Int32 i=0, n=test_values.size(); i<n; ++i )
    test_values[i] = 2 + (i*3);

  for( const String& name : names ){
    info() << "Create service name=" << name;
    Ref<IHashAlgorithm> service = builder.createReference(name);
    HashAlgorithmValue hash_value;
    service->computeHash(asBytes(test_values),hash_value);
    SmallSpan<const std::byte> hash_bytes(hash_value.bytes());
    vc.areEqual(hash_bytes.size(),service->hashSize(),"HashSize");
    info() << "Algo=" << name << " value=" << Convert::toHexaString(hash_bytes);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void UtilsUnitTest::
_testDataTypeNames()
{
  ValueChecker vc(A_FUNCINFO);
  vc.areEqual(String(dataTypeName(DT_BFloat16)),String("BFloat16"),"name BFloat16");
  vc.areEqual(String(dataTypeName(DT_Float16)),String("Float16"),"name Float16");
  vc.areEqual(String(dataTypeName(DT_Float32)),String("Float32"),"name Float32");

  vc.areEqual(dataTypeFromName("BFloat16"),DT_BFloat16,"-> BFloat16");
  vc.areEqual(dataTypeFromName("Float16"),DT_Float16,"-> Float16");
  vc.areEqual(dataTypeFromName("Float32"),DT_Float32,"-> Float32");

  vc.areEqual(dataTypeSize(DT_BFloat16),2,"sizeof BFloat16");
  vc.areEqual(dataTypeSize(DT_Float16),2,"sizeof Float16");
  vc.areEqual(dataTypeSize(DT_Float32),4,"sizeof Float32");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
