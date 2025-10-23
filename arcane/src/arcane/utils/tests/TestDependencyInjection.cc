// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#include <gtest/gtest.h>

#include "arccore/base/internal/DependencyInjection.h"
#include "arcane/utils/FatalErrorException.h"

#include "arcane/utils/ITraceMng.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arcane;

namespace DI_Test
{
using namespace Arcane::DependencyInjection;

class IA
{
 public:
  virtual ~IA() = default;
  virtual int value() const = 0;
};

class IB
{
 public:
  virtual ~IB() = default;
  virtual int value() const = 0;
};

class IA2
{
 public:
  virtual ~IA2() = default;
  virtual int value() const = 0;
  virtual IB* bValue() const = 0;
};

class IB2
{
 public:
  virtual ~IB2() = default;
  virtual int value() const = 0;
  virtual String stringValue() const = 0;
};

class IC
{
 public:
  virtual ~IC() = default;
  virtual int value() const = 0;
};

class ID
{
 public:
  virtual ~ID() = default;
};

class IE
{
 public:
  virtual ~IE() = default;
  virtual int intValue() const = 0;
  virtual String stringValue() const = 0;
};

class INone
{
 public:

  virtual ~INone() = default;
};

class AImpl
: public IA
{
 public:
  AImpl() {}
  int value() const override { return 5; }
};

class BImpl
: public IB
{
 public:
  BImpl() {}
  int value() const override { return 12; }
};

class B2Impl
: public IB2
{
 public:
  B2Impl(const String& x)
  : m_test(x)
  {}
  int value() const override { return 32; }
  String stringValue() const override { return m_test; }

 private:
  String m_test;
};

class EImpl
: public IE
{
 public:
  EImpl(Int32 int_value, const String& string_value)
  : m_string_value(string_value)
  , m_int_value(int_value)
  {
  }
  int intValue() const override { return m_int_value; }
  String stringValue() const override { return m_string_value; }

 private:
  String m_string_value;
  Int32 m_int_value;
};

class CDImpl
: public ID
, public IC
{
 public:
  CDImpl(int a,double b) : m_int_value(a+(int)b){}
  CDImpl(int a) : m_int_value(a){}
  CDImpl() : m_int_value(2){}
  int value() const override { return m_int_value; }
 private:
  int m_int_value;
};

class A2Impl
: public IA2
{
 public:
  A2Impl(int a,IB* ib,IA*) : m_a(a), m_ib(ib) {}
  A2Impl(int a, IB* ib)
  : m_a(a)
  , m_ib(ib)
  {}

 public:
  int value() const override { return m_a; }
  IB* bValue() const override { return m_ib; }

 private:
  int m_a;
  IB* m_ib;
};

ARCANE_DI_REGISTER_PROVIDER(AImpl,
                            ProviderProperty("AImplProvider"),
                            ARCANE_DI_INTERFACES(IA),
                            ARCANE_DI_EMPTY_CONSTRUCTOR());

ARCANE_DI_REGISTER_PROVIDER(BImpl,
                            ProviderProperty("BImplProvider"),
                            ARCANE_DI_INTERFACES(IB),
                            ARCANE_DI_EMPTY_CONSTRUCTOR());

ARCANE_DI_REGISTER_PROVIDER(B2Impl,
                            ProviderProperty("B2ImplProvider"),
                            ARCANE_DI_INTERFACES(IB2),
                            ARCANE_DI_CONSTRUCTOR(String));

ARCANE_DI_REGISTER_PROVIDER(EImpl,
                            ProviderProperty("EImplProvider"),
                            ARCANE_DI_INTERFACES(IE),
                            ARCANE_DI_CONSTRUCTOR(Int32, String));

ARCANE_DI_REGISTER_PROVIDER(A2Impl,
                            ProviderProperty("A2ImplProvider"),
                            ARCANE_DI_INTERFACES(IA2),
                            ARCANE_DI_CONSTRUCTOR(int, IB*));

ARCANE_DI_REGISTER_PROVIDER(CDImpl,
                            ProviderProperty("CDImplProvider2"),
                            ARCANE_DI_INTERFACES(IC, ID),
                            ARCANE_DI_CONSTRUCTOR(int));

ARCANE_DI_REGISTER_PROVIDER(CDImpl,
                            ProviderProperty("CDImplProvider3"),
                            ARCANE_DI_INTERFACES(IC),
                            ARCANE_DI_EMPTY_CONSTRUCTOR());

ARCANE_DI_REGISTER_PROVIDER(CDImpl,
                            ProviderProperty("CDImplProvider4"),
                            ARCANE_DI_INTERFACES(IC, ID),
                            ARCANE_DI_CONSTRUCTOR(int),
                            ARCANE_DI_CONSTRUCTOR(int, double),
                            ARCANE_DI_EMPTY_CONSTRUCTOR());
} // namespace DI_Test

TEST(DependencyInjection,TestPrintFactories)
{
  using namespace Arcane::DependencyInjection;
  Injector injector;
  injector.fillWithGlobalFactories();

  std::cout << "FACTORIES=" << injector.printFactories() << "\n";
}

namespace
{
template <typename T> void
_testNotFoundThrow(Arcane::DependencyInjection::Injector& injector)
{
  try {
    Ref<T> ic = injector.createInstance<T>("Test1");
    FAIL() << "Expected FatalErrorException";
  }
  catch (const FatalErrorException& ex) {
    std::cout << "EX=" << ex << "\n";
  }
  catch (...) {
    FAIL() << "Expected FatalErrorException";
  }
}
} // namespace

TEST(DependencyInjection, TestNotFound)
{
  using namespace Arcane::DependencyInjection;
  using namespace DI_Test;
  Injector injector;
  injector.fillWithGlobalFactories();

  _testNotFoundThrow<INone>(injector);
  _testNotFoundThrow<IA>(injector);
  _testNotFoundThrow<IC>(injector);
  Ref<IC> ic2 = injector.createInstance<IC>("Test1", true);
  ASSERT_EQ(ic2.get(), nullptr);
}

TEST(DependencyInjection,TestBind1)
{
  using namespace Arcane::DependencyInjection;
  std::cout << "INJECTOR TEST\n";
  Injector injector;
  ITraceMng* tm = Arccore::arccoreCreateDefaultTraceMng();
  Ref<ITraceMng> ref_tm = makeRefFromInstance<ITraceMng>(tm);
  injector.bind(ref_tm);
  Ref<ITraceMng> tm2 = injector.get<Ref<ITraceMng>>();
  std::cout << "TM=" << tm << "TM2=" << tm2.get() << "\n";
  ASSERT_EQ(tm,tm2.get()) << "Bad Get Reference";
}

TEST(DependencyInjection,ProcessGlobalProviders)
{
  using namespace Arcane::DependencyInjection;
  using namespace DI_Test;

  Injector injector;
  injector.fillWithGlobalFactories();

  Ref<IA> ia = injector.createInstance<IA>({});
  EXPECT_TRUE(ia.get());
  ASSERT_EQ(ia->value(),5);

  Ref<IA> ia2 = injector.createInstance<IA>("AImplProvider");
  EXPECT_TRUE(ia2.get());
  ASSERT_EQ(ia2->value(),5);

  Ref<IB> ib = injector.createInstance<IB>({});
  EXPECT_TRUE(ib.get());
  ASSERT_EQ(ib->value(),12);
}

void _TestBindValue()
{
  using namespace Arcane::DependencyInjection;
  using namespace DI_Test;

  {
    Injector injector;
    injector.fillWithGlobalFactories();
    String wanted_string("Toto");

    injector.bind(wanted_string);

    Ref<IB2> ib = injector.createInstance<IB2>({});
    EXPECT_TRUE(ib.get());
    ASSERT_EQ(ib->value(), 32);
    ASSERT_EQ(ib->stringValue(), wanted_string);
  }

  {
    Injector injector;
    injector.fillWithGlobalFactories();
    String wanted_string{ "Tata" };
    Int32 wanted_int{ 25 };

    //! Injecte les valeurs souhaitées
    injector.bind(wanted_string, "Name");
    injector.bind(wanted_int, "Value");

    //! Injecte des valeurs non utilisées pour tester.
    //injector.bind("FalseString","AnyName");
    //injector.bind(38,"SomeName");
    //injector.bind(3.2,"DoubleName");

    Ref<IE> ie = injector.createInstance<IE>("EImplProvider");
    EXPECT_TRUE(ie.get());
    ASSERT_EQ(ie->intValue(), wanted_int);
    ASSERT_EQ(ie->stringValue(), wanted_string);
  }
}

TEST(DependencyInjection,TestBindValue)
{
  try{
    _TestBindValue();
  }
  catch(const Exception& ex){
    std::cerr << "ERROR=" << ex << "\n";
    throw;
  }
}

TEST(DependencyInjection, ConstructorCall)
{
  using namespace DI_Test;
  namespace di = Arcane::DependencyInjection;
  using ConstructorType = di::impl::ConstructorRegisterer<int, IB*>;

  di::impl::ConcreteFactory<IA2, A2Impl, ConstructorType> c2f;

  int x = 3;

  try {
    Injector injector;
    std::unique_ptr<IB> ib{ std::make_unique<BImpl>() };
    injector.bind(ib.get());
    injector.bind(x);
    Ref<IA2> a2 = c2f.createReference(injector);
    ARCANE_CHECK_POINTER(a2.get());
    ASSERT_EQ(a2->value(), 3);
    ASSERT_EQ(a2->bValue(), ib.get());
  }
  catch (const Exception& ex) {
    std::cerr << "ERROR=" << ex << "\n";
    throw;
  }
}

TEST(DependencyInjection,Impl2)
{
  using namespace DI_Test;
  namespace di = Arcane::DependencyInjection;

  try{
    {
      // Test avec le constructeur CDImpl(int)
      Injector injector;
      injector.fillWithGlobalFactories();

      injector.bind<int>(25);
      Ref<IC> ic = injector.createInstance<IC>("CDImplProvider2");
      ARCANE_CHECK_POINTER(ic.get());
      ASSERT_EQ(ic->value(),25);
    }
    {
      // Test avec le constructeur sans arguments (CDImpl())
      // Dans ce cas la valeur IC::value() doit valoir 2
      // (voir constructeur de CDImpl)
      Injector injector;
      injector.fillWithGlobalFactories();
      Ref<IC> ic = injector.createInstance<IC>("CDImplProvider3");
      ARCANE_CHECK_POINTER(ic.get());
      ASSERT_EQ(ic->value(),2);
    }
    {
      // Test avec le constructeur avec 2 arguments (CDImpl(int,double))
      // Dans ce cas la valeur IC::value() doit valoir 25+12
      // (voir constructeur de CDImpl)
      Injector injector;
      injector.fillWithGlobalFactories();
      injector.bind<int>(25);
      injector.bind<double>(12.0);
      Ref<IC> ic = injector.createInstance<IC>("CDImplProvider4");
      ARCANE_CHECK_POINTER(ic.get());
      ASSERT_EQ(ic->value(),37);
    }
  }
  catch(const Exception& ex){
    std::cerr << "ERROR=" << ex << "\n";
    throw;
  }
}
