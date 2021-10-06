// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#include <gtest/gtest.h>

#include "arcane/utils/DependencyInjection.h"

#include "arcane/utils/ITraceMng.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#define ARCANE_DI_INJECT(Signature)                                                                                    \
  using Inject = Signature;                                                                                            \
                                                                                                                       \
  Signature

using namespace Arcane;

namespace DI_Test
{
using namespace Arcane::DependencyInjection;

class IA
{
 public:
  virtual ~IA() = default;
  virtual int value() const =0;
};

class IB
{
 public:
  virtual ~IB() = default;
  virtual int value() const =0;
};


class IB2
{
 public:
  virtual ~IB2() = default;
  virtual int value() const =0;
  virtual String stringValue() const =0;
};

class IC
{
 public:
  virtual ~IC() = default;
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
  virtual int intValue() const =0;
  virtual String stringValue() const =0;
};

class AImpl
: public IA
{
 public:
  AImpl(const Injector&){}
  int value() const override { return 5; }
};

class BImpl
: public IB
{
 public:
  BImpl(const Injector&){}
  int value() const override { return 12; }
};

class B2Impl
: public IB2
{
 public:
  B2Impl(Injector& x): m_test(x.get<String>()) { }
  int value() const override { return 32; }
  String stringValue() const override { return m_test; }
 private:
  String m_test;
};

class EImpl
: public IE
{
 public:
  EImpl(Injector& x)
  : m_string_value(x.get<String>("Name")),
    m_int_value(x.get<Int32>("Value"))
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
  CDImpl(const Injector&){}
};

class A2Impl
: public IA
{
 public:
  ARCANE_DI_INJECT(A2Impl(int a,IB* ib)) : m_a(a), m_ib(ib) {}
 public:
  //AImpl(const Injector&){}
  int value() const override { return 23; }
 private:
  int m_a;
  IB* m_ib;
};

ARCANE_DI_REGISTER_PROVIDER(AImpl,
                            ProviderProperty("AImplProvider"),
                            ARCANE_DI_SERVICE_INTERFACE(IA));

ARCANE_DI_REGISTER_PROVIDER(BImpl,
                            ProviderProperty("BImplProvider"),
                            ARCANE_DI_SERVICE_INTERFACE(IB));

ARCANE_DI_REGISTER_PROVIDER(B2Impl,
                            ProviderProperty("B2ImplProvider"),
                            ARCANE_DI_SERVICE_INTERFACE(IB2));

ARCANE_DI_REGISTER_PROVIDER(CDImpl,
                            ProviderProperty("CDImplProvider"),
                            ARCANE_DI_SERVICE_INTERFACE(IC),
                            ARCANE_DI_SERVICE_INTERFACE(ID)
                            );

ARCANE_DI_REGISTER_PROVIDER(EImpl,
                            ProviderProperty("EImplProvider"),
                            ARCANE_DI_SERVICE_INTERFACE(IE));

}

TEST(DependencyInjection,TestBind1)
{
  using namespace Arcane::DependencyInjection;
  std::cout << "INJECTOR TEST\n";
  Injector injector;
  ITraceMng* tm = Arccore::arccoreCreateDefaultTraceMng();
  Ref<ITraceMng> ref_tm = makeRefFromInstance<ITraceMng>(tm);
  injector.bind<ITraceMng>(ref_tm);
  Ref<ITraceMng> tm2 = injector.getRef<ITraceMng>();
  std::cout << "TM=" << tm << "TM2=" << tm2.get() << "\n";
  ASSERT_EQ(tm,tm2.get()) << "Bad Get Reference";
}

TEST(DependencyInjection,ProcessGlobalProviders)
{
  using namespace Arcane::DependencyInjection;
  using namespace DI_Test;

  Injector injector;
  injector.fillWithGlobalFactories();

  Ref<IA> ia = injector.createInstance<IA>();
  EXPECT_TRUE(ia.get());
  ASSERT_EQ(ia->value(),5);

  Ref<IA> ia2 = injector.createInstance<IA>("AImplProvider");
  EXPECT_TRUE(ia2.get());
  ASSERT_EQ(ia2->value(),5);

  Ref<IB> ib = injector.createInstance<IB>();
  EXPECT_TRUE(ib.get());
  ASSERT_EQ(ib->value(),12);
}

TEST(DependencyInjection,TestBindValue)
{
  using namespace Arcane::DependencyInjection;
  using namespace DI_Test;

  {
    Injector injector;
    injector.fillWithGlobalFactories();
    String wanted_string("Toto");

    injector.bind(wanted_string);

    Ref<IB2> ib = injector.createInstance<IB2>();
    EXPECT_TRUE(ib.get());
    ASSERT_EQ(ib->value(),32);
    ASSERT_EQ(ib->stringValue(),wanted_string);
  }

  {
    Injector injector;
    injector.fillWithGlobalFactories();
    String wanted_string{"Tata"};
    Int32 wanted_int{25};

    //! Injecte les valeurs souhaitées
    injector.bind(wanted_string,"Name");
    injector.bind(wanted_int,"Value");

    //! Injecte des valeurs non utilisées pour tester.
    injector.bind("FalseString","AnyName");
    injector.bind(38,"SomeName");
    injector.bind(3.2,"DoubleName");

    Ref<IE> ie = injector.createInstance<IE>("EImplProvider");
    EXPECT_TRUE(ie.get());
    ASSERT_EQ(ie->intValue(),wanted_int);
    ASSERT_EQ(ie->stringValue(),wanted_string);
  }
}
