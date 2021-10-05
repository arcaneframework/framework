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

class CDImpl
: public ID
, public IC
{
 public:
  CDImpl(const Injector&){}
};

ARCANE_DI_REGISTER_PROVIDER(AImpl,
                            ProviderProperty("AImplProvider"),
                            ARCANE_DI_SERVICE_INTERFACE(IA));

ARCANE_DI_REGISTER_PROVIDER(BImpl,
                            ProviderProperty("BImplProvider"),
                            ARCANE_DI_SERVICE_INTERFACE(IB));

ARCANE_DI_REGISTER_PROVIDER(CDImpl,
                            ProviderProperty("CDImplProvider"),
                            ARCANE_DI_SERVICE_INTERFACE(IC),
                            ARCANE_DI_SERVICE_INTERFACE(ID)
                            );

}

TEST(DependencyInjection,TestBind1)
{
  using namespace Arcane::DependencyInjection;
  std::cout << "INJECTOR TEST\n";
  Injector injector;
  ITraceMng* tm = Arccore::arccoreCreateDefaultTraceMng();
  Ref<ITraceMng> ref_tm = makeRefFromInstance<ITraceMng>(tm);
  injector.bind<ITraceMng>(ref_tm);
  Ref<ITraceMng> tm2 = injector.get<ITraceMng>();
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

  Ref<IB> ib = injector.createInstance<IB>();
  EXPECT_TRUE(ib.get());
  ASSERT_EQ(ib->value(),12);
}
