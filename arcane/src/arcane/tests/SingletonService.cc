// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SingletonService.cc                                         (C) 2000-2018 */
/*                                                                           */
/* Service singleton de test.                                                */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Collection.h"
#include "arcane/utils/List.h"

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/BasicUnitTest.h"

#include "arcane/ServiceBuildInfo.h"

#include "arcane/tests/ArcaneTestGlobal.h"
#include "arcane/tests/IServiceInterface.h"
#include "arcane/tests/SingletonService_axl.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace ArcaneTest
{
using namespace Arcane;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Test des particules
 */
class SingletonService
: public ArcaneSingletonServiceObject
, public IServiceInterface1
{
 public:

  SingletonService(const ServiceBuildInfo& cb);
  ~SingletonService();

 public:
  Integer value() override { return 0; }
  void* getPointer1() override { return this; }
  void* getPointer2() override { return this; }
  String implementationName() const override { return "SingletonService"; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

SingletonService::
SingletonService(const ServiceBuildInfo& sbi)
: ArcaneSingletonServiceObject(sbi)
{
  info() << "INFO: Building SingletonService";
  options()->interface3.setDefaultValue("ServiceTestInterface3Impl2");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

SingletonService::
~SingletonService()
{
  info() << "INFO: Destroy SingletonService";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE_SINGLETONSERVICE(SingletonService1,SingletonService);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace ArcaneTest

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
