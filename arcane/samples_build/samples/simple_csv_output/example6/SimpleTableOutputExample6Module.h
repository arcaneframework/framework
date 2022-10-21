// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SimpleTableOutputExample6Module.hh                          (C) 2000-2022 */
/*                                                                           */
/* Exemple 5 de module utilisant ISimpleTableOutput en tant que service.     */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include <arcane/IParallelMng.h>
#include <arcane/ITimeLoopMng.h>
#include <arcane/ServiceBuilder.h>
#include <arcane/ISimpleTableOutput.h>


#include "example6/SimpleTableOutputExample6_axl.h"

using namespace Arcane;

/*!
  \brief Module SimpleTableOutputExample6.
 */
class SimpleTableOutputExample6Module : 
public ArcaneSimpleTableOutputExample6Object
{

 public:
  explicit SimpleTableOutputExample6Module(const ModuleBuildInfo& mbi)
  : ArcaneSimpleTableOutputExample6Object(mbi)
  {}

 public:
  void initModule() override;
  void loopModule() override;
  void endModule() override;

  VersionInfo versionInfo() const override { return VersionInfo(1, 0, 0); }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_MODULE_SIMPLETABLEOUTPUTEXAMPLE6(SimpleTableOutputExample6Module);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
