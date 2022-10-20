// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SimpleTableComparatorExample3Module.hh                          (C) 2000-2022 */
/*                                                                           */
/* Exemple 2 de module utilisant ISimpleTableOutput en tant que singleton.   */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include <arcane/IParallelMng.h>
#include <arcane/ITimeLoopMng.h>
#include <arcane/ServiceBuilder.h>
#include <arcane/ISimpleTableOutput.h>
#include <arcane/ISimpleTableComparator.h>


#include "example3/SimpleTableComparatorExample3_axl.h"

using namespace Arcane;

/*!
  \brief Module SimpleTableComparatorExample3.
 */
class SimpleTableComparatorExample3Module : 
public ArcaneSimpleTableComparatorExample3Object
{

 public:
  explicit SimpleTableComparatorExample3Module(const ModuleBuildInfo& mbi)
  : ArcaneSimpleTableComparatorExample3Object(mbi)
  {}

 public:
  void initModule() override;
  void loopModule() override;
  void endModule() override;

  VersionInfo versionInfo() const override { return VersionInfo(1, 0, 0); }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_MODULE_SIMPLETABLECOMPARATOREXAMPLE3(SimpleTableComparatorExample3Module);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
