// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SimpleTableComparatorExample1Module.hh                          (C) 2000-2022 */
/*                                                                           */
/* Exemple 1 de module utilisant ISimpleTableOutput en tant que singleton.   */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include <arcane/IParallelMng.h>
#include <arcane/ITimeLoopMng.h>
#include <arcane/ServiceBuilder.h>

#include <arcane/ISimpleTableOutput.h>
#include <arcane/ISimpleTableComparator.h>

#include "example1/SimpleTableComparatorExample1_axl.h"

using namespace Arcane;

/*!
  \brief Module SimpleTableComparatorExample1.
 */
class SimpleTableComparatorExample1Module : 
public ArcaneSimpleTableComparatorExample1Object
{

 public:
  explicit SimpleTableComparatorExample1Module(const ModuleBuildInfo& mbi)
  : ArcaneSimpleTableComparatorExample1Object(mbi)
  {}

 public:
  void initModule() override;
  void loopModule() override;
  void endModule() override;

  VersionInfo versionInfo() const override { return VersionInfo(1, 0, 0); }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_MODULE_SIMPLETABLECOMPARATOREXAMPLE1(SimpleTableComparatorExample1Module);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
