// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SimpleTableOutputExample4Module.hh                          (C) 2000-2022 */
/*                                                                           */
/* Exemple 4 de module utilisant ISimpleTableOutput en tant que service.     */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include <arcane/IParallelMng.h>
#include <arcane/ITimeLoopMng.h>
#include <arcane/ServiceBuilder.h>
#include <arcane/ISimpleTableOutput.h>


#include "example4/SimpleTableOutputExample4_axl.h"

using namespace Arcane;

/*!
  \brief Module SimpleTableOutputExample4.
 */
class SimpleTableOutputExample4Module : 
public ArcaneSimpleTableOutputExample4Object
{

 public:
  explicit SimpleTableOutputExample4Module(const ModuleBuildInfo& mbi)
  : ArcaneSimpleTableOutputExample4Object(mbi)
  {}

 public:
  void initModule() override;
  void loopModule() override;
  void endModule() override;

  VersionInfo versionInfo() const override { return VersionInfo(1, 0, 0); }

 protected:
  Integer m_pos_fis, m_pos_col;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_MODULE_SIMPLETABLEOUTPUTEXAMPLE4(SimpleTableOutputExample4Module);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
