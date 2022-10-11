// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ArcaneDriverMain.cc                                         (C) 2000-2022 */
/*                                                                           */
/* Programme principal par défaut.                                           */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/CommandLineArguments.h"
#include "arcane/utils/ApplicationInfo.h"
#include "arcane/utils/Exception.h"

#include "arcane/impl/ArcaneMain.h"

using namespace Arcane;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

int
_mainHelper(int argc,char* argv[])
{
  ApplicationInfo& app_info = ArcaneMain::defaultApplicationInfo();
  app_info.setCommandLineArguments(CommandLineArguments(&argc,&argv));
  app_info.setCodeName("ArcaneDriver");
  app_info.setCodeVersion(VersionInfo(1,0,0));
  app_info.addDynamicLibrary("arcane_mpi");
  app_info.addDynamicLibrary("arcane_ios");
  app_info.addDynamicLibrary("arcane_std");
  app_info.addDynamicLibrary("arcane_mesh");
  app_info.addDynamicLibrary("arcane_cea");
  return ArcaneMain::run();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

int
main(int argc,char* argv[])
{
  int r = 0;
  int r2 = arcaneCallFunctionAndCatchException([&](){ r = _mainHelper(argc,argv); });
  if (r2!=0)
    return r2;
  return r;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
