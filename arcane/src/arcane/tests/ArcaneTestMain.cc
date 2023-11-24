// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#include <iostream>
#include "arcane_packages.h"
#include "arcane/utils/Exception.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/NotSupportedException.h"
#include "arcane/launcher/ArcaneLauncher.h"

// Avec MPC, il faut inclure ce fichier car MPC redéfini la fonction
// 'main' pour pouvoir l'exécuter avec plusieurs threads.
// Globalement, le programme se comporte alors comme si on exécutait
// la fonction 'main' une fois pour chaque thread.
#ifdef ARCANE_USE_MPC
#include <mpi.h>
#endif

#include "arcane/impl/ArcaneSimpleExecutor.h"

#include "arcane/MeshReaderMng.h"
#include "arcane/IMesh.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arcane;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Fonction d'initialisation de ApplicationInfo.
extern "C" ARCANE_IMPORT void
arcaneTestSetApplicationInfo();

extern "C++" ARCANE_IMPORT void
arcaneTestSetDotNetInitInfo();

void
_initDefaultApplicationInfo(const CommandLineArguments& cmd_line_args)
{
  ArcaneLauncher::init(cmd_line_args);
  arcaneTestSetApplicationInfo();
}

extern "C++" ARCANE_EXPORT int
arcaneTestDirectExecution(const CommandLineArguments& cmd_line_args,
                          const String& direct_execution_method);

extern "C++" ARCANE_EXPORT int
arcaneTestStandaloneSubDomainLauncher(const CommandLineArguments& cmd_line_args,
                                      const String& standalone_method);

extern "C++" ARCANE_EXPORT int
arcaneTestStandaloneAcceleratorLauncher(const CommandLineArguments& cmd_line_args,
                                        const String& standalone_method);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

int
_mainHelper(int argc,char* argv[])
{
  CommandLineArguments cmd_line_args(&argc,&argv);

  String standalone_subdomain_method = cmd_line_args.getParameter("StandaloneSubDomainMethod");
  if (!standalone_subdomain_method.null())
    return arcaneTestStandaloneSubDomainLauncher(cmd_line_args,standalone_subdomain_method);

#if defined(ARCANE_HAS_ACCELERATOR_API)
  String standalone_accelerator_method = cmd_line_args.getParameter("StandaloneAcceleratorMethod");
  if (!standalone_accelerator_method.null())
    return arcaneTestStandaloneAcceleratorLauncher(cmd_line_args,standalone_accelerator_method);
#endif

  _initDefaultApplicationInfo(cmd_line_args);
  String direct_exec_method = cmd_line_args.getParameter("DirectExecutionMethod");
  bool use_direct = !direct_exec_method.null();
  if (use_direct)
    return arcaneTestDirectExecution(cmd_line_args,direct_exec_method);

  arcaneTestSetDotNetInitInfo();

  int r = ArcaneLauncher::run();
  return r;
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
