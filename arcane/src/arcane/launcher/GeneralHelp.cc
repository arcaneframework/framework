// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* GeneralHelp.cc                                              (C) 2000-2025 */
/*                                                                           */
/* Class managing the generic help message.                                  */
/*---------------------------------------------------------------------------*/

#include "arcane/launcher/GeneralHelp.h"

#include "arcane/core/ApplicationBuildInfo.h"

#include "arcane/impl/ArcaneMain.h"

#include "arcane/utils/ApplicationInfo.h"
#include "arcane/utils/CommandLineArguments.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void GeneralHelp::
printHelp()
{
  ApplicationInfo& infos = ArcaneMain::defaultApplicationInfo();
  const CommandLineArguments& args = infos.commandLineArguments();

  std::cout << infos.codeName() << " v" << infos.codeVersion() << std::endl;
  std::cout << std::endl;
  std::cout << "Usage:" << std::endl;
  std::cout << "    " << *args.commandLineArgv()[0] << " [OPTIONS] dataset.arc" << std::endl;
  std::cout << std::endl;
  std::cout << "General options:" << std::endl;
  std::cout << "    -h, --help                        Give this help list" << std::endl;
  std::cout << std::endl;
  std::cout << "Arcane option usage: -A,Option1=Value,Option2=Value" << std::endl;
  std::cout << "                          and/or" << std::endl;
  std::cout << "                     -A,Option1=Value -A,Option2=Value" << std::endl;
  std::cout << std::endl;
  std::cout << "Arcane options:" << std::endl;
  std::cout << "    -A,T=<Int32>                      Number of concurrent tasks to execute (default=1)" << std::endl;
  std::cout << "    -A,S=<Int32>                      Number of subdomains in shared memory" << std::endl;
  std::cout << "    -A,R=<Int32>                      Number of replicated subdomains (default=1)" << std::endl;
  std::cout << "    -A,P=<Int32>                      Number of processes to use for subdomains. This value is normally calculated automatically based on MPI parameters. It is only useful if you wish to use fewer processes for domain partitioning than those allocated for computation." << std::endl;
  std::cout << "    -A,AcceleratorRuntime=<String>    Accelerator runtime to use. The two possible values are cuda or hip. Arcane must be compiled with accelerator support for this option to be accessible. " << std::endl;
  std::cout << "    -A,MaxIteration=<VALUE>           Maximum number of iterations to perform for the execution. If the number of iterations specified by this variable is reached, the calculation stops." << std::endl;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
