// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* GeneralHelp.cc                                              (C) 2000-2025 */
/*                                                                           */
/* Classe gérant le message d'aide générique.                                */
/*---------------------------------------------------------------------------*/

#include "arcane/launcher/GeneralHelp.h"
#include "arcane/utils/ApplicationInfo.h"
#include "arcane/core/ApplicationBuildInfo.h"

#include <arcane/impl/ArcaneMain.h>
#include <arcane/utils/CommandLineArguments.h>

namespace Arcane
{
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
  std::cout << "    -A,T=<Int32>                      Nombre de tâches concurrentes à exécuter (default=1)" << std::endl;
  std::cout << "    -A,S=<Int32>                      Nombre de sous-domaines en mémoire partagée" << std::endl;
  std::cout << "    -A,R=<Int32>                      Nombre de sous-domaines répliqués (default=1)" << std::endl;
  std::cout << "    -A,P=<Int32>                      Nombre de processus à utiliser pour les sous-domaines. Cette valeur est normalement calculée automatiquement en fonction des paramètres MPI. Elle n'est utile que si on souhaite utiliser moins de processus pour le partitionnement de domaine que ceux alloués pour le calcul." << std::endl;
  std::cout << "    -A,AcceleratorRuntime=<String>    Runtime accélérateur à utiliser. Les deux valeurs possibles sont cuda ou hip. Il faut avoir compiler Arcane avec le support des accélérateurs pour que cette option soit accessible. " << std::endl;
  std::cout << "    -A,MaxIteration=<VALUE>           Nombre maximum d'itérations à effectuer pour l'exécution. Si le nombre d'itérations spécifié par cette variable est atteint, le calcul s'arrête." << std::endl;
}

}
