// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#include "arcane/launcher/ArcaneLauncher.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/StringList.h"

#include "arcane/MeshReaderMng.h"
#include "arcane/IMesh.h"
#include "arcane/ISubDomain.h"
#include "arcane/IParallelMng.h"
#include "arcane/ItemGroup.h"

#include <map>

using namespace Arcane;

namespace
{
String simple_exec_case_file_name;
typedef void (*DirectExecutionFunctorType)(DirectExecutionContext& x);
typedef void (*DirectSubDomainExecuteFunctorType)(DirectSubDomainExecutionContext& x);
std::map<String,DirectExecutionFunctorType> direct_func_map;
std::map<String,DirectSubDomainExecuteFunctorType> direct_sub_domain_func_map;

void
_addFunction(const String& func_name,DirectExecutionFunctorType functor)
{
  direct_func_map.insert(std::make_pair(func_name,functor));
}

void
_addFunction(const String& func_name,DirectSubDomainExecuteFunctorType functor)
{
  direct_sub_domain_func_map.insert(std::make_pair(func_name,functor));
}

void
TestRunDirect1(DirectExecutionContext& ctx)
{
  ISubDomain* sd = ctx.createSequentialSubDomain();
  MeshReaderMng mrm(sd);
  IMesh* mesh = mrm.readMesh("Mesh1","sod.vtk");
  std::cout << "NB_CELL=" << mesh->nbCell() << "\n";
}

void
TestRunDirectCartesianSequential(DirectExecutionContext& ctx)
{
  if (simple_exec_case_file_name.empty())
    ARCANE_FATAL("No case file specified");
  ISubDomain* sd = ctx.createSequentialSubDomain(simple_exec_case_file_name);
  IMesh* mesh = sd->defaultMesh();
  std::cout << "NB_CELL=" << mesh->nbCell() << "\n";
}

void
TestRunDirectCartesian(DirectSubDomainExecutionContext& ctx)
{
  ISubDomain* sd = ctx.subDomain();
  IMesh* mesh = sd->defaultMesh();
  std::cout << "NB_CELL=" << mesh->nbCell() << "\n";
}

void
TestRunDirectHelloWorld(DirectSubDomainExecutionContext& ctx)
{
  ISubDomain* sd = ctx.subDomain();
  std::cout << "SUB_DOMAIN_MY_RANK=" << sd->parallelMng()->commRank() << "\n";
}
}

extern "C++" ARCANE_EXPORT int
arcaneTestDirectExecution(const CommandLineArguments& cmd_line_args,
                          const String& direct_execution_method)
{
  // Ajoute un groupe nul avant initialisation pour tester la création de variable
  // globale de groupe.
  ItemGroup my_null_group;

  StringList all_args;
  cmd_line_args.fillArgs(all_args);
  // Récupère le jeu de données en considérant qu'il s'agit de la dernière option
  // de la ligne de commande.
  Integer nb_arg = all_args.count();
  if (nb_arg>=1)
    simple_exec_case_file_name = all_args[nb_arg-1];

  _addFunction("TestRunDirect1",TestRunDirect1);
  _addFunction("TestRunDirectCartesianSequential",TestRunDirectCartesianSequential);
  _addFunction("TestRunDirectCartesian",TestRunDirectCartesian);
  _addFunction("TestRunDirectHelloWorld",TestRunDirectHelloWorld);

  auto x1 = direct_func_map.find(direct_execution_method);
  auto x2 = direct_sub_domain_func_map.find(direct_execution_method);
  if (x1!=direct_func_map.end()){
    auto f1 = [=](DirectExecutionContext& ctx) -> int
              {
                (x1->second)(ctx);
                return 0;
              };
    return ArcaneLauncher::run(f1);
  }
  else if (x2!=direct_sub_domain_func_map.end()){
    auto f2 = [=](DirectSubDomainExecutionContext& ctx) -> int
              {
                (x2->second)(ctx);
                return 0;
              };
    return ArcaneLauncher::run(f2);
  }
  else{
    std::cerr << "ERROR: arcaneTestDirectExecution(): can not find method '" << direct_execution_method << "'\n";
  }
  return (-1);
}
