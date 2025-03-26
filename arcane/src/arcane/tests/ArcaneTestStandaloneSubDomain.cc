// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ArcaneTestStandaloneArcaneSubDomain.cc                      (C) 2000-2025 */
/*                                                                           */
/* Test de ArcaneLauncher::createStandaloneSubDomain().                      */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/launcher/ArcaneLauncher.h"

#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/FatalErrorException.h"

#include "arcane/core/MeshReaderMng.h"
#include "arcane/core/IMesh.h"
#include "arcane/core/ISubDomain.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/ItemGroup.h"
#include "arcane/accelerator/core/IAcceleratorMng.h"
#include "arcane/accelerator/core/Runner.h"
#include "arcane/accelerator/core/DeviceMemoryInfo.h"

#if defined(ARCANE_HAS_ACCELERATOR_API)
#include "arcane/utils/NumArray.h"
#include "arcane/accelerator/NumArrayViews.h"
#include "arcane/accelerator/RunCommandLoop.h"
#endif

#include "arcane/utils/Exception.h"

#include <fstream>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arcane;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{
#if defined(ARCANE_HAS_ACCELERATOR_API)
void _testSum(IAcceleratorMng* acc_mng)
{
  // Test la somme de deux tableaux 'a' et 'b' dans un tableau 'c'.

  int nb_value = 10000;
  NumArray<Int64, MDDim1> a(nb_value);
  NumArray<Int64, MDDim1> b(nb_value);
  NumArray<Int64, MDDim1> c(nb_value);
  for (int i = 0; i < nb_value; ++i) {
    a(i) = i + 2;
    b(i) = i + 3;
  }

  {
    auto command = makeCommand(acc_mng->queue());
    auto in_a = viewIn(command, a);
    auto in_b = viewIn(command, b);
    auto out_c = viewOut(command, c);
    command << RUNCOMMAND_LOOP1(iter, nb_value)
    {
      auto [i] = iter();
      out_c(i) = in_a(i) + in_b(i);
    };
  }

  Int64 total = 0.0;
  for (int i = 0; i < nb_value; ++i)
    total += c(i);
  std::cout << "TOTAL=" << total << "\n";
  Int64 expected_total = 100040000;
  if (total != expected_total)
    ARCANE_FATAL("Bad value for sum={0} (expected={1})", total, expected_total);

  Accelerator::DeviceMemoryInfo dmi = acc_mng->runner().deviceMemoryInfo();
  std::cout << "DeviceMemoryInfo: free_mem=" << dmi.freeMemory()
            << " total=" << dmi.totalMemory() << "\n";
}
#endif

int _testStandaloneSubDomainLauncher1(const CommandLineArguments& cmd_line_args)
{
  std::cout << "TEST1: StandaloneSubDomain\n";
  ArcaneLauncher::init(cmd_line_args);
  StandaloneSubDomain launcher{ ArcaneLauncher::createStandaloneSubDomain(String{}) };
  ISubDomain* sd = launcher.subDomain();
  ITraceMng* tm = launcher.traceMng();
  ARCANE_CHECK_POINTER(sd);
  ARCANE_CHECK_POINTER(tm);
  MeshReaderMng mrm(sd);
  IMesh* mesh = mrm.readMesh("Mesh1", "plancher.msh");
  Int32 nb_cell = mesh->nbCell();
  tm->info() << "NB_CELL=" << nb_cell;
  Int32 expected_nb_cell = 196;
  if (nb_cell != expected_nb_cell) {
    tm->error() << String::format("Bad number of cells n={0} expected={1}", nb_cell, expected_nb_cell);
    return 1;
  }
  return 0;
}

int _testStandaloneSubDomainLauncher2(const CommandLineArguments& cmd_line_args)
{
  String case_file = "<?xml version=\"1.0\"?>"
                     "<case codename=\"Arcane\" xml:lang=\"en\" codeversion=\"1.0\">"
                     " <arcane>"
                     "  <titre>Test Mesh HoneyComb 2D</titre>"
                     "  <description>Test Mesh HoneyComb 2D</description>"
                     "  <timeloop>UnitTest</timeloop>"
                     " </arcane>"
                     ""
                     " <meshes>"
                     "   <mesh>"
                     "     <generator name=\"HoneyComb2D\">"
                     "       <origin>0.0 0.0</origin>"
                     "       <pitch-size>2.0</pitch-size>"
                     "       <nb-layer>10</nb-layer>"
                     "     </generator>"
                     "   </mesh>"
                     " </meshes>"
                     "</case>";
  std::cout << "TEST2: StandaloneSubDomain\n";
  String case_file_name = "subdomain_generated.arc";
  {
    std::ofstream ofile(case_file_name.localstr());
    ofile << case_file;
  }
  ArcaneLauncher::init(cmd_line_args);
  auto launcher{ ArcaneLauncher::createStandaloneSubDomain(case_file_name) };
  ISubDomain* sd = launcher.subDomain();
  ITraceMng* tm = launcher.traceMng();
  IAcceleratorMng* acc_mng = sd->acceleratorMng();
  IMesh* mesh = sd->defaultMesh();
  Int32 own_nb_cell = mesh->ownCells().size();
  Int32 nb_cell = mesh->parallelMng()->reduce(Parallel::ReduceSum, own_nb_cell);
  tm->info() << "Accelerator Runtime=" << acc_mng->runner().executionPolicy();
  tm->info() << "NB_CELL=" << nb_cell;
  Int32 expected_nb_cell = 271;
  if (nb_cell != expected_nb_cell) {
    tm->error() << String::format("Bad number of cells n={0} expected={1}", nb_cell, expected_nb_cell);
    return 1;
  }
#if defined(ARCANE_HAS_ACCELERATOR_API)
  _testSum(acc_mng);
#endif
  return 0;
}

int _testStandaloneSubDomainLauncher3()
{
  std::cout << "TEST3: StandaloneSubDomain\n";
  ArcaneLauncher::init(CommandLineArguments{});
  StandaloneSubDomain launcher{ ArcaneLauncher::createStandaloneSubDomain(String{}) };
  ISubDomain* sd = launcher.subDomain();
  ITraceMng* tm = launcher.traceMng();
  ARCANE_CHECK_POINTER(sd);
  ARCANE_CHECK_POINTER(tm);
  MeshReaderMng mrm(sd);
  IMesh* mesh = mrm.readMesh("Mesh1", "plancher.msh");
  Int32 nb_cell = mesh->nbCell();
  tm->info() << "NB_CELL=" << nb_cell;
  Int32 expected_nb_cell = 196;
  if (nb_cell != expected_nb_cell) {
    tm->error() << String::format("Bad number of cells n={0} expected={1}", nb_cell, expected_nb_cell);
    return 1;
  }
  return 0;
}

} // namespace

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_EXPORT int
arcaneTestStandaloneSubDomainLauncher(const CommandLineArguments& cmd_line_args,
                                      const String& method_name)
{
  int r = -1;
  try {
    if (method_name == "Test1")
      r = _testStandaloneSubDomainLauncher1(cmd_line_args);
    else if (method_name == "Test2")
      r = _testStandaloneSubDomainLauncher2(cmd_line_args);
    else if (method_name == "Test3")
      r = _testStandaloneSubDomainLauncher3();
    else
      ARCANE_FATAL("Unknown test name='{0}'", method_name);
  }
  catch (const Exception& ex) {
    std::cerr << "EXCEPTION: " << ex << "\n";
  }
  return r;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
