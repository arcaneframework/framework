// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ArcaneTestStandaloneArcaneSubDomain.cc                      (C) 2000-2023 */
/*                                                                           */
/* Test de ArcaneLauncher::createStandaloneSubDomain().                      */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/launcher/ArcaneLauncher.h"

#include "arcane/utils/ITraceMng.h"

#include "arcane/core/MeshReaderMng.h"
#include "arcane/core/IMesh.h"

#include "arcane/utils/Exception.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arcane;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{

int _testStandaloneSubDomainLauncher(const CommandLineArguments& cmd_line_args,
                                     const String& method_name)
{
  ARCANE_UNUSED(method_name);
  std::cout << "TEST: StandaloneSubDomain\n";
  ArcaneLauncher::init(cmd_line_args);
  StandaloneSubDomain launcher{ ArcaneLauncher::createStandaloneSubDomain() };
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
    r = _testStandaloneSubDomainLauncher(cmd_line_args, method_name);
  }
  catch (const Exception& ex) {
    std::cerr << "EXCEPTION: " << ex << "\n";
  }
  return r;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
