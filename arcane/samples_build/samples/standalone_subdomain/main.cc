// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* StandaloneSubDomain.cc                                      (C) 2000-2023 */
/*                                                                           */
/* Sample for ArcaneLauncher::createStandaloneSubDomain().                   */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! [StandaloneSubDomainFull]

#include "arcane/launcher/ArcaneLauncher.h"

#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/Real3.h"

#include "arcane/core/MeshReaderMng.h"
#include "arcane/core/IMesh.h"
#include "arcane/core/ISubDomain.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/ItemGroup.h"
#include "arcane/core/VariableTypes.h"

#include "arcane/utils/Exception.h"

#include <iostream>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arcane;

void executeSample()
{

  // Create a standalone subdomain
  // Arcane will automatically call finalization when the variable
  // goes out of scope.
  Arcane::StandaloneSubDomain launcher{ ArcaneLauncher::createStandaloneSubDomain(String{}) };
  Arcane::ISubDomain* sd = launcher.subDomain();

  // Get the trace class to display messages
  Arcane::ITraceMng* tm = launcher.traceMng();

  // Create an instance of the Mesh Reader.
  Arcane::MeshReaderMng mrm(sd);

  // Create a mesh named 'Mesh1' from the file 'plancher.msh'.
  // The format is automatically choosen from the extension
  Arcane::IMesh* mesh = mrm.readMesh("Mesh1", "plancher.msh");

  Int32 nb_cell = mesh->nbCell();
  tm->info() << "NB_CELL=" << nb_cell;

  // Loop over the cells and compute the center of each cell
  Arcane::VariableNodeReal3& nodes_coordinates = mesh->nodesCoordinates();
  ENUMERATE_ (Cell, icell, mesh->allCells()) {
    Arcane::Cell cell = *icell;
    Arcane::Real3 cell_center;
    // Iteration over nodes of the cell
    for (Node node : cell.nodes()) {
      cell_center += nodes_coordinates[node];
    }
    cell_center /= cell.nbNode();
    tm->info() << "Cell=" << cell.uniqueId() << " center=" << cell_center;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

int main(int argc, char* argv[])
{
  auto func = [&]{
    std::cout << "Sample: StandaloneSubDomain\n";
    // Initialize Arcane
    Arcane::CommandLineArguments cmd_line_args(&argc, &argv);
    Arcane::ArcaneLauncher::init(cmd_line_args);

    executeSample();
  };

  return arcaneCallFunctionAndCatchException(func);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! [StandaloneSubDomainFull]

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
