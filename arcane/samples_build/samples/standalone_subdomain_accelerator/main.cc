// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* StandaloneSubDomainAccelerator.cc                           (C) 2000-2026 */
/*                                                                           */
/* Sample for standalone SubDomain with accelerator loop.                    */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/launcher/ArcaneLauncher.h"

#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/FixedArray.h"

#include "arcane/core/IMesh.h"
#include "arcane/core/ISubDomain.h"
#include "arcane/core/IVariableMng.h"
#include "arcane/core/IPostProcessorWriter.h"
#include "arcane/core/VariableTypes.h"
#include "arcane/core/VariableBuildInfo.h"
#include "arcane/core/ServiceBuilder.h"
#include "arcane/core/UnstructuredMeshConnectivity.h"

#include "arcane/accelerator/core/IAcceleratorMng.h"
#include "arcane/accelerator/core/RunQueue.h"
#include "arcane/accelerator/RunCommandEnumerate.h"
#include "arcane/accelerator/VariableViews.h"

#include <iostream>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arcane;

// This sample will read the mesh from the data file `case_file` and
// then compute a perfect gas equation of state on the cells on this mesh
// using Arcane Accelerator API.

void executeSample(const String& case_file)
{
  // Create a standalone subdomain
  // Arcane will automatically call finalization when the variable
  // goes out of scope.
  Arcane::StandaloneSubDomain launcher{ ArcaneLauncher::createStandaloneSubDomain(case_file) };
  Arcane::ISubDomain* sd = launcher.subDomain();
  Arcane::IMesh* mesh = sd->defaultMesh();
  Arcane::IAcceleratorMng* acc_mng = sd->acceleratorMng();
  // Use the queue provided by 'acc_mng'. The backend will be
  // used depending of the command line.
  RunQueue run_queue = acc_mng->queue();

  // Get the trace class to display messages
  Arcane::ITraceMng* tm = launcher.traceMng();

  Int32 nb_cell = mesh->nbCell();
  tm->info() << "End reading mesh. NB_CELL=" << nb_cell;

  // Declare variables for perfect gaz equation of state
  VariableCellReal cell_pressure(VariableBuildInfo(mesh,"Pressure"));
  VariableCellReal cell_density(VariableBuildInfo(mesh,"Density"));
  VariableCellReal cell_adiabatic_cst(VariableBuildInfo(mesh,"AdiabaticCst"));
  VariableCellReal cell_internal_energy(VariableBuildInfo(mesh,"InternalEnergy"));
  VariableCellReal cell_sound_speed(VariableBuildInfo(mesh,"SoundSpeed"));

  // <variable nom="Density" valeur="0.125" groupe="ZD" />
  // <variable nom="Pressure" valeur="0.1" groupe="ZD" />
  // <variable nom="AdiabaticCst" valeur="1.4" groupe="ZD" />

  // This variable contains the coordinates of the nodes
  VariableNodeReal3& nodes_coordinates = mesh->nodesCoordinates();
  UnstructuredMeshConnectivityView connectivity_view(mesh);

  {
    tm->info() << "Initialize Pressure, Density and AdiabaticCst";
    // To make sure we do not have the same values for all cells,
    // the density will increase with the distance from origin for
    // the first node of the cell
    auto command = makeCommand(run_queue);
    auto in_nodes_coords = viewIn(command,nodes_coordinates);
    auto out_pressure = viewOut(command, cell_pressure);
    auto out_density = viewOut(command, cell_density);
    auto out_adiabatic_cst = viewOut(command, cell_adiabatic_cst);
    // Get the Cell->Node connectivity
    auto cnc = connectivity_view.cellNode();

    command << RUNCOMMAND_ENUMERATE (Cell, cell_id, mesh->allCells())
    {
      out_pressure[cell_id] = 0.125;
      out_adiabatic_cst[cell_id] = 1.4;
      // Get the first node of the cell and get the coordinate of this node
      NodeLocalId first_node = cnc.nodeId(cell_id, 0);
      Real d = 0.1 + math::normL2(in_nodes_coords[first_node]);
      out_density[cell_id] = d;
    };  
  }

  {
    tm->info() << "Compute SoundSpeed and InternalEnergy";
    auto command = makeCommand(run_queue);
    auto in_pressure = viewIn(command, cell_pressure);
    auto in_density = viewIn(command, cell_density);
    auto in_adiabatic_cst = viewIn(command, cell_adiabatic_cst);

    auto out_internal_energy = viewOut(command, cell_internal_energy);
    auto out_sound_speed = viewOut(command, cell_sound_speed);

    command << RUNCOMMAND_ENUMERATE (Cell, cell_id, mesh->allCells())
    {
      Real pressure = in_pressure[cell_id];
      Real adiabatic_cst = in_adiabatic_cst[cell_id];
      Real density = in_density[cell_id];
      out_internal_energy[cell_id] = pressure / ((adiabatic_cst - 1.0) * density);
      out_sound_speed[cell_id] = math::sqrt(adiabatic_cst * pressure / density);
    };
  }

  tm->info() << "End of computation";

  // Do post-processing
  {
    ServiceBuilder<IPostProcessorWriter> service_builder(sd);
    String service_name = "Ensight7PostProcessor";
    Ref<IPostProcessorWriter> post_processor = service_builder.createReference(service_name);
    post_processor->setBaseDirectoryName(".");
    // Only one time in this simulation
    FixedArray<Real, 1> times({ 0.0 });
    post_processor->setTimes(times.view());

    // Set the list of post-processed variables
    VariableList variables;
    variables.add(cell_pressure);
    variables.add(cell_density);
    variables.add(cell_adiabatic_cst);
    variables.add(cell_sound_speed);
    variables.add(cell_internal_energy);
    post_processor->setVariables(variables);
    sd->variableMng()->writePostProcessing(post_processor.get());

    tm->info() << "End doing post-processing";
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

int main(int argc, char* argv[])
{
  String case_file;

  auto func = [&] {
    std::cout << "Sample: StandaloneSubDomainAccelerator\n";
    // Initialize Arcane
    Arcane::CommandLineArguments cmd_line_args(&argc, &argv);
    Arcane::ArcaneLauncher::init(cmd_line_args);
    if (argc > 1)
      case_file = argv[argc - 1];
    executeSample(case_file);
  };

  return arcaneCallFunctionAndCatchException(func);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
