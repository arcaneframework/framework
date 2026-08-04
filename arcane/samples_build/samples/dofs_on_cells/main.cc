// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* main.cc                                                     (C) 2000-2026 */
/*                                                                           */
/* Sample usage of DoF on cells.                                             */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

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
#include "arcane/core/IDoFFamily.h"
#include "arcane/core/IIndexedIncrementalItemConnectivityMng.h"
#include "arcane/core/IIndexedIncrementalItemConnectivity.h"
#include "arcane/core/IndexedItemConnectivityView.h"
#include "arcane/core/IIncrementalItemConnectivity.h"
#include "arcane/core/IItemConnectivityInfo.h"

#include "arcane/utils/Exception.h"

#include <iostream>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arcane;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief This class will handle the DoFs on Cells.
 *
 * There is one DoF for each node of the cell.
 */
class FemDoFsOnCells
: public TraceAccessor
{
 public:

  explicit FemDoFsOnCells(ITraceMng* tm)
  : TraceAccessor(tm)
  {
  }

 public:

  void initialize(IMesh* mesh);

 public:

  IndexedCellDoFConnectivityView cellDoFConnecvitiyView() const
  {
    return m_cell_dof_connectivity->view();
  }
  IItemFamily* dofFamily() const { return m_dof_family; }

 private:

  Ref<IIndexedIncrementalItemConnectivity> m_cell_dof_connectivity;
  IItemFamily* m_dof_family = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void FemDoFsOnCells::
initialize(IMesh* mesh)
{
  IItemFamily* cell_family = mesh->cellFamily();
  IItemFamily* dof_family_interface = mesh->findItemFamily(Arcane::IK_DoF, "DoFCellFamily", true);
  IDoFFamily* dof_family = ARCANE_CHECK_POINTER(dof_family_interface->toDoFFamily());
  m_dof_family = dof_family_interface;

  // Get the maximum number of nodes of cells across all sub-domains.
  // This will be used to compute the uniqueId of the DoF and make sure they
  // are always the same.
  Int32 max_nb_dof_per_cell = cell_family->globalConnectivityInfos()->maxNodePerItem();
  info() << "MAX_NB_DOF_PER_CELL=" << max_nb_dof_per_cell;

  ItemGroup all_cells = cell_family->allItems();
  // Create the DoFs
  UniqueArray<Int64> uids(all_cells.size() * max_nb_dof_per_cell);
  {
    Integer dof_index = 0;
    // Use a mask to make sure the uniqueId() of the dof
    // can not be negative if we multiply the uniqueId().
    const UInt64 uid_mask = (1 << 28) - 1;
    ENUMERATE_ (Cell, icell, all_cells) {
      Cell cell = *icell;
      Int32 nb_dof_per_cell = cell.nbNode();
      Int64 cell_unique_id = cell.uniqueId().asInt64();
      for (Integer i = 0; i < nb_dof_per_cell; ++i) {
        uids[dof_index] = (cell_unique_id & uid_mask) * max_nb_dof_per_cell + i;
        ++dof_index;
      }
    }
    uids.resize(dof_index);
  }
  //info() << "ADD_Dofs list=" << uids;
  Int32UniqueArray dof_lids(uids.size());
  dof_family->addDoFs(uids, dof_lids);
  dof_family->endUpdate();
  info() << "NB_DOF=" << dof_family->allItems().size();

  // Create Cell -> DoF connectivity.
  m_cell_dof_connectivity = mesh->indexedConnectivityMng()->findOrCreateConnectivity(mesh->cellFamily(), m_dof_family, "DoFCell");
  auto* cn = m_cell_dof_connectivity->connectivity();
  {
    Integer dof_index = 0;
    ENUMERATE_ (Cell, icell, all_cells) {
      Cell cell = *icell;
      Int32 nb_dof_per_cell = cell.nbNode();
      for (Integer i = 0; i < nb_dof_per_cell; ++i) {
        cn->addConnectedItem(cell, DoFLocalId(dof_lids[dof_index]));
        ++dof_index;
      }
    }
  }
  info() << "End build Dofs";

  IndexedCellDoFConnectivityView cell_dof(m_cell_dof_connectivity->view());
  {
    // Set the owners of the DoF.
    // It is only used when using message passing (i.e MPI)
    IParallelMng* pm = mesh->parallelMng();
    Int32 my_rank = pm->commRank();
    DoFInfoListView dofs_view(m_dof_family);
    ENUMERATE_ (Cell, icell, mesh->allCells()) {
      Cell cell = *icell;
      Int32 cell_owner = cell.owner();
      for (DoFLocalId dof : cell_dof.dofs(cell)) {
        dofs_view[dof].mutableItemBase().setOwner(cell_owner, my_rank);
      }
    }
    m_dof_family->notifyItemsOwnerChanged();
    m_dof_family->computeSynchronizeInfos();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void executeSample(const String& case_file)
{
  std::cout << "Sample: DoFsOnCells\n";

  // Create a standalone subdomain
  // Arcane will automatically call finalization when the variable
  // goes out of scope.
  StandaloneSubDomain launcher{ ArcaneLauncher::createStandaloneSubDomain(case_file) };
  ISubDomain* sd = launcher.subDomain();

  // Get the trace class to display messages
  ITraceMng* tm = launcher.traceMng();

  IMesh* mesh = sd->defaultMesh();
  IItemFamily* cell_family = mesh->cellFamily();

  Int32 nb_cell = mesh->nbCell();
  tm->info() << "NB_CELL=" << nb_cell;

  // We suppose the mesh is a bar with Y varying from 0.0 to 1.0.
  // The cells from 0.0 to 0.5 are triangles and cells from 0.5 to 1 are quadrangles.
  // The cells from 0.0 to 0.65 will be in group Mat1 and the remaining cells will be in group Mat2.
  VariableNodeReal3& nodes_coordinates = mesh->nodesCoordinates();
  UniqueArray<Int32> mat1_cells_id;
  UniqueArray<Int32> mat2_cells_id;
  ENUMERATE_ (Cell, icell, cell_family->allItems()) {
    Cell cell = *icell;
    Real3 cell_center;
    for (Node node : cell.nodes()) {
      cell_center += nodes_coordinates[node];
    }
    cell_center /= cell.nbNode();
    bool is_in_mat1 = cell_center.x < 0.65;
    //tm->info() << "Cell=" << cell.uniqueId() << " center=" << cell_center << " nb_node=" << cell.nbNode() << " is_mat1=" << is_in_mat1;
    if (is_in_mat1)
      mat1_cells_id.add(cell.localId());
    else
      mat2_cells_id.add(cell.localId());
  }

  // Create the groups
  CellGroup mat1_cells = cell_family->createGroup("Mat1", mat1_cells_id);
  CellGroup mat2_cells = cell_family->createGroup("Mat2", mat2_cells_id);
  tm->info() << "NB_MAT1=" << mat1_cells.size();
  tm->info() << "NB_MAT2=" << mat2_cells.size();

  // Create the dofs on cells.
  FemDoFsOnCells dofs_on_cells(tm);
  dofs_on_cells.initialize(mesh);

  // Declare a variable on DoFs and fill the value with the uniqueId() of the DoF.
  // The variable will be destroyed when the instance var_on_dof goes out of scope
  // because there is no remaining reference on it
  IItemFamily* dof_family = dofs_on_cells.dofFamily();
  VariableDoFReal var_on_dof(VariableBuildInfo(dof_family, "MyVarOnDoF"));
  ENUMERATE_ (DoF, idof, dof_family->allItems()) {
    DoF dof = *idof;
    var_on_dof[dof] = static_cast<Real>(dof.uniqueId().asInt64());
  }

  {
    // Iterate over the DoFs of cells of group Mat1.
    IndexedCellDoFConnectivityView cell_dof(dofs_on_cells.cellDoFConnecvitiyView());
    DoFInfoListView dofs_view(dof_family);
    ENUMERATE_ (Cell, icell, mat1_cells) {
      Cell cell = *icell;
      for (DoFLocalId dof : cell_dof.dofs(cell)) {
        tm->info() << "Cell=" << cell.uniqueId() << " dof_value=" << var_on_dof[dof] << " dof_uid=" << dofs_view[dof].uniqueId();
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

int main(int argc, char* argv[])
{

  auto func = [&] {
    // Initialize Arcane
    Arcane::CommandLineArguments cmd_line_args(&argc, &argv);
    Arcane::ArcaneLauncher::init(cmd_line_args);
    if (argc <= 1) {
      std::cout << "Usage: DoFsOnCells case_file.arc\n";
      return;
    }

    String case_file = argv[argc - 1];
    executeSample(case_file);
  };

  return arcaneCallFunctionAndCatchException(func);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! [StandaloneSubDomainFull]

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
