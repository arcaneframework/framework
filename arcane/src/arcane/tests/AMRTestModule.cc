// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AMRUnitTest.cc                                              (C) 2000-2022 */
/*                                                                           */
/* AMR refinement/coarsening test service.                                   */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/StringBuilder.h"
#include "arcane/utils/AMRComputeFunction.h"
#include "arcane/utils/ScopedPtr.h"

#include "arcane/core/MeshVariableInfo.h"
#include "arcane/core/EntryPoint.h"
#include "arcane/core/ITimeLoop.h"
#include "arcane/core/ITimeLoopMng.h"
#include "arcane/core/TimeLoopEntryPointInfo.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/IMesh.h"
#include "arcane/core/IMeshModifier.h"
#include "arcane/core/Properties.h"
#include "arcane/core/Timer.h"
#include "arcane/core/SharedVariable.h"
#include "arcane/core/IMeshUtilities.h"
#include "arcane/core/ServiceBuilder.h"
#include "arcane/core/ItemPrinter.h"
#include "arcane/core/VariableCollection.h"
#include "arcane/core/IItemOperationByBasicType.h"

#include "arcane/tests/ArcaneTestGlobal.h"

#include "arcane/core/IMeshWriter.h"
#include "arcane/core/IPostProcessorWriter.h"
#include "arcane/core/Directory.h"
#include "arcane/core/IVariableMng.h"
#include "arcane/core/MeshStats.h"

#include "arcane/tests/AMRTest_axl.h"

#include "arcane/tests/AMR/ExactSolution.h"
#include "arcane/tests/AMR/ErrorEstimate.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace ArcaneTest
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arcane;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Mesh test module
 */
class AMRTestModule
: public ArcaneAMRTestObject
{
 public:

  explicit AMRTestModule(const ModuleBuildInfo& cb);
  ~AMRTestModule();

 public:

  void init();
  void compute();
  VariableCellReal* getData() { return old_data; }
  void transportFunction(Array<ItemInternal*>& old_items, AMROperationType op);

 private:

  void _refine(Integer nb_to_refine);
  void _coarsen(Integer nb_to_coarsen);
  void _loadBalance();

  void _writeMesh(const String& filename);
  void _postProcessAMR();
  void _checkCreateOutputDir();

  Integer _executeAnalyticAdaptiveLoop(RealArray& sol, IMesh* mesh);

  void _checkParents();

 private:

  IMesh* new_mesh = nullptr;
  // Post-processing
  RealUniqueArray times;
  RealUniqueArray m_error;
  VariableCellReal* new_data = nullptr;
  VariableCellReal* old_data = nullptr;
  Directory m_output_directory;
  bool m_output_dir_created;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_DEFINE_STANDARD_MODULE(AMRTestModule, AMRTest);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AMRTestModule::
AMRTestModule(const ModuleBuildInfo& mb)
: ArcaneAMRTestObject(mb)
, m_output_dir_created(false)
{
  addEntryPoint(this, "Init",
                &AMRTestModule::init,
                IEntryPoint::WInit);
  addEntryPoint(this, "compute",
                &AMRTestModule::compute);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AMRTestModule::
~AMRTestModule()
{
  delete old_data;
  delete new_data;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AMRTestModule::
init()
{
  //! AMR

  Integer nb_cell = mesh()->nbCell();
  Real amr_ratio = options()->amrRatio();
  amr_ratio = math::min(1.0, amr_ratio);
  amr_ratio = math::max(0.0, amr_ratio);
  const Integer nb_cell_old = static_cast<Integer>(nb_cell * amr_ratio);
  info() << "AMR Test nb_cell=" << nb_cell << " nb_to_refine=" << nb_cell_old;

  old_data = new VariableCellReal(Arcane::VariableBuildInfo(mesh(), "Proc",
                                                            mesh()->cellFamily()->name(),
                                                            Arcane::IVariable::PNoDump | Arcane::IVariable::PNoNeedSync));
  //	debug() << "OLD DATA SIZE " <<  old_data->variable()->nbElement() << "\n";
  ENUMERATE_CELL (icell, allCells()) {
    (*old_data)[icell] = 1.;
  }
  // creation of the data transport functor from one mesh to another
  AMRComputeFunction f(this, &AMRTestModule::transportFunction);
  // Registration of the functor by the functor manager associated with the mesh
  // NOTE: the object responsible for calling refinement must perform this registration
  mesh()->modifier()->registerCallBack(&f);

  FaceGroup face_group = mesh()->allCells().outerFaceGroup();

  new_mesh = subDomain()->defaultMesh();
  {
    //_refine(nb_cell_old);
    // We can perform as many phases as we want. However, currently
    // for performance reasons, we only perform 2 phases.
    for (Integer i = 0; i < 2; ++i) {
      _refine(nb_cell_old);
      _coarsen(nb_cell_old / 2);
    }
    Integer nb_active = new_mesh->allCells().size();
    info() << "NB_ACTIVE=" << nb_active;
    m_error.resize(nb_active);
    m_error.fill(0.0);
    _executeAnalyticAdaptiveLoop(m_error, new_mesh);
  }

  const Integer nb_cell_new = new_mesh->nbCell();
  info() << "NB_CELL_OLD= " << nb_cell_old << " NB_CELL_NEW= " << nb_cell_new << "\n";

  // Statistics on the new mesh
  MeshStats stats(traceMng(), new_mesh, subDomain()->parallelMng());
  stats.dumpStats();

  // NOTE: before destroying old_data, we can keep it to visualize the projection
  // of the data stored inside
  // old_data is used here to display the parallel procs
  delete old_data;

  old_data = new VariableCellReal(VariableBuildInfo(new_mesh, "Proc",
                                                    new_mesh->cellFamily()->name(),
                                                    IVariable::PNoDump | IVariable::PNoNeedSync));
  //debug() << "OLD DATA SIZE " <<  old_data->variable()->nbElement() << "\n";
  ENUMERATE_CELL (icell, allCells()) {
    (*old_data)[icell] = icell->owner();
  }
  new_data = new VariableCellReal(VariableBuildInfo(new_mesh, "Sol", new_mesh->cellFamily()->name(),
                                                    IVariable::PNoDump | IVariable::PNoNeedSync));
  {
    StringBuilder filename = "amrmesh";
    filename += subDomain()->subDomainId();
    filename += ".mli";
    _writeMesh(filename.toString());
  }
  {
    _checkCreateOutputDir();
    IPostProcessorWriter* post_processor = options()->format();
    post_processor->setBaseDirectoryName(m_output_directory.path());
    _postProcessAMR();
  }

  mesh()->modifier()->unRegisterCallBack(&f);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AMRTestModule::
compute()
{
  subDomain()->timeLoopMng()->stopComputeLoop(true);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AMRTestModule::
_refine(Integer nb_to_refine)
{
  Int32UniqueArray cells_local_id;
  const Integer nb_cell = mesh()->nbCell();
  ARCANE_ASSERT((nb_to_refine <= nb_cell), ("NB CELL TO REFINE EXCEED NB CELL OF THE MESH"));
  // Search for the first nb_to_refine cells of type IT_Hexaedron8
  ENUMERATE_CELL (icell, mesh()->ownActiveCells()) {
    Cell cell = *icell;
    if (cell.type() == IT_Hexaedron8 || cell.type() == IT_Quad4) {
      cells_local_id.add(cell.localId());
      nb_to_refine--;
      if (nb_to_refine == 0)
        break;
    }
  }
  info() << "NB_CELL_TO_REFINE=" << cells_local_id.size();
  mesh()->modifier()->flagCellToRefine(cells_local_id);
  mesh()->modifier()->adapt();
  _checkParents();
  MeshStats ms(traceMng(), mesh(), mesh()->parallelMng());
  ms.dumpStats();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AMRTestModule::
_coarsen(Integer nb_to_coarsen)
{
  info() << "Coarsening cells nb_to_coarsen=" << nb_to_coarsen;
  Int32UniqueArray cells_local_id;
  // Search for the first nb_to_coarsen cells of type IT_Hexaedron8
  Integer nb_child_to_coarsen = nb_to_coarsen * 8;
  ENUMERATE_CELL (icell, mesh()->ownActiveCells()) {
    Cell cell = *icell;
    if (cell.type() == IT_Hexaedron8 && cell.level() > 0) {

      Cell parent_cell = cell.hParent();
      for (Integer c = 0; c < parent_cell.nbHChildren(); c++) {
        Cell child = parent_cell.hChild(c);
        cells_local_id.add(child.localId());
      }
      nb_child_to_coarsen--;
      if (nb_child_to_coarsen <= 0)
        break;
    }
  }
  info() << "Computed nb to coarsen=" << cells_local_id.size();
  mesh()->modifier()->flagCellToCoarsen(cells_local_id);
  mesh()->modifier()->adapt();
  _checkParents();
  MeshStats ms(traceMng(), mesh(), mesh()->parallelMng());
  ms.dumpStats();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AMRTestModule::
_loadBalance()
{
  // Migration test
  VariableItemInt32& cells_new_owner = mesh()->toPrimaryMesh()->itemsNewOwner(IK_Cell);
  ENUMERATE_FACE (iface, allFaces()) {
    if (!iface->isOwn())
      for (CellLocalId icell : iface->cells())
        cells_new_owner[icell] = iface->owner();
  }
  info() << "Own cells before migration (" << ownCells().size() << " / " << allCells().size() << " )";
  //     ENUMERATE_CELL(icell,mesh()->ownCells()) info() << icell.index() << ": " << ItemPrinter(*icell);
  cells_new_owner.synchronize();
  Integer moved_cell_count = 0;
  ENUMERATE_CELL (icell, ownCells()) {
    if (cells_new_owner[icell] != icell->owner())
      ++moved_cell_count;
  }
  info() << "Own cells to move in migration : " << moved_cell_count;
  mesh()->utilities()->changeOwnersFromCells();
  //   for(IItemFamilyCollection::Enumerator i(mesh()->itemFamilies()); ++i;) {
  //     IItemFamily * family = *i;
  //     VariableItemInt32& owner = family->itemsNewOwner();
  //     const Integer subDomainId = subDomain()->subDomainId();
  //     UniqueArray<Integer> counts(subDomain()->nbSubDomain(),0);
  //     ENUMERATE_ITEM(iitem,family->allItems().own())
  //       ++counts[owner[iitem]];
  //     parallelMng()->reduce(Parallel::ReduceMax,counts);
  //     info() << "Total " << family->itemKind() << " for domain " << subDomainId << " : " << counts[subDomainId];
  //   }

  mesh()->modifier()->setDynamic(true);
  bool compact = mesh()->properties()->getBool("compact");
  mesh()->properties()->setBool("compact", true);
  mesh()->toPrimaryMesh()->exchangeItems();
  mesh()->properties()->setBool("compact", compact);
  info() << "Own cells after migration (" << ownCells().size() << " / " << allCells().size() << " )";
  //     ENUMERATE_CELL(icell,mesh()->ownCells()) info() << icell.index() << ": " << ItemPrinter(*icell);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AMRTestModule::
_writeMesh(const String& filename)
{
  ServiceBuilder<IMeshWriter> sb(subDomain());
  auto mesh_io(sb.createReference("Lima", SB_AllowNull));
  if (mesh_io.get())
    mesh_io->writeMeshToFile(mesh(), filename);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AMRTestModule::
_postProcessAMR()
{
  info() << "Post-process AMR";
  IPostProcessorWriter* post_processor = options()->format();
  times.add(m_global_time());
  post_processor->setTimes(times);
  post_processor->setMesh(new_mesh);

  /*m_data.fill(0.);
  Integer i=0;
  ENUMERATE_CELL(icell,new_mesh->allActiveCells()) {
    m_data[icell] = m_error[i++];
  }*/
  new_data->fill(0.);
  Integer i = 0;
  ENUMERATE_CELL (icell, new_mesh->allActiveCells()) {
    (*new_data)[icell] = m_error[i++];
  }

  VariableList variables;
  variables.add(new_data->variable());
  variables.add(old_data->variable());
  post_processor->setVariables(variables);
  ItemGroupList groups;
  groups.add(allCells());
  post_processor->setGroups(groups);
  IVariableMng* vm = subDomain()->variableMng();
  vm->writePostProcessing(post_processor);
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AMRTestModule::
_checkCreateOutputDir()
{
  if (m_output_dir_created)
    return;
  m_output_directory = Directory(subDomain()->exportDirectory(), "depouillement3");
  m_output_directory.createDirectory();
  m_output_dir_created = true;
  info() << "Creating output dir '" << m_output_directory.path() << "' for export";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer AMRTestModule::
_executeAnalyticAdaptiveLoop(RealArray& sol, IMesh* mesh)
{
  // Parse the adaptation options
  const Integer max_adapt_iters = 4;
  const Integer max_level = 3;
  const Real refine_percentage = 0.5;
  const Real coarsen_percentage = 0.2;

  singularity = true;

  // Creation of the ExactSolution object and attachment of solution functors
  ErrorEstimate exact_sol;
  exact_sol.attachExactValue(exact3DSolution);
  exact_sol.attachExactGradient(exact3DGradient);

  // Adaptive loop.
  Integer adapt_iter;
  RealUniqueArray error;
  for (adapt_iter = 0; adapt_iter < max_adapt_iters - 1; adapt_iter++) {
    info() << "Beginning Adaptive Loop " << adapt_iter << "\n";
    // Adaptation block
    {
      info() << "  Refining the mesh..." << "\n";

      // an ErrorEstimate object queries an approximate solution
      // and assigns a positive error value to each cell
      // This value is used for refinement decision making
      // coarsening.
      // For this simple test case, we use an error
      // of interpolation on the exact solution
      // For real cases, we need an error indicator
      // on the approximate solution.

      // Calculate the error for each active cell using the error indicator
      // Note: in the general case, a specific error estimator is needed
      // for the application.
      exact_sol.computeError(error, mesh);

      // infos
      info() << "L2-Error is: " << exact_sol.l2Error() << "\n";
      info() << "LInf-Error is: " << exact_sol.lInfError() << "\n";

      // Based on the error calculated in \p error, we decide which cell will be
      // refined or coarsened. In this example, the approach is as follows:
      // each cell with a percentage of 20% of the maximum error
      // will be refined, and each cell with 10% of the minimum error can be coarsened
      // It should be noted that cells flagged for refinement will be refined,
      // but cells flagged for coarsening can be coarsened.
      // ErrorToFlagConverter(error);
      exact_sol.errorToFlagConverter(error, refine_percentage, coarsen_percentage, max_level, mesh);

      // Adapt the mesh by refining and coarsening the flagged cells.
      // Projection of solutions, parameters, etc.
      // from the old mesh to the new mesh. For this, callbacks
      // are made in the MeshRefinement class
      mesh->modifier()->adapt();
    }
  }
  // the last iteration to calculate the error
  {
    info() << "Beginning Adaptive Loop " << adapt_iter << "\n";

    // Calculate the error.
    exact_sol.computeError(error, mesh);
    info() << "L2-Error is: " << exact_sol.l2Error() << "\n";
    info() << "LInf-Error is: " << exact_sol.lInfError() << "\n";

    exact_sol.computeSol(sol, mesh);
  }

  // All done
  return 0;
}

// -------------------------------------------------------------------
// Example of a callback function allowing the projection of
// variables/solutions during an AMR iteration.
// This function will be registered by the AMRTest module and will be
// therefore called throughout the AMR iterations.
// --------------------------------------------------------------------
void AMRTestModule::
transportFunction(Array<ItemInternal*>& old_items, AMROperationType op)
{
  // Prolongation/Restriction depending on the mesh operation
  VariableCellReal& data = *this->getData();
  switch (op) {
  case Restriction:
    for (Integer i = 0, is = old_items.size(); i < is; i++) {
      Cell parent = old_items[i];
      UInt32 nb_children = parent.nbHChildren();
      Real value = 0.;
      for (UInt32 j = 0; j < nb_children; j++) {
        value += data[parent.hChild(j)];
      }
      data[parent] = value / nb_children;
    }
    break;
  case Prolongation:
    for (Integer i = 0, is = old_items.size(); i < is; i++) {
      Cell parent = old_items[i];
      // coarse-to-fine: interpolation
      Real value = data[parent];
      for (UInt32 j = 0, js = parent.nbHChildren(); j < js; j++) {
        data[parent.hChild(j)] = value;
      }
    }
    break;
  default:
    ARCANE_FATAL("No callback function should be called with this operation {1}", op);
  }
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AMRTestModule::
_checkParents()
{
  ENUMERATE_CELL (icell, allCells()) {
    Cell c = *icell;
    Cell parent = c.topHParent();
    if (parent.null())
      ARCANE_FATAL("No topHParent() for cell {0}", ItemPrinter(c));
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace ArcaneTest

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
