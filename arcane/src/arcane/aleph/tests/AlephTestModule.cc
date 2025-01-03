// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AlephUnitTest.cc                                            (C) 2000-2024 */
/*                                                                           */
/* Service du test du service Aleph.                                         */
/*---------------------------------------------------------------------------*/

#include "arcane/aleph/tests/AlephTest.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace ArcaneTest
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arcane;

AlephTestModule::
AlephTestModule(const ModuleBuildInfo& mbi)
: ArcaneAlephTestModuleObject(mbi)
, m_total_nb_cell(0)
, m_local_nb_cell(0)
, m_rows_nb_element(0)
, m_vector_indexs(0)
, m_vector_zeroes(0)
, m_vector_rhs(0)
, m_vector_lhs(0)
, m_aleph_kernel(NULL)
, m_aleph_factory(new AlephFactory(subDomain()->application(), traceMng()))
, m_aleph_params(NULL)
, m_aleph_mat(0)
, m_aleph_rhs(0)
, m_aleph_sol(0)
, m_get_solution_idx(0)
, m_fake_nb_iteration(0)
{
  debug() << "\33[37m[AlephTestModule::AlephTestModule] Loading AlephTestModule\33[0m";
}

AlephTestModule::
~AlephTestModule()
{
  delete m_aleph_kernel;
  delete m_aleph_params;
  delete m_aleph_factory;
}

/***************************************************************************
 * AlephTestModule::init                                                   *
 ***************************************************************************/
void AlephTestModule::
init()
{
  ItacFunction(AlephTestModule);
  m_aleph_mat.resize(options()->alephNumberOfSolvers);
  m_aleph_rhs.resize(options()->alephNumberOfSolvers);
  m_aleph_sol.resize(options()->alephNumberOfSolvers);

  // On récupère le nombre de mailles
  m_local_nb_cell = MESH_OWN_ACTIVE_CELLS(mesh()).size();
  m_total_nb_cell = subDomain()->parallelMng()->reduce(Parallel::ReduceSum, m_local_nb_cell);
  debug() << "\33[37m[AlephTestModule::init] m_total_nb_cell=" << m_total_nb_cell
          << ", local=" << m_local_nb_cell << "\33[0m";

  // On set notre pas de temps
  m_global_deltat = options()->deltaT;

  // initialisation de la temperature sur toutes les mailles et faces
  ENUMERATE_CELL (icell, allCells())
    m_cell_temperature[icell] = options()->initTemperature();
  ENUMERATE_FACE (iFace, allFaces())
    m_face_temperature[iFace] = options()->initTemperature();

  // Mise à jour des variables UID et SDID
  ENUMERATE_CELL (icell, allCells())
    m_sub_domain_id[icell] = subDomain()->subDomainId();
  ENUMERATE_CELL (icell, allCells())
    m_unique_id[icell] = icell->uniqueId().asInteger();

  // initialisation de la température aux limites
  options()->schema()->boundaries(options());

  // On raffine le ratio de mailles depuis le jeu de données
  if ((options()->initAmr() < 0) || (options()->initAmr() > 1.0))
    throw FatalErrorException("AlephTestModule::init()", "AMR ratio out of range");

  initAmrRefineMesh((Integer)(mesh()->cellFamily()->allItems().own().size() * options()->initAmr()));
  mesh()->checkValidMeshFull();

  initAlgebra();
}

/***************************************************************************
 *   
 ***************************************************************************/

void AlephTestModule::
initAlgebra()
{
  const Integer nb_row_size = m_total_nb_cell;
  const Integer nb_row_rank = m_local_nb_cell;
  ItacFunction(AlephTestModule);

  Int32 underlying_solver = options()->alephUnderlyingSolver;

  info() << "[AlephTestModule::initAlgebra] ALEPH init, " << nb_row_size
         << " lines in total, " << nb_row_rank << " for me";
  info() << "[AlephTestModule::initAlgebra] alephUnderlyingSolver="
         << underlying_solver;
  info() << "[AlephTestModule::initAlgebra] alephNumberOfCores="
         << options()->alephNumberOfCores;

  delete m_aleph_factory;
  m_aleph_factory = new AlephFactory(subDomain()->application(), traceMng());

  // On créé en place notre politique de scheduling par rapport à la topologie décrite
  if (m_aleph_kernel)
    delete m_aleph_kernel;

  m_aleph_kernel = new AlephKernel(traceMng(),
                                   subDomain(),
                                   m_aleph_factory,
                                   underlying_solver,
                                   options()->alephNumberOfCores,
                                   options()->alephCellOrdering);
  // Pour tester l'initialisation spécifique de PETSc, ajoute des arguments
  // si le nombre de PE est pair

  const bool use_is_even = (subDomain()->parallelMng()->commSize() % 2) == 0;
  const bool use_petsc_args = (underlying_solver==AlephKernel::SOLVER_PETSC && use_is_even);
  if (use_petsc_args){
    StringList slist1;
    slist1.add("-trmalloc");
    slist1.add("-log_trace");
    slist1.add("-help");
    CommandLineArguments args(slist1);
    m_aleph_kernel->solverInitializeArgs().setCommandLineArguments(args);
  }
  m_aleph_kernel->initialize(nb_row_size, nb_row_rank);

  delete m_aleph_params;
  m_aleph_params = new AlephParams(traceMng(),
                                   1.0e-10, // m_param_epsilon epsilon de convergence
                                   2000, // m_param_max_iteration nb max iterations
                                   TypesSolver::DIAGONAL, // m_param_preconditioner_method: DIAGONAL, AMG, IC
                                   TypesSolver::PCG, // m_param_solver_method méthode de résolution
                                   -1, // m_param_gamma
                                   -1.0, // m_param_alpha
                                   false, // m_param_xo_user par défaut Xo n'est pas égal à 0
                                   false, // m_param_check_real_residue
                                   false, // m_param_print_real_residue
                                   false, // m_param_debug_info
                                   1.e-20, // m_param_min_rhs_norm
                                   false, // m_param_convergence_analyse
                                   true, // m_param_stop_error_strategy
                                   false, // m_param_write_matrix_to_file_error_strategy
                                   "SolveErrorAlephMatrix.dbg", // m_param_write_matrix_name_error_strategy
                                   false, // m_param_listing_output
                                   0., // m_param_threshold
                                   false, // m_param_print_cpu_time_resolution
                                   0, // m_param_amg_coarsening_method: par défault celui de Sloop,
                                   0, // m_param_output_level
                                   1, // m_param_amg_cycle: 1=V, 2= W, 3=Full Multigrid V
                                   1, // m_param_amg_solver_iterations
                                   1, // m_param_amg_smoother_iterations
                                   TypesSolver::SymHybGSJ_smoother, // m_param_amg_smootherOption
                                   TypesSolver::ParallelRugeStuben, // m_param_amg_coarseningOption
                                   TypesSolver::CG_coarse_solver, // m_param_amg_coarseSolverOption
                                   false, // m_param_keep_solver_structure
                                   false, // m_param_sequential_solver
                                   TypesSolver::RB); // m_param_criteria_stop

  // Calcul des indices globaux de la matrice mailles x mailles
  // Et on prépare une fois pour toutes les indices des vecteurs
  m_cell_matrix_idx.fill(-1);
  {
    AlephInt idx = 0;
    // Tres important, car on y revient lors de la reconfiguration!!
    m_vector_indexs.resize(0);
    m_vector_zeroes.resize(0);
    m_vector_lhs.resize(0);
    const AlephInt row_offset = m_aleph_kernel->topology()->part()[m_aleph_kernel->rank()];
    ENUMERATE_CELL (iCell, MESH_OWN_ACTIVE_CELLS(mesh())) {
      m_cell_matrix_idx[iCell] = row_offset + idx;
      m_vector_indexs.add(m_cell_matrix_idx[iCell] = row_offset + idx);
      m_vector_zeroes.add(0.0);
      m_vector_lhs.add(0.0);
      idx += 1;
    }
  }
  m_cell_matrix_idx.synchronize();

  // Maintenant, on va compter le nombre d'éléments par lignes
  // On fait un scan comme on devra en faire dans la boucle
  //debug() << "\33[37m[AlephTestModule::initAlgebra] Comptage du nombre d'éléments des lignes locales"<<"\33[0m";
  m_rows_nb_element.resize(nb_row_rank);
  // Can now be forgotten
  options()->schema()->preFetchNumElementsForEachRow(m_rows_nb_element,
                                                     m_aleph_kernel->topology()->part()[m_aleph_kernel->rank()]);
  debug() << "\33[37m[AlephTestModule::initAlgebra] done\33[0m";
}

/***************************************************************************
 * Entry points leads us here
 ***************************************************************************/
void AlephTestModule::
compute()
{
  Integer delta_amr = 0;
  Integer nb_iteration;
  Real residual_norm[4];
  debug() << "\33[37m[AlephTestModule::compute]\33[0m";
  const Integer rank_offset = m_aleph_kernel->topology()->part()[m_aleph_kernel->rank()];
  ItacFunction(AlephTestModule);

  for (int i = 0; i < options()->alephNumberOfSolvers; i += 1)
    postSolver(i);

  // And should be able to get the solutions after the last resolution was fired
  for (int i = options()->alephNumberOfSolvers - 1; i >= 0; i -= 1) {
    debug() << "\33[37m[AlephTestModule::compute] Getting solution #" << i << "\33[0m";
    AlephVector* solution = m_aleph_kernel->syncSolver(i, nb_iteration, &residual_norm[0]);
    //if (i!=m_get_solution_idx) continue;
    info() << "\33[37mSolved in \33[7m" << nb_iteration << "\33[m iterations,"
           << "residuals=[\33[1m" << residual_norm[0] << "\33[m," << residual_norm[3] << "]";
    debug() << "\33[37m[AlephTestModule::compute] Applying solution #" << m_get_solution_idx << "\33[0m";
    solution->getLocalComponents(m_vector_indexs.size(), m_vector_indexs.view(), m_vector_lhs.view());
    m_get_solution_idx += 1;
  }
  m_get_solution_idx %= options()->alephNumberOfSolvers;

  //////////////////////////
  // AMR COARSEN & REFINE //
  //////////////////////////
  /*if (options()->schema()->amrRefine(m_vector_lhs, options()->trigRefine)==true){
    debug()<<"\33[37m[AlephTestModule::compute] AMR REFINE"<<"\33[0m";
    initAlgebra();
    delta_amr+=1;
    return;
  }
  if (options()->schema()->amrCoarsen(m_vector_lhs, options()->trigCoarse)==true){
    debug()<<"\33[37m[AlephTestModule::compute] AMR COARSEN"<<"\33[0m";
    initAlgebra();
    delta_amr+=1;
    return;
    }*/

  /////////////////////////////////////////////////////
  // DELETE or NOT the KERNEL between each iteration //
  /////////////////////////////////////////////////////
  if (options()->alephDeleteKernel() == true) {
    debug() << "\33[37m[AlephTestModule::compute] DELETE the KERNEL between each iteration"
            << "\33[0m";
    initAlgebra();
  }

  // Sinon, on recopie les résultats
  debug() << "\33[37m[AlephTestModule::compute] Now get our results"
          << "\33[0m";
  ENUMERATE_CELL (iCell, MESH_OWN_ACTIVE_CELLS(mesh())) {
    m_cell_temperature[iCell] = m_vector_lhs[m_cell_matrix_idx[iCell] - rank_offset];
  }

  // Si on a atteint notre maximum d'itérations, on sort
  if (subDomain()->commonVariables().globalIteration() >= options()->iterations + delta_amr)
    subDomain()->timeLoopMng()->stopComputeLoop(true);

  debug() << "\33[37m[AlephTestModule::compute] done"
          << "\33[0m";
}

/***************************************************************************
 * job
 ***************************************************************************/
void AlephTestModule::
postSolver(const Integer i)
{
  ItacFunction(AlephTestModule);
  debug() << "\33[37m[AlephTestModule::postSolver] #" << i << "\33[0m";
  const Integer rank_offset = m_aleph_kernel->topology()->part()[m_aleph_kernel->rank()];

  // Remplissage du second membre: conditions limites + second membre
  debug() << "\33[37m[AlephTestModule::postSolver] Remplissage du second membre"
          << "\33[0m";
  m_vector_rhs.resize(0);
  ENUMERATE_CELL (iCell, MESH_OWN_ACTIVE_CELLS(mesh())) {
    m_vector_rhs.add(m_cell_temperature[iCell]);
  }

  debug() << "\33[37m[AlephTestModule::postSolver] ENUMERATE_FACE"
          << "\33[0m";
  ENUMERATE_FACE (iFace, OUTER_ACTIVE_FACE_GROUP(allCells())) {
    if (!iFace->cell(0).isOwn())
      continue;
    m_vector_rhs[m_cell_matrix_idx[iFace->cell(0)] - rank_offset] +=
    options()->deltaT * (m_face_temperature[iFace]) / geoFaceSurface(*iFace, nodesCoordinates());
  }

  // Création de la matrice MatVec et des besoins Aleph
  m_aleph_mat.setAt(i, m_aleph_kernel->createSolverMatrix());
  m_aleph_rhs.setAt(i, m_aleph_kernel->createSolverVector()); // First vector returned IS the rhs
  m_aleph_sol.setAt(i, m_aleph_kernel->createSolverVector()); // Next one IS the solution

  //m_aleph_mat.at(i)->create(m_rows_nb_element);
  m_aleph_mat.at(i)->create();
  m_aleph_rhs.at(i)->create();
  m_aleph_sol.at(i)->create();

  // Remplissage de la matrice et assemblage
  options()->schema()->setValues(options()->deltaT, m_aleph_mat.at(i));
  m_aleph_mat.at(i)->assemble();

  debug() << "\33[37m[AlephTestModule::postSolver] setLocalComponents"
          << "\33[0m";
  m_aleph_rhs.at(i)->setLocalComponents(m_vector_indexs.size(), m_vector_indexs.view(), m_vector_rhs.view());
  m_aleph_rhs.at(i)->assemble();

  m_aleph_sol.at(i)->setLocalComponents(m_vector_indexs.size(), m_vector_indexs.view(), m_vector_zeroes.view());
  m_aleph_sol.at(i)->assemble();

  // Now solve with Aleph
  debug() << "\33[37m[AlephTestModule::postSolver] Now solve with Aleph"
          << "\33[0m";
  m_aleph_mat.at(i)->solve(m_aleph_sol.at(i),
                           m_aleph_rhs.at(i),
                           m_fake_nb_iteration,
                           &m_fake_residual_norm[0],
                           m_aleph_params,
                           true); // On souhaite poster de façon asynchrone
}

/***************************************************************************
 * geoFaceSurface
 ***************************************************************************/
Real AlephTestModule::
geoFaceSurface(Face face, VariableItemReal3& nodes_coords)
{
  if (face.nbNode() == 4) {
    const Real3 xyz0 = nodes_coords[face.node(0)];
    const Real3 xyz1 = nodes_coords[face.node(1)];
    const Real3 xyz3 = nodes_coords[face.node(3)];
    return (xyz0 - xyz1).normL2() * (xyz0 - xyz3).normL2();
  }
  if (face.nbNode() == 2) {
    const Real3 xyz0 = nodes_coords[face.node(0)];
    const Real3 xyz1 = nodes_coords[face.node(1)];
    return (xyz0 - xyz1).squareNormL2();
  }
  throw FatalErrorException("geoFace", "Nb nodes != 4 !=2");
}

/***************************************************************************
 * amrRefineMesh                                                           *
 ***************************************************************************/
void AlephTestModule::
initAmrRefineMesh(Integer nb_to_refine)
{
  ItacFunction(AlephTestModule);
  if (nb_to_refine == 0)
    return;

  info() << "Refining nb_to_refine=" << nb_to_refine;

  Int32UniqueArray cells_local_id;
  ENUMERATE_CELL (iCell, allCells()) {
    Cell cell = *iCell;
    if ((cell.type() == IT_Hexaedron8) || (cell.type() == IT_Quad4)) {
      if (nb_to_refine-- <= 0)
        break;
      info(5) << "\t[amrRefineMesh] refine cell.uid=" << cell.uniqueId();
      cells_local_id.add(cell.localId());
      //iItem->setFlags(iItem->flags() | ItemInternal::II_Refine);
    }
  }

  info() << "now refine";
	mesh()->modifier()->flagCellToRefine(cells_local_id);
	mesh()->modifier()->adapt();

  //MESH_MODIFIER_REFINE_ITEMS(mesh());

  // Now callBack the values
  CellInfoListView cells(mesh()->cellFamily());
  //ItemInternalList faces = mesh()->faceFamily()->itemsInternal();
  for (Integer i = 0, is = cells_local_id.size(); i < is; ++i) {
    Int32 lid = cells_local_id[i];
    Cell cell = cells[lid];
    //debug()<<"[amrRefineMesh] focus on cell #"<<lid<<", nbHChildren="<<cell.nbHChildren();
    for (Integer j = 0, js = CELL_NB_H_CHILDREN(cell); j < js; ++j) {
      //debug()<<"\t\t[amrRefineMesh] child cell #"<<cell.hChild(j).localId();
      m_cell_temperature[cells[CELL_H_CHILD(cell, j).localId()]] = m_cell_temperature[cells[lid]];
      auto faces = allCells().view()[CELL_H_CHILD(cell, j).localId()].toCell().faces();
      Integer index = 0;
      for( Face face : faces ){
        if (face.isSubDomainBoundary()) {
          //debug() << "\t\t\t[amrRefineMesh] outer face #"<< (*iFace).localId()<<", index="<<iFace.index()<<", T="<<m_face_temperature[cell.face(iFace.index())];
          m_face_temperature[face] = m_face_temperature[cell.face(index)];
        }
        else {
          //debug() << "\t\t\t[amrRefineMesh] inner face #"<< (*iFace).localId();//<<", T="<<m_face_temperature[face.toFace()];
          m_face_temperature[face] = 0;
        }
        ++index;
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_DEFINE_STANDARD_MODULE(AlephTestModule, AlephTest);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
