// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AlephIndexTest.cc                                           (C) 2000-2015 */
/*                                                                           */
/* Service du test du service Aleph+Index.                                   */
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ScopedPtr.h"

#include "arcane/aleph/tests/AlephTest.h"
#include "arcane/aleph/tests/AlephIndexTest.h"
#include "arcane/aleph/tests/AlephIndexTest_axl.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANETEST_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arcane;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class AlephIndexTest
: public ArcaneAlephIndexTestObject
{
 public:
  AlephIndexTest(const ModuleBuildInfo&);
  ~AlephIndexTest(void);
  void init(void);
  void compute(void);

 private:
  void setValues(const Real, AlephMatrix*);
  void postSolver(const Integer, Real, Array<Real>&, Array<Integer>&);
  static Real geoFaceSurface(Face, VariableItemReal3&);

 public:
  Integer m_total_nb_cell;
  Integer m_local_nb_cell;
  UniqueArray<Real> m_vector_zeroes;
  AlephKernel* m_aleph_kernel;
  UniqueArray<AlephMatrix*> m_aleph_mat;
  UniqueArray<AlephVector*> m_aleph_rhs;
  UniqueArray<AlephVector*> m_aleph_sol;
  UniqueArray<AlephParams*> m_aleph_params;
  Integer m_get_solution_idx;
  Integer m_nb_solver;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AlephIndexTest::
AlephIndexTest(const ModuleBuildInfo& mbi)
: ArcaneAlephIndexTestObject(mbi)
, m_total_nb_cell(0)
, m_local_nb_cell(0)
, m_vector_zeroes(0)
, m_aleph_kernel(0)
, m_get_solution_idx(0)
, m_nb_solver(0)
{}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AlephIndexTest::
~AlephIndexTest(void)
{
  debug() << "[AlephIndexTest::AlephIndexTest] Delete & Free";
}

/***************************************************************************
 * AlephIndexTest::init                                                   *
 ***************************************************************************/
void AlephIndexTest::
init(void)
{
  m_nb_solver = options()->alephNumberOfSolvers;

  m_aleph_mat.resize(m_nb_solver);
  m_aleph_rhs.resize(m_nb_solver);
  m_aleph_sol.resize(m_nb_solver);
  m_aleph_params.resize(m_nb_solver);

  // On set notre pas de temps
  m_global_deltat = 1.0;

  // initialisation de la temperature sur toutes les mailles et faces
  m_cell_temperature.fill(options()->initTemperature());
  m_face_temperature.fill(options()->initTemperature());

  // initialisation de la température aux limites, boucle sur les conditions aux limites
  for (int i = options()->boundaryCondition.size() - 1; i >= 0; --i) {
    Real temperature = options()->boundaryCondition[i]->value();
    FaceGroup face_group = options()->boundaryCondition[i]->surface();
    // boucle sur les faces de la surface
    ENUMERATE_FACE (iFace, face_group)
      m_face_temperature[iFace] = temperature;
  }
  mesh()->checkValidMeshFull();

  // On instancie un kernel minimaliste qui va prendre en charge l'init à notre place
  // L'AlephKernel(subDomain(),0,0); semble poser soucis, mais pas le 2,0, ni le 0,1:
  // problème d'initialisation de topologie pre-kernel?
  // La valeur 2 correspond au Kernel pour Hypre.
  m_aleph_kernel = new AlephKernel(subDomain(), 2, 1);

  // Encore une initialisation des indices des vecteurs
  ENUMERATE_CELL (cell, ownCells()) {
    m_vector_zeroes.add(0.0);
  }
}

/***************************************************************************
 * Entry points leads us here
 ***************************************************************************/
void AlephIndexTest::
compute(void)
{
  UniqueArray<SharedArray<Integer>> indexs;
  UniqueArray<SharedArray<Real>> values;
  Integer nb_iteration;
  Real residual_norm[4];

  if (m_nb_solver == 0) {
    subDomain()->timeLoopMng()->stopComputeLoop(true);
    return;
  }

  indexs.resize(m_nb_solver);
  values.resize(m_nb_solver);
  for (int i = 0; i < m_nb_solver; ++i)
    postSolver(i, options()->deltaT, values[i], indexs[i]);

  // And should be able to get the solutions after the last resolution was fired
  for (int i = m_nb_solver - 1; i >= 0; --i) {
    debug() << "[AlephIndexTest::compute] Getting solution #" << i;
    AlephVector* solution = m_aleph_kernel->syncSolver(i, nb_iteration, &residual_norm[0]);
    if (i != m_get_solution_idx)
      continue;
    debug() << "Solved in \33[7m" << nb_iteration << "\33[m iterations,"
            << "residuals=[\33[1m" << residual_norm[0] << "\33[m," << residual_norm[3] << "]";
    debug() << "[AlephIndexTest::compute] Applying solution #" << m_get_solution_idx;
    solution->getLocalComponents(values[i]);
    m_get_solution_idx += 1;
  }
  m_get_solution_idx %= m_nb_solver;

  // Sinon, on recopie les résultats
  debug() << "[AlephIndexTest::compute] Now get our results";
  ENUMERATE_CELL (cell, ownCells())
    m_cell_temperature[cell] = values[m_get_solution_idx][m_aleph_kernel->indexing()->get(m_cell_temperature, *cell)];

  // Si on a atteint notre maximum d'itérations, on sort
  if (subDomain()->commonVariables().globalIteration() >= options()->iterations)
    subDomain()->timeLoopMng()->stopComputeLoop(true);

  debug() << "[AlephIndexTest::compute] done";
}

/***************************************************************************
 * job
 ***************************************************************************/
void AlephIndexTest::
postSolver(const Integer i, Real optionDeltaT,
           Array<Real>& values,
           Array<Integer>& indexs)
{
  // On force les deltaT à être différents pour avoir des temps de calculs que l'on pourra ordonnancer
  Real deltaT = (1.0 + (Real)i) * optionDeltaT;
  Integer fake_nb_iteration = 0;
  Real fake_residual_norm[4];

  m_aleph_params[i] = new AlephParams(traceMng(),
                                      1.0e-10, // m_param_epsilon epsilon de convergence
                                      2000, // m_param_max_iteration nb max iterations
                                      TypesSolver::AMG, // m_param_preconditioner_method préconditionnement: DIAGONAL, AMG, IC
                                      TypesSolver::PCG, // m_param_solver_method méthode de résolution
                                      -1, // m_param_gamma
                                      -1.0, // m_param_alpha
                                      false, // m_param_xo_user par défaut Xo n'est pas égal à 0
                                      false, // m_param_check_real_residue
                                      false, // m_param_print_real_residue
                                      false, // m_param_debug_info
                                      1.e-40, // m_param_min_rhs_norm
                                      false, // m_param_convergence_analyse
                                      true, // m_param_stop_error_strategy
                                      false, // m_param_write_matrix_to_file_error_strategy
                                      "SolveErrorAlephMatrix.dbg", // m_param_write_matrix_name_error_strategy
                                      false, // m_param_listing_output
                                      0., // m_param_threshold
                                      false, // m_param_print_cpu_time_resolution
                                      0, // m_param_amg_coarsening_method: par défault celui de Sloop,
                                      0, // m_param_output_level
                                      1, // m_param_amg_cycle: 1-cycle amg en V, 2= cycle amg en W, 3=cycle en Full Multigrid V
                                      1, // m_param_amg_solver_iterations
                                      1, // m_param_amg_smoother_iterations
                                      TypesSolver::SymHybGSJ_smoother, // m_param_amg_smootherOption
                                      TypesSolver::ParallelRugeStuben, // m_param_amg_coarseningOption
                                      TypesSolver::CG_coarse_solver, // m_param_amg_coarseSolverOption
                                      false, // m_param_keep_solver_structure
                                      false, // m_param_sequential_solver
                                      TypesSolver::RB); // m_param_criteria_stop

  // Remplissage du second membre: conditions limites + second membre
  debug() << "[AlephIndexTest::compute] Resize (" << m_vector_zeroes.size() << ") + remplissage du second membre";
  values.resize(m_vector_zeroes.size());
  indexs.resize(m_vector_zeroes.size());
  ENUMERATE_CELL (cell, ownCells()) {
    Integer row = m_aleph_kernel->indexing()->get(m_cell_temperature, cell);
    indexs[row] = row;
    values[row] = m_cell_temperature[cell];
  }

  debug() << "[AlephIndexTest::compute] ENUMERATE_FACE";
  ENUMERATE_FACE (iFace, allCells().outerFaceGroup()) {
    if (!iFace->cell(0).isOwn())
      continue;
    values[m_aleph_kernel->indexing()->get(m_cell_temperature, iFace->cell(0))] +=
    deltaT * (m_face_temperature[iFace]) / geoFaceSurface(*iFace, nodesCoordinates());
  }

  // Création de la matrice MatVec et des besoins Aleph
  m_aleph_mat[i] = m_aleph_kernel->createSolverMatrix();
  m_aleph_rhs[i] = m_aleph_kernel->createSolverVector(); // First vector returned IS the rhs
  m_aleph_sol[i] = m_aleph_kernel->createSolverVector(); // Next one IS the solution

  m_aleph_mat[i]->create();
  m_aleph_rhs[i]->create();
  m_aleph_sol[i]->create();
  m_aleph_mat[i]->reset();

  // Remplissage de la matrice et assemblage
  setValues(deltaT, m_aleph_mat[i]);
  m_aleph_mat[i]->assemble();

  debug() << "[AlephIndexTest::job] setLocalComponents";
  //m_aleph_rhs[i]->setLocalComponents(indexs.size(), indexs.view(), values.view());
  m_aleph_rhs[i]->setLocalComponents(values.view());
  m_aleph_rhs[i]->assemble();

  //m_aleph_sol[i]->setLocalComponents(indexs.size(), indexs.view(), m_vector_zeroes.view());
  m_aleph_sol[i]->setLocalComponents(m_vector_zeroes.view());
  m_aleph_sol[i]->assemble();
  debug() << "[AlephIndexTest::job] done setLocalComponents";
  traceMng()->flush();

  // Now solve with Aleph
  debug() << "[AlephIndexTest::job] Now solve with Aleph";
  m_aleph_mat[i]->solve(m_aleph_sol[i],
                        m_aleph_rhs[i],
                        fake_nb_iteration,
                        &fake_residual_norm[0],
                        m_aleph_params[i],
                        true); // On souhaite poster de façon asynchrone
  debug() << "[AlephIndexTest::job] done with Aleph solve";
  traceMng()->flush();
}

/***************************************************************************
 * AlephTestModule::_FaceComputeSetValues                                  *
 ***************************************************************************/
void AlephIndexTest::setValues(const Real deltaT, AlephMatrix* aleph_mat)
{
  // On flush les coefs
  ENUMERATE_CELL (iCell, ownCells())
    m_cell_coefs[iCell] = 0.;
  // Faces 'inner'
  debug() << "[AlephIndexTest::setValues] inner-faces";
  ENUMERATE_FACE (iFace, allCells().innerFaceGroup()) {
    if (iFace->backCell().isOwn()) {
      const Real surface = geoFaceSurface(*iFace, nodesCoordinates());
      aleph_mat->setValue(m_cell_temperature, iFace->backCell(),
                          m_cell_temperature, iFace->frontCell(),
                          -deltaT / surface);
      m_cell_coefs[iFace->backCell()] += 1.0 / surface;
    }
    if (iFace->frontCell().isOwn()) {
      const Real surface = geoFaceSurface(*iFace, nodesCoordinates());
      aleph_mat->setValue(m_cell_temperature, iFace->frontCell(),
                          m_cell_temperature, iFace->backCell(),
                          -deltaT / surface);
      m_cell_coefs[iFace->frontCell()] += 1.0 / surface;
    }
  }
  debug() << "[AlephIndexTest::setValues] outer-faces";
  ENUMERATE_FACE (iFace, allCells().outerFaceGroup()) {
    if (!iFace->cell(0).isOwn())
      continue;
    m_cell_coefs[iFace->cell(0)] += 1.0 / geoFaceSurface(*iFace, nodesCoordinates());
  }
  debug() << "[AlephIndexTest::setValues] diagonale";
  ENUMERATE_CELL (cell, ownCells()) {
    aleph_mat->setValue(m_cell_temperature, cell,
                        m_cell_temperature, cell,
                        1.0 + deltaT * m_cell_coefs[cell]);
  }
  debug() << "[AlephIndexTest::setValues] done";
}

/***************************************************************************
 * geoFaceSurface
 ***************************************************************************/
Real AlephIndexTest::geoFaceSurface(Face face, VariableItemReal3& nodes_coords)
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
    return (xyz0 - xyz1).normL2();
  }
  throw FatalErrorException("geoFace", "Nb nodes != 4 !=2");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_DEFINE_STANDARD_MODULE(AlephIndexTest, AlephIndexTest);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANETEST_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
