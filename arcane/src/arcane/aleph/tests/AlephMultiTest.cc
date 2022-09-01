// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AlephMultiTest.cc                                           (C) 2000-2015 */
/*                                                                           */
/* Service du test du service Aleph+Multi.                                   */
/*---------------------------------------------------------------------------*/
#include "arcane/aleph/tests/AlephTest.h"
#include "arcane/aleph/tests/AlephMultiTest_axl.h"
#include "arcane/aleph/tests/AlephMultiTest.h"

ARCANETEST_BEGIN_NAMESPACE
using namespace Arcane;

// ****************************************************************************
// * Classe 'solver' qui va mimer ce qui est fait à plus haut niveau
// ****************************************************************************
class AlephSolver
: public TraceAccessor
, public MeshAccessor
{
 public:
  AlephSolver(ITraceMng* traceMng,
              ISubDomain* subDomain,
              Integer numberOfResolutionsPerSolvers,
              Integer underlyingSolver,
              Integer numberOfCores,
              Real deltaT)
  : TraceAccessor(traceMng)
  , MeshAccessor(subDomain->defaultMesh())
  , m_sub_domain(subDomain)
  , m_aleph_number_of_resolutions_per_solvers(numberOfResolutionsPerSolvers)
  , m_aleph_underlying_solver(underlyingSolver)
  , m_aleph_number_of_cores(numberOfCores)
  , m_delta_t(deltaT)
  , m_aleph_params(new AlephParams())
  ,
  // On instancie un kernel minimaliste qui va prendre en charge l'init à notre place
  m_aleph_kernel(new AlephKernel(subDomain, underlyingSolver, numberOfCores))
  , m_aleph_mat(0)
  , m_aleph_rhs(0)
  , m_aleph_sol(0)
  , m_vector_zeroes(0)
  , m_get_solution_idx(0)
  {
    info() << Trace::Color::cyan() << "[AlephSolver] New solver "
           << "with " << numberOfResolutionsPerSolvers << " resolutions, "
           << "underlying solver is '" << underlyingSolver << "', "
           << "and numberOfCores is " << numberOfCores;
    m_aleph_mat.resize(numberOfResolutionsPerSolvers);
    m_aleph_rhs.resize(numberOfResolutionsPerSolvers);
    m_aleph_sol.resize(numberOfResolutionsPerSolvers);
    ENUMERATE_CELL (cell, ownCells())
      m_vector_zeroes.add(0.0);
  }

  ~AlephSolver()
  {
    info() << Trace::Color::cyan() << "[AlephSolver] Delet solver";
    delete m_aleph_kernel;
    delete m_aleph_params;
  }

  // ****************************************************************************
  // * launchResolutions
  // ****************************************************************************
  void launchResolutions(VariableCellReal& cell_temperature,
                         VariableFaceReal& face_temperature)
  {
    UniqueArray<SharedArray<Integer>> indexs;
    UniqueArray<SharedArray<Real>> values;
    Integer nb_iteration;
    Real residual_norm[4];

    indexs.resize(m_aleph_number_of_resolutions_per_solvers);
    values.resize(m_aleph_number_of_resolutions_per_solvers);

    for (int i = 0;
         i < m_aleph_number_of_resolutions_per_solvers;
         i += 1)
      postSingleResolution(cell_temperature,
                           face_temperature,
                           i, m_delta_t, values[i], indexs[i]);

    // And should be able to get the solutions after the last resolution was fired
    for (int i = m_aleph_number_of_resolutions_per_solvers - 1; i >= 0; i -= 1) {
      debug() << Trace::Color::cyan() << "[AlephSolver::launchResolutions] Getting solution #" << i;
      AlephVector* solution = m_aleph_kernel->syncSolver(i, nb_iteration, &residual_norm[0]);
      if (i != m_get_solution_idx)
        continue;
      info() << Trace::Color::cyan() << "Solved in \33[7m" << nb_iteration << "\33[m iterations,"
             << "residuals=[\33[1m" << residual_norm[0] << "\33[m," << residual_norm[3] << "]";
      debug() << Trace::Color::cyan() << "[AlephSolver::launchResolutions] Applying solution #" << m_get_solution_idx;
      solution->getLocalComponents(values[i]);
      m_get_solution_idx += 1;
    }
    m_get_solution_idx %= m_aleph_number_of_resolutions_per_solvers;
    // Sinon, on recopie les résultats
    debug() << Trace::Color::cyan() << "[AlephSolver::launchResolutions] Now get our results";
    ENUMERATE_CELL (cell, ownCells())
      cell_temperature[cell] =
      values[m_get_solution_idx][m_aleph_kernel->indexing()->get(cell_temperature, *cell)];
    debug() << Trace::Color::cyan() << "[AlephSolver::launchResolutions] done";
  }

  // ****************************************************************************
  // * postSingleResolution
  // ****************************************************************************
  void postSingleResolution(VariableCellReal& cell_temperature,
                            VariableFaceReal& face_temperature,
                            const Integer i,
                            Real optionDeltaT,
                            Array<Real>& values,
                            Array<Integer>& indexs)
  {
    // On force les deltaT à être différents pour avoir des temps de calculs que l'on pourra ordonnancer
    Real deltaT = (1.0 + (Real)i) * optionDeltaT;
    Integer fake_nb_iteration = 0;
    Real fake_residual_norm[4];

    // Remplissage du second membre: conditions limites + second membre
    debug() << Trace::Color::cyan()
            << "[AlephSolver::postSingleResolution] Resize (" << m_vector_zeroes.size()
            << ") + remplissage du second membre";
    values.resize(m_vector_zeroes.size());
    indexs.resize(m_vector_zeroes.size());
    ENUMERATE_CELL (cell, ownCells()) {
      Integer row = m_aleph_kernel->indexing()->get(cell_temperature, cell);
      //info()<<Trace::Color::yellow()<<"[AlephSolver::postSingleResolution] row="<<row;
      indexs[row] = row;
      values[row] = cell_temperature[cell];
    }

    debug() << Trace::Color::cyan() << "[AlephSolver::postSingleResolution] ENUMERATE_FACE";
    ENUMERATE_FACE (iFace, allCells().outerFaceGroup()) {
      if (!iFace->cell(0).isOwn())
        continue;
      values[m_aleph_kernel->indexing()->get(cell_temperature, iFace->cell(0))] +=
      deltaT * (face_temperature[iFace]) / geoFaceSurface(*iFace, nodesCoordinates());
    }

    // Création de la matrice MatVec et des besoins Aleph
    m_aleph_mat.setAt(i, m_aleph_kernel->createSolverMatrix());
    m_aleph_rhs.setAt(i, m_aleph_kernel->createSolverVector()); // First vector returned IS the rhs
    m_aleph_sol.setAt(i, m_aleph_kernel->createSolverVector()); // Next one IS the solution

    m_aleph_mat.at(i)->create();
    m_aleph_rhs.at(i)->create();
    m_aleph_sol.at(i)->create();
    m_aleph_mat.at(i)->reset();

    // Remplissage de la matrice et assemblage
    setValues(cell_temperature, face_temperature, deltaT, m_aleph_mat.at(i));
    m_aleph_mat.at(i)->assemble();

    debug() << Trace::Color::cyan() << "[AlephSolver::postSingleResolution] setLocalComponents";
    //m_aleph_rhs.at(i)->setLocalComponents(indexs.size(), indexs.view(), values.view());
    m_aleph_rhs.at(i)->setLocalComponents(values.view());
    m_aleph_rhs.at(i)->assemble();

    //m_aleph_sol.at(i)->setLocalComponents(indexs.size(), indexs.view(), m_vector_zeroes.view());
    m_aleph_sol.at(i)->setLocalComponents(m_vector_zeroes.view());
    m_aleph_sol.at(i)->assemble();

    debug() << Trace::Color::cyan() << "\33[37m[AlephSolver::postSingleResolution] Now solve with Aleph";
    m_aleph_mat.at(i)->solve(m_aleph_sol.at(i),
                             m_aleph_rhs.at(i),
                             fake_nb_iteration,
                             &fake_residual_norm[0],
                             m_aleph_params,
                             true); // On souhaite poster de façon asynchrone
  }

  // ****************************************************************************
  // * setValues for a matrix
  // ****************************************************************************
  void setValues(VariableCellReal& cell_temperature,
                 [[maybe_unused]] VariableFaceReal& face_temperature,
                 const Real deltaT, AlephMatrix* aleph_mat)
  {
    VariableCellReal coefs(VariableBuildInfo(m_sub_domain->defaultMesh(), "cellCoefs"));
    // On flush les coefs
    ENUMERATE_CELL (cell, ownCells())
      coefs[cell] = 0.;
    // Faces 'inner'
    debug() << Trace::Color::cyan() << "[AlephSolver::setValues] inner-faces";
    ENUMERATE_FACE (iFace, allCells().innerFaceGroup()) {
      if (iFace->backCell().isOwn()) {
        const Real surface = geoFaceSurface(*iFace, nodesCoordinates());
        aleph_mat->setValue(cell_temperature, iFace->backCell(),
                            cell_temperature, iFace->frontCell(),
                            -deltaT / surface);
        coefs[iFace->backCell()] += 1.0 / surface;
      }
      if (iFace->frontCell().isOwn()) {
        const Real surface = geoFaceSurface(*iFace, nodesCoordinates());
        aleph_mat->setValue(cell_temperature, iFace->frontCell(),
                            cell_temperature, iFace->backCell(),
                            -deltaT / surface);
        coefs[iFace->frontCell()] += 1.0 / surface;
      }
    }
    debug() << Trace::Color::cyan() << "[AlephSolver::setValues] outer-faces";
    ENUMERATE_FACE (iFace, allCells().outerFaceGroup()) {
      if (!iFace->cell(0).isOwn())
        continue;
      coefs[iFace->cell(0)] += 1.0 / geoFaceSurface(*iFace, nodesCoordinates());
    }
    debug() << Trace::Color::cyan() << "[AlephSolver::setValues] diagonale";
    ENUMERATE_CELL (cell, ownCells()) {
      aleph_mat->setValue(cell_temperature, cell,
                          cell_temperature, cell,
                          1.0 + deltaT * coefs[cell]);
    }
    debug() << Trace::Color::cyan() << "[AlephSolver::setValues] done";
  }

 private:
  // ****************************************************************************
  // * geoFaceSurface
  // ****************************************************************************
  Real geoFaceSurface(Face face, VariableItemReal3& nodes_coords)
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

 private:
  ISubDomain* m_sub_domain;
  Integer m_aleph_number_of_resolutions_per_solvers;
  Integer m_aleph_underlying_solver;
  Integer m_aleph_number_of_cores;
  Real m_delta_t;
  AlephParams* m_aleph_params;
  AlephKernel* m_aleph_kernel;
  UniqueArray<AlephMatrix*> m_aleph_mat;
  UniqueArray<AlephVector*> m_aleph_rhs;
  UniqueArray<AlephVector*> m_aleph_sol;
  UniqueArray<Real> m_vector_zeroes;
  Integer m_get_solution_idx;
};

// ****************************************************************************
// * Classe AlephMultiTest + deletes
// ****************************************************************************
AlephMultiTest::
AlephMultiTest(const ModuleBuildInfo& mbi)
: ArcaneAlephMultiTestObject(mbi)
{
}

AlephMultiTest::
~AlephMultiTest(void)
{
  // NOTE: il ne faut pas détruire les éléments de \a m_posted_solvers
  delete m_aleph_factory;
  debug() << Trace::Color::cyan() << "[AlephMultiTest::AlephMultiTest] Delete";
  for (Integer i = 0, n = m_global_aleph_solver.size(); i < n; ++i)
    delete m_global_aleph_solver[i];
}

// ****************************************************************************
// * Point d'entrée d'initialisations
// ****************************************************************************
void AlephMultiTest::
init(void)
{
  ISubDomain* sd = subDomain();
  m_aleph_factory = new AlephFactory(sd->application(), sd->traceMng());

  // Initialisation du pas de temps
  m_global_deltat = options()->deltaT;
  // Initialisation des temperatures des mailles et des faces extérieures
  m_cell_temperature.fill(options()->iniTemperature());

  ENUMERATE_FACE (iFace, outerFaces()) {
    m_face_temperature[iFace] = options()->hotTemperature();
  }
  mesh()->checkValidMeshFull();

  // Initialisation des solveurs globaux
  for (Integer i = 0; i < options()->alephNumberOfSuccessiveSolvers(); ++i) {
    SolverBuildInfo sbi;
    Integer underlying_solver = (options()->alephUnderlyingSolver >> (4ul * i)) & 0xFul;
    if (!m_aleph_factory->hasSolverImplementation(underlying_solver)) {
      info() << "Skipping solver index " << underlying_solver
             << " because implementation is not available";
      continue;
    }

    sbi.m_number_of_resolution_per_solvers = (options()->alephNumberOfResolutionsPerSolvers >> (4ul * i)) & 0xFul;
    sbi.m_underliying_solver = underlying_solver;
    sbi.m_number_of_core = (options()->alephNumberOfCores >> (4ul * i)) & 0xFul;
    m_solvers_build_info.add(sbi);
  }

  // Initialisation des solveurs globaux
  for (Integer i = 0, n = m_solvers_build_info.size(); i < n; ++i) {
    const SolverBuildInfo& sbi = m_solvers_build_info[i];
    m_global_aleph_solver.add(new AlephSolver(traceMng(), subDomain(),
                                              sbi.m_number_of_resolution_per_solvers,
                                              sbi.m_underliying_solver,
                                              sbi.m_number_of_core,
                                              options()->deltaT));
  }
}

// ****************************************************************************
// * Point d'entrée de la boucle de calcul
// ****************************************************************************
void AlephMultiTest::
compute(void)
{
  // Pour chaque solver, on lance les résolutions
  for (Integer i = 0, n = m_solvers_build_info.size(); i < n; ++i) {
    const SolverBuildInfo& sbi = m_solvers_build_info[i];
    // Création et lancement des solveurs locaux les options shiftées
    // Ceci afin de tester les deletes
    AlephSolver* s = new AlephSolver(traceMng(), subDomain(),
                                     sbi.m_number_of_resolution_per_solvers,
                                     sbi.m_underliying_solver,
                                     sbi.m_number_of_core,
                                     options()->deltaT);
    m_posted_solvers.add(s);
    s->launchResolutions(m_cell_temperature, m_face_temperature);
    // Lancement des solveurs globaux
    m_global_aleph_solver.at(i)->launchResolutions(m_cell_temperature, m_face_temperature);
  }

  // Si on a atteint notre maximum d'itérations, on quitte
  if (subDomain()->commonVariables().globalIteration() >= options()->iterations)
    subDomain()->timeLoopMng()->stopComputeLoop(true);
}

// ****************************************************************************
// * REGISTER + NAMESPACE
// ****************************************************************************
ARCANE_DEFINE_STANDARD_MODULE(AlephMultiTest, AlephMultiTest);
ARCANETEST_END_NAMESPACE
