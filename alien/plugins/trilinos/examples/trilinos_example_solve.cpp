/*
 * Copyright 2022 IFPEN-CEA
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 * http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 * SPDX-License-Identifier: Apache-2.0
 */

#include <arccore/message_passing_mpi/StandaloneMpiMessagePassingMng.h>

#include <alien/move/AlienMoveSemantic.h>
#include <alien/move/handlers/scalar/VectorWriter.h>
#include <alien/move/data/VectorData.h>

#include <alien/trilinos/backend.h>
#include <alien/trilinos/options.h>

#include <alien/core/backend/IMatrixConverter.h>
#include <alien/kernels/simple_csr/SimpleCSRBackEnd.h>

int test(const Alien::Trilinos::OptionTypes::eSolver& solv, const Alien::Trilinos::OptionTypes::ePreconditioner& prec, const std::string& mat_filename, const std::string& vec_filename)
{
  auto* pm = Arccore::MessagePassing::Mpi::StandaloneMpiMessagePassingMng::create(MPI_COMM_WORLD);
  auto* tm = Arccore::arccoreCreateDefaultTraceMng();
  Alien::setTraceMng(tm);

  if (pm->commRank() == 0)
    tm->info() << "Read matrix file : " << mat_filename;
  auto A = Alien::Move::readFromMatrixMarket(pm, mat_filename);

  /**
	 *  Vecteur xe (ones)
	 *********************/
  auto xe = Alien::Move::VectorData(A.distribution().colDistribution());
  {
    Alien::Move::LocalVectorWriter v_build(std::move(xe));
    for (int i = 0; i < v_build.size(); i++) {
      v_build[i] = 1.0;
    }
    xe = v_build.release();
  }
  xe.distribution();

  /**
	 *  Vecteur b
	 *************/
  //Alien::SimpleCSRLinearAlgebra algebra;
  Alien::Trilinos::LinearAlgebra algebra;
  Alien::Move::VectorData b(A.rowSpace(), A.distribution().rowDistribution());

  if (vec_filename != "") {
    if (pm->commRank() == 0)
      tm->info() << "Read vector file : " << vec_filename;
    b = Alien::Move::readFromMatrixMarket(A.distribution().rowDistribution(), vec_filename);
  }
  else {
    if (pm->commRank() == 0)
      tm->info() << "Vector b is computed  : b = A * xe";
    algebra.mult(A, xe, b);
  }

  /**
	 *  Préparation du solveur pour le calcul de x, tq : Ax = b
	 ********************************************/
  Alien::Move::VectorData x(A.colSpace(), A.distribution().rowDistribution());
  Alien::Trilinos::Options options;
  options.numIterationsMax(500);
  options.stopCriteriaValue(1e-8);
  options.preconditioner(prec); // Jacobi, NoPC
  options.solver(solv); //CG, GMRES, BICG, BICGSTAB
  auto solver = Alien::Trilinos::LinearSolver(options);

  /**
	 *  BENCH
	 ********************************************/

  int nbRuns = 5;
  for (int i = 0; i < nbRuns; i++) {
    if (pm->commRank() == 0) {
      std::cout << "\n************************************************** " << std::endl;
      std::cout << "*                   RUN  # " << i << "                     * " << std::endl;
      std::cout << "************************************************** \n"
                << std::endl;
    }

    // init vector x with zeros
    Alien::Move::LocalVectorWriter writer(std::move(x));
    for (int i = 0; i < writer.size(); i++) {
      writer[i] = 0;
    }
    x = writer.release();

    // solve
    solver.solve(A, b, x);

    // compute explicit residual r = ||Ax - b|| ~ 0
    Alien::Move::VectorData r(A.rowSpace(), A.distribution().rowDistribution());
    algebra.mult(A, x, r);
    algebra.axpy(-1., b, r);
    auto norm_r = algebra.norm2(r);
    auto norm_b = algebra.norm2(b);

    if (pm->commRank() == 0) {
      std::cout << "||Ax-b|| = " << norm_r << std::endl;
      std::cout << "||b|| = " << norm_b << std::endl;
      std::cout << "||Ax-b||/||b|| = " << norm_r / norm_b << std::endl;
    }

    /* Check results :
     * min(x), max(x), min|x|, max|x|
     * err_max : ||Ax-b||_{inf}
     * rerr_max :||Ax-b||_{inf} / ||b|| _{inf}
     */

    /* std::cout << "max(x) : " << vecMax(x) << std::endl;
    std::cout << "min(x) : " << vecMin(x) << std::endl;
    std::cout << "maxAbs(x) : " << vecMaxAbs(x) << std::endl;
    std::cout << "minAbs(x) : " << vecMinAbs(x) << std::endl;
    std::cout << "max_error : " << vecMaxAbs(r) << std::endl;
    // std::cout << "absmaxB(b) : " << vecMaxAbs(b) << std::endl;
    std::cout << "rmax_error : " << vecMaxAbs(r) / vecMaxAbs(b) << std::endl;
    std::cout << "=================================== " << std::endl;*/
  }

  return 0;
}

int main(int argc, char** argv)
{

  MPI_Init(&argc, &argv);

  if (argc < 4) {
    std::cerr << "Usage : ./example_trilinos [solver] [preconditioner] [matrix] [vector] \n"
              << "  - solver : (Relaxation|*) \n"
              << "  - preconditioner : (Relaxation|NoPC) \n"
              << "  - MTX matrix file \n"
              << "  - optional MTX vector file \n";
    return -1;
  }

  // Read the solver
  Alien::Trilinos::OptionTypes::eSolver solver;
  if (std::string(argv[1]) == "CG") {
    solver = Alien::Trilinos::OptionTypes::CG;
  }
  else if (std::string(argv[1]) == "GMRES") {
    solver = Alien::Trilinos::OptionTypes::GMRES;
  }
  else if (std::string(argv[1]) == "BICGSTAB") {
    solver = Alien::Trilinos::OptionTypes::BICGSTAB;
  }
  else {
    std::cerr << "Unrecognized solver : " << argv[1] << "\n"
              << "  - solver list : (CG|GMRES|BICGSTAB) \n";
    return -1;
  }

  // Read the preconditioner
  Alien::Trilinos::OptionTypes::ePreconditioner prec;
  if (std::string(argv[2]) == "Relaxation") {
    prec = Alien::Trilinos::OptionTypes::Relaxation;
  }
  else if (std::string(argv[2]) == "NoPC") {
    prec = Alien::Trilinos::OptionTypes::NoPC;
  }
  else if (std::string(argv[2]) == "AMG") {
    prec = Alien::Trilinos::OptionTypes::MueLu;
  }
  else {
    std::cerr << "Unrecognized preconditioner : " << argv[2] << "\n"
              << "  - preconditioner list : (Relaxation|NoPC) \n";
    return -1;
  }

  // Read Matrix file
  std::string matrix_file;

  // Read matrix file
  if (argv[3]) {
    matrix_file = std::string(argv[3]);
  }
  else {
    std::cerr << "Matrix File is needed for this bench.";
    return -1;
  }

  // Read optional Vector file
  std::string vec_file = "";
  if (argv[4]) {
    vec_file = std::string(argv[4]);
  }

  auto ret = 0;
  try {
    ret = test(solver, prec, matrix_file, vec_file);
  }
  catch (const Arccore::Exception& ex) {
    std::cerr << "Exception: " << ex << '\n';
    ret = 3;
  }
  catch (const std::exception& ex) {
    std::cerr << "** A standard exception occured: " << ex.what() << ".\n";
    ret = 2;
  }
  catch (...) {
    std::cerr << "** An unknown exception has occured...\n";
    ret = 1;
  }

  MPI_Finalize();
  return ret;
}