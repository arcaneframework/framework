/*
 * Copyright 2020 IFPEN-CEA
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

#include <alien/benchmark/ILinearProblem.h>
#include <alien/benchmark/LinearBench.h>

#include <alien/hypre/backend.h>
#include <alien/hypre/options.h>
#include "alien/expression/solver/SolverStat.h"

constexpr int NB_RUNS = 5;

int test(const Alien::Hypre::OptionTypes::eSolver& solv, const Alien::Hypre::OptionTypes::ePreconditioner& prec, const std::string& mat_filename, const std::string& vec_filename)
{
  auto* pm = Arccore::MessagePassing::Mpi::StandaloneMpiMessagePassingMng::create(MPI_COMM_WORLD);
  auto* tm = Arccore::arccoreCreateDefaultTraceMng();
  Alien::setTraceMng(tm);

  auto problem = Alien::Benchmark::buildFromMatrixMarket(pm, mat_filename, vec_filename);

  auto bench = Alien::Benchmark::LinearBench(std::move(problem));

  auto scattered = true;
  if (std::getenv("ALIEN_REDUCE_COMPACT")) {
    scattered = false;
  }

  for (int i = 0; i < NB_RUNS; i++) {
    Alien::Hypre::Options options;
    options.numIterationsMax(10000);
    options.stopCriteriaValue(1e-8);
    options.preconditioner(prec); // Jacobi, NoPC
    options.solver(solv); //CG, GMRES, BICG, BICGSTAB
    auto solver = Alien::Hypre::LinearSolver(options);

    tm->info() << "Running Hypre";
    auto solution = bench.solve(&solver);
    auto commsize = pm->commSize();
    for (auto mask = 1; commsize >> mask; mask++) {
      // Run Hypre from a reduced communicator.
      Arccore::MessagePassing::IMessagePassingMng* hypre_pm = nullptr;
      if (scattered) {
        // Scattered version
        hypre_pm = mpSplit(pm, !(pm->commRank() % (1 << mask)));
      }
      else {
        // Compact version
        hypre_pm = mpSplit(pm, pm->commRank() < commsize >> mask);
      }
      if (hypre_pm && hypre_pm->commRank() == 0) {
        tm->info() << "Running Hypre on a reduced communicator of size = " << hypre_pm->commSize() << " scattered : " << scattered;
      }
      auto solution = bench.solveWithRedistribution(&solver, hypre_pm);
      delete hypre_pm;
    }
  }

  return 0;
}

int main(int argc, char** argv)
{

  MPI_Init(&argc, &argv);

  if (argc != 5 && argc != 1) {
    std::cerr << "Usage : ./bench_Hypre [solver] [preconditioner] [matrix] [vector] \n"
              << "  - solver : (CG|GMRES|BICG|BICGSTAB) \n"
              << "  - preconditioner : (Jacobi|NoPC) \n"
              << "  - MTX matrix file \n"
              << "  - optional MTX vector file \n";
    return -1;
  }

  // Default options, to run as test
  auto solver = Alien::Hypre::OptionTypes::GMRES;
  auto prec = Alien::Hypre::OptionTypes::DiagPC;
  std::string matrix_file = "matrix.mtx";
  std::string vec_file = "rhs.mtx";

  if (argc == 1) {
    // do nothing
  }
  else {
    // Read the solver
    if (std::string(argv[1]) == "CG") {
      solver = Alien::Hypre::OptionTypes::CG;
    }
    else if (std::string(argv[1]) == "GMRES") {
      solver = Alien::Hypre::OptionTypes::GMRES;
    }
    else {
      std::cerr << "Unrecognized solver : " << argv[1] << "\n"
                << "  - solver list : (CG|GMRES|BICG|BICGSTAB) \n";
      return -1;
    }

    if (std::string(argv[2]) == "Jacobi") {
      prec = Alien::Hypre::OptionTypes::DiagPC;
    }
    else if (std::string(argv[2]) == "NoPC") {
      prec = Alien::Hypre::OptionTypes::NoPC;
    }
    else if (std::string(argv[2]) == "AMG") {
      prec = Alien::Hypre::OptionTypes::AMGPC;
    }
    else {
      std::cerr << "Unrecognized preconditioner : " << argv[2] << "\n"
                << "  - preconditioner list : (Jacobi|NoPC) \n";
      return -1;
    }

    matrix_file = std::string(argv[3]);
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