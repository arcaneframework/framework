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

#include <alien/ginkgo/backend.h>
#include <alien/ginkgo/options.h>

constexpr int NB_RUNS = 5;

int test(const Alien::Ginkgo::OptionTypes::eSolver& solv, const Alien::Ginkgo::OptionTypes::ePreconditioner& prec, const std::string& mat_filename, const std::string& vec_filename, const int& block_size)
{
  auto* pm = Arccore::MessagePassing::Mpi::StandaloneMpiMessagePassingMng::create(MPI_COMM_WORLD);
  auto* tm = Arccore::arccoreCreateDefaultTraceMng();
  Alien::setTraceMng(tm);

  auto problem = Alien::Benchmark::buildFromMatrixMarket(pm, mat_filename, vec_filename);

  auto bench = Alien::Benchmark::LinearBench(std::move(problem));

  for (int i = 0; i < NB_RUNS; i++) {
    Alien::Ginkgo::Options options;
    options.numIterationsMax(500);
    options.stopCriteriaValue(1e-8);
    options.preconditioner(prec); // Jacobi, NoPC
    options.blockSize(block_size); // block size
    options.solver(solv); //CG, GMRES, BICG, BICGSTAB
    auto solver = Alien::Ginkgo::LinearSolver(options);

    if (pm->commSize() == 1) {
      tm->info() << "Running Ginkgo";
      auto solution = bench.solve(&solver);
    }
    else {
      tm->info() << "Running Ginkgo on a reduced communicator";
      // Run ginkgo from a sequential communicator.
      std::unique_ptr<Arccore::MessagePassing::IMessagePassingMng> ginkgo_pm(mpSplit(pm, pm->commRank() == 0));
      auto solution = bench.solveWithRedistribution(&solver, ginkgo_pm.get());
    }
  }

  return 0;
}

int main(int argc, char** argv)
{

  MPI_Init(&argc, &argv);

  if (argc != 1 && (argc > 6 || argc < 4)) {
    std::cerr << "Usage : ./bench_ginkgo [solver] [preconditioner] [matrix] [vector] \n"
              << "  - solver : (CG|GMRES|BICG|BICGSTAB) \n"
              << "  - preconditioner : (Jacobi|Ilu0|NoPC) \n"
              << "  - MTX matrix file \n"
              << "  - optional MTX vector file \n"
              << "  - optional block size \n";
    return -1;
  }

  // Default options, to run as test
  auto solver = Alien::Ginkgo::OptionTypes::GMRES;
  auto prec = Alien::Ginkgo::OptionTypes::Jacobi;
  std::string matrix_file = "matrix.mtx";
  std::string vec_file = "rhs.mtx";
  int block_size = 1;

  if (argc == 1) {
    // do nothing
  }
  else {
    // Read the solver
    if (std::string(argv[1]) == "CG") {
      solver = Alien::Ginkgo::OptionTypes::CG;
    }
    else if (std::string(argv[1]) == "GMRES") {
      solver = Alien::Ginkgo::OptionTypes::GMRES;
    }
    else if (std::string(argv[1]) == "BICG") {
      solver = Alien::Ginkgo::OptionTypes::BICG;
    }
    else if (std::string(argv[1]) == "BICGSTAB") {
      solver = Alien::Ginkgo::OptionTypes::BICGSTAB;
    }
    else {
      std::cerr << "Unrecognized solver : " << argv[1] << "\n"
                << "  - solver list : (CG|GMRES|BICG|BICGSTAB) \n";
      return -1;
    }

    // Read the preconditioner
    if (std::string(argv[2]) == "Jacobi") {
      prec = Alien::Ginkgo::OptionTypes::Jacobi;
    }
    else if (std::string(argv[2]) == "Ilu") {
      prec = Alien::Ginkgo::OptionTypes::Ilu;
    }
    else if (std::string(argv[2]) == "NoPC") {
      prec = Alien::Ginkgo::OptionTypes::NoPC;
    }
    else {
      std::cerr << "Unrecognized preconditioner : " << argv[2] << "\n"
                << "  - preconditioner list : (Jacobi|Ilu|NoPC) \n";
      return -1;
    }

    matrix_file = std::string(argv[3]);
    if (argc >= 5) {
      vec_file = std::string(argv[4]);
    }
    else {
      vec_file = "";
    }
    // Read the optional block size
    if (argc >= 6) {
      block_size = std::atoi(argv[5]);
    }
  }

  auto ret = 0;
  try {
    ret = test(solver, prec, matrix_file, vec_file, block_size);
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