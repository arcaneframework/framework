/*
 * Copyright 2021 IFPEN-CEA
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
 *  SPDX-License-Identifier: Apache-2.0
 */

#include <arccore/message_passing_mpi/StandaloneMpiMessagePassingMng.h>

#include <alien/move/AlienMoveSemantic.h>
#include <alien/move/handlers/scalar/VectorWriter.h>

#include <alien/hypre/backend.h>
#include <alien/hypre/options.h>

int test(const std::string& filename)
{
  auto* pm = Arccore::MessagePassing::Mpi::StandaloneMpiMessagePassingMng::create(MPI_COMM_WORLD);
  auto* tm = Arccore::arccoreCreateDefaultTraceMng();

  Alien::setTraceMng(tm);
  Alien::setVerbosityLevel(Alien::Verbosity::Debug);

  auto A = Alien::Move::readFromMatrixMarket(pm, filename);

  tm->info() << "* xe = 1";

  auto xe = Alien::Move::VectorData(A.distribution().colDistribution());
  {
    Alien::Move::LocalVectorWriter v_build(std::move(xe));
    for (int i = 0; i < v_build.size(); i++) {
      v_build[i] = 1.0;
    }
    xe = v_build.release();
  }

  tm->info() << "=> Vector Distribution : " << xe.distribution();

  tm->info() << "* b = A * xe";

  Alien::Move::VectorData b(A.rowSpace(), A.distribution().rowDistribution());

  Alien::Hypre::LinearAlgebra algebra;

  algebra.mult(A, xe, b);

  Alien::Move::VectorData x(A.colSpace(), A.distribution().rowDistribution());

  tm->info() << "* x = A^-1 b";

  auto options = Alien::Hypre::Options()
                 .numIterationsMax(100)
                 .stopCriteriaValue(1e-10)
                 .preconditioner(Alien::Hypre::OptionTypes::AMGPC)
                 .solver(Alien::Hypre::OptionTypes::GMRES);

  auto solver = Alien::Hypre::LinearSolver(options);

  solver.solve(A, b, x);

  {
    tm->info() << "* r = Ax - b";

    Alien::Move::VectorData r(A.rowSpace(), A.distribution().rowDistribution());
    algebra.mult(A, x, r);
    tm->info() << "r -= b";
    algebra.axpy(-1., b, r);

    auto norm = algebra.norm2(r);
    auto norm_b = algebra.norm2(b);

    tm->info() << " => ||r|| = " << norm << " ; ||r||/||b|| = " << norm / norm_b;
  }

  {
    tm->info() << "|| x - xe ||";
    algebra.axpy(-1., xe, x);

    auto norm = algebra.norm2(x);
    auto norm_xe = algebra.norm2(xe);

    tm->info() << " => ||x-xe|| = " << norm << " ; ||r||/||b|| = " << norm / norm_xe;
  }
  tm->info() << " ";
  tm->info() << "... example finished !!!";

  return 0;
}

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);

  auto ret = 0;

  try {
    ret = test("mesh1em6.mtx");
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
