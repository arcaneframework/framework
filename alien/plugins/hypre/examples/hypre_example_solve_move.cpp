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

#include <alien/hypre/backend.h>
#include <alien/hypre/options.h>

int test()
{
  auto* pm = Arccore::MessagePassing::Mpi::StandaloneMpiMessagePassingMng::create(MPI_COMM_WORLD);
  auto* tm = Arccore::arccoreCreateDefaultTraceMng();

  Alien::setTraceMng(tm);
  Alien::setVerbosityLevel(Alien::Verbosity::Debug);

  auto size = 100;

  tm->info() << "Example Alien :";
  tm->info() << "Use of scalar builder (RefSemantic API) for Laplacian problem";
  tm->info() << " => solving linear system Ax = b";
  tm->info() << " * problem size = " << size;
  tm->info() << " ";
  tm->info() << "Start example...";
  tm->info() << " ";

  auto distribution = Alien::MatrixDistribution(size, size, pm);

  Alien::Move::MatrixData A(distribution);

  // Distributions calculée
  const auto& dist = A.distribution();
  int offset = dist.rowOffset();
  int lsize = dist.localRowSize();
  int gsize = dist.globalRowSize();

  tm->info() << "build matrix with direct matrix builder";
  {
    Alien::Move::DirectMatrixBuilder builder(std::move(A), Alien::DirectMatrixOptions::eResetValues);
    builder.reserve(3); // Réservation de 3 coefficients par ligne
    builder.allocate(); // Allocation de l'espace mémoire réservé

    for (int irow = offset; irow < offset + lsize; ++irow) {
      builder(irow, irow) = 2.;
      if (irow - 1 >= 0)
        builder(irow, irow - 1) = -1.;
      if (irow + 1 < gsize)
        builder(irow, irow + 1) = -1.;
    }
    A = builder.release();
  }

  tm->info() << "* xe = 1";

  auto xe = Alien::Move::VectorData(distribution.colDistribution());
  {
    Alien::Move::LocalVectorWriter v_build(std::move(xe));
    for (int i = 0; i < v_build.size(); i++) {
      v_build[i] = 1.0;
    }
    xe = v_build.release();
  }

  tm->info() << "=> Vector Distribution : " << xe.distribution();

  tm->info() << "* b = A * xe";

  Alien::Move::VectorData b(distribution.rowDistribution());

  Alien::Hypre::LinearAlgebra algebra;

  algebra.mult(A, xe, b);

  Alien::Move::VectorData x(distribution.colDistribution());

  tm->info() << "* x = A^-1 b";

  auto options = Alien::Hypre::Options()
                 .numIterationsMax(100)
                 .stopCriteriaValue(1e-10)
                 .preconditioner(Alien::Hypre::OptionTypes::AMGPC)
                 .solver(Alien::Hypre::OptionTypes::GMRES);

  auto solver = Alien::Hypre::LinearSolver(options);

  solver.solve(A, b, x);

  tm->info() << "* r = Ax - b";

  Alien::Move::VectorData r(distribution.colDistribution());

  {
    Alien::Move::VectorData tmp(distribution.colDistribution());
    tm->info() << "t = Ax";
    algebra.mult(A, x, tmp);
    tm->info() << "r = t";
    algebra.copy(tmp, r);
    tm->info() << "r -= b";
    algebra.axpy(-1., b, r);
  }

  auto norm = algebra.norm2(r);

  tm->info() << " => ||r|| = " << norm;

  tm->info() << "* r = || x - xe ||";

  {
    tm->info() << "r = x";
    algebra.copy(x, r);
    tm->info() << "r -= xe";
    algebra.axpy(-1., xe, r);
  }

  tm->info() << " => ||r|| = " << norm;

  tm->info() << " ";
  tm->info() << "... example finished !!!";

  return 0;
}

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);

  auto ret = 0;

  try {
    ret = test();
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
