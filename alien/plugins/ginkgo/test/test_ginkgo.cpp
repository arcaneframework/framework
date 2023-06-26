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

#include <alien/ginkgo/backend.h>
#include <alien/ginkgo/options.h>

#include <alien/ref/AlienRefSemantic.h>
#include <arccore/message_passing_mpi/StandaloneMpiMessagePassingMng.h>
#include <alien/kernels/simple_csr/algebra/SimpleCSRLinearAlgebra.h>

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

  /***
	 * Matrice A, diagonale
	 * 2 -1 
	 * -1 2 -1 
	 *   -1  2 -1
	 *      -1  2 -1        
	 ***/

  Alien::Matrix A(size, size, pm);

  // Distribution
  const auto& dist = A.distribution();
  int offset = dist.rowOffset();
  int lsize = dist.localRowSize();
  int gsize = dist.globalRowSize();

  /* seq : 0, 100, 100 */
  /* 2 mpi :
     * - 0/50/100
     * - 50/50/100
     */

  // Remplissage avec builder
  tm->info() << "build matrix with direct matrix builder";
  {
    Alien::DirectMatrixBuilder builder(A, Alien::DirectMatrixOptions::eResetValues); // par defaut, la matrice est symétrique ? et stockage CSR et distribution par lignes
    builder.reserve(3); // Réservation de 3 coefficients par ligne
    builder.allocate(); // Allocation de l'espace mémoire réservé

    for (int irow = offset; irow < offset + lsize; ++irow) {
      builder(irow, irow) = 2.;
      if (irow - 1 >= 0)
        builder(irow, irow - 1) = -1.;
      if (irow + 1 < gsize)
        builder(irow, irow + 1) = -1.;
    }
  }

  /**
	 *  Vecteur xe (ones)
	 ********************************************/

  tm->info() << "* xe = 1";
  Alien::Vector xe = Alien::ones(size, pm);
  tm->info() << "=> Vector Distribution : " << xe.distribution();

  /**
	 *  Vecteur b = A * xe 
	 ********************************************/
  tm->info() << "* b = A * xe";
  Alien::Vector b(size, pm);
  //Alien::SimpleCSRLinearAlgebra algebra;

  Alien::Ginkgo::LinearAlgebra algebra;
  algebra.mult(A, xe, b);

  /**
	 *  Calcul x, tq : Ax = b 
	 ********************************************/
  tm->info() << "* Calcul de x, tel que  :  A x = b";

  Alien::Vector x(size, pm);

  Alien::Ginkgo::Options options;
  options.numIterationsMax(100);
  options.stopCriteriaValue(1e-10);
  options.preconditioner(Alien::Ginkgo::OptionTypes::Jacobi);
  options.solver(Alien::Ginkgo::OptionTypes::CG);
  auto solver = Alien::Ginkgo::LinearSolver(options);
  solver.solve(A, b, x);

  /**
	 *  Calcul du résidu ||Ax - b|| ~ 0
	 ********************************************/
  tm->info() << "* r = Ax - b";

  Alien::Vector r(size, pm);
  Alien::Vector tmp(size, pm);

  tm->info() << "t = Ax";
  algebra.mult(A, x, tmp);

  tm->info() << "r = t";
  algebra.copy(tmp, r);

  tm->info() << "r -= b"; // r = r + (-1 * b)
  algebra.axpy(-1., b, r);

  auto norm = algebra.norm2(r);
  tm->info() << " => ||r|| = " << norm;

  /**
	 *  Calcul de ||x -xe|| ~ 0
	 ********************************************/

  tm->info() << "* r = || x - xe ||";

  tm->info() << "r = x";
  algebra.copy(x, r);
  tm->info() << "r -= xe";
  algebra.axpy(-1., xe, r);
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
