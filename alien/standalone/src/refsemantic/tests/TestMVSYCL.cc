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

#include <alien/ref/AlienRefSemantic.h>
#include <alien/utils/Precomp.h>
#include <cmath>
#include <gtest/gtest.h>

#include <alien/ref/AlienImportExport.h>
//#include <alien/ref/mv_expr/MVExpr.h>

#include <alien/kernels/sycl/SYCLPrecomp.h>
#include <alien/kernels/sycl/data/SYCLBEllPackMatrix.h>
#include <alien/kernels/sycl/data/SYCLVector.h>

#include <alien/kernels/sycl/algebra/SYCLLinearAlgebra.h>
#include <alien/kernels/simple_csr/algebra/SimpleCSRLinearAlgebra.h>

#include <Environment.h>

// Tests the default c'tor.
TEST(TestSYCLMV, SYCLExpr)
{
  using namespace Alien;
  Alien::ITraceMng* trace_mng = AlienTest::Environment::traceMng();
  Integer global_size = 1050;
  const Alien::Space s(global_size, "MySpace");
  Alien::MatrixDistribution mdist(s, s, AlienTest::Environment::parallelMng());
  Alien::VectorDistribution vdist(s, AlienTest::Environment::parallelMng());
  Alien::Matrix A(mdist); // A.setName("A") ;
  Alien::Vector x(vdist); // x.setName("x") ;
  Alien::Vector y(vdist); // y.setName("y") ;
  Alien::Vector r(vdist); // r.setName("r") ;
  //Alien::Real lambda = 0.5;

  auto local_size = vdist.localSize();
  auto offset = vdist.offset();
  {
    Alien::MatrixProfiler profiler(A);
    for (Integer i = 0; i < local_size; ++i) {
      Integer row = offset + i;
      profiler.addMatrixEntry(row, row);
      if (row + 1 < global_size)
        profiler.addMatrixEntry(row, row + 1);
      if (row - 1 >= 0)
        profiler.addMatrixEntry(row, row - 1);
    }
  }
  {
    Alien::ProfiledMatrixBuilder builder(A, Alien::ProfiledMatrixOptions::eResetValues);
    for (Integer i = 0; i < local_size; ++i) {
      Integer row = offset + i;
      builder(row, row) = 2.;
      if (row + 1 < global_size)
        builder(row, row + 1) = -1.;
      if (row - 1 >= 0)
        builder(row, row - 1) = -1.;
    }
  }
  {
    Alien::LocalVectorWriter writer(x);
    for (Integer i = 0; i < local_size; ++i)
      writer[i] = 1.;
  }
  {
    Alien::LocalVectorWriter writer(y);
    for (Integer i = 0; i < local_size; ++i)
      writer[i] = i;
  }

  {
    Alien::LocalVectorWriter writer(r);
    for (Integer i = 0; i < local_size; ++i)
      writer[i] = 0.;
  }

  Alien::SimpleCSRLinearAlgebra alg;
  trace_mng->info() << " NORME X : " << alg.norm2(x);
  trace_mng->info() << " NORME Y : " << alg.norm2(y);
  trace_mng->info() << " NORME R : " << alg.norm2(r);

  Alien::SYCLLinearAlgebra sycl_alg;
  {
    trace_mng->info() << "TEST COPY : r = y";
    sycl_alg.copy(y, r);
    {
      Alien::LocalVectorReader reader(r);
      for (Integer i = 0; i < std::min(10, local_size); ++i) {
        trace_mng->info() << "R[" << i << "]=" << reader[i];
      }
    }
  }

  {
    trace_mng->info() << "TEST AXPY : y += a*x ";
    sycl_alg.axpy(1., x, y);

    {
      Alien::LocalVectorReader reader(y);
      for (Integer i = 0; i < std::min(10, local_size); ++i) {
        trace_mng->info() << "Y[" << i << "]=" << reader[i];
      }
    }
  }

  {
    trace_mng->info() << "TEST DOT : dot(x,y) ";
    Real x_dot_y_ref = 0.;
    {
      Alien::LocalVectorReader reader_x(x);
      Alien::LocalVectorReader reader_y(y);
      for (Integer i = 0; i < local_size; ++i)
        x_dot_y_ref += reader_x[i] * reader_y[i];
    }

    Real x_dot_y = sycl_alg.dot(x, y);
    trace_mng->info() << "SYCL DOT(X,Y) = " << x_dot_y << " REF=" << x_dot_y_ref;
  }

  {
    trace_mng->info() << "TEST SPMV : y = A*x ";
    const auto& ma = A.impl()->get<Alien::BackEnd::tag::sycl>();

    const auto& vx = x.impl()->get<Alien::BackEnd::tag::sycl>();
    auto& vy = y.impl()->get<Alien::BackEnd::tag::sycl>(true);

    sycl_alg.mult(A, x, y);
    {
      Alien::LocalVectorReader reader(y);
      for (Integer i = 0; i < std::min(10, local_size); ++i) {
        trace_mng->info() << "Y[" << i << "]=" << reader[i];
      }
    }
  }
}
