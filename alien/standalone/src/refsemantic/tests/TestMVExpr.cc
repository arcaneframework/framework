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
#include <alien/ref/mv_expr/MVExpr.h>

#include <Environment.h>

// Tests the default c'tor.
TEST(TestMVExpr, ExprEngine)
{
  using namespace Alien;
  Alien::ITraceMng* trace_mng = AlienTest::Environment::traceMng();
  Integer global_size = 10;
  const Alien::Space s(10, "MySpace");
  Alien::MatrixDistribution mdist(s, s, AlienTest::Environment::parallelMng());
  Alien::VectorDistribution vdist(s, AlienTest::Environment::parallelMng());
  Alien::Matrix A(mdist); // A.setName("A") ;
  Alien::Matrix B(mdist); // B.setName("B") ;
  Alien::Matrix C(mdist); // C.setName("C") ;
  Alien::Vector x(vdist); // x.setName("x") ;
  Alien::Vector y(vdist); // y.setName("y") ;
  Alien::Vector r(vdist); // r.setName("r") ;
  Alien::Real lambda = 0.5;

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
    SystemWriter writer("MatrixA.txt");
    writer.dump(A);
  }
  {
    Alien::LocalVectorWriter writer(x);
    for (Integer i = 0; i < local_size; ++i)
      writer[i] = 1.;
  }

  using namespace Alien::MVExpr;

  {
    auto expr = mul(ref(A), ref(x));
    auto evaluated = eval(expr);
  }

  {
    auto expr = add(ref(x), ref(y));
    auto evaluated = eval(expr);
  }

  {
    auto expr = mul(ref(A), add(ref(x), ref(y)));
    auto evaluated = eval(expr);
  }

  {
    // auto expr = mul(ref(A), add(ref(x),mul(cst(lambda),ref(y)))) ;
    // auto evaluated = eval(expr);
  }

  Alien::SimpleCSRLinearAlgebra alg;
  Alien::SimpleCSRLinearAlgebraExpr alg_expr;
  // trace_mng->info()<<" NORME : "<<x.name()<<" = "<<alg.norm2(x) ;
  ASSERT_EQ(alg.norm2(x), std::sqrt(global_size));
  {
    trace_mng->info() << "y = lambda*x";
    assign(y, lambda * x);
    y = lambda * x;
    // trace_mng->info()<<" NORME : "<<y.name()<<" = "<<alg.norm2(y)  ;
    ASSERT_EQ(alg.norm2(y), std::sqrt(global_size * 0.25));

    trace_mng->info() << "y=A*x";
    assign(y, mul(ref(A), ref(x)));
    y = A * x;
    {
      Alien::LocalVectorReader view(y);
      for (Integer i = 0; i < local_size; ++i)
        trace_mng->info() << "Y[" << i << "]=" << view[i];
    }

    trace_mng->info() << " NORME : " << alg.norm2(y);
    ASSERT_EQ(alg.norm2(y), std::sqrt(2));

    trace_mng->info() << "r=y-A*x";
    r = y - A * x;
    // trace_mng->info()<<" NORME : "<<r.name()<<" = "<<alg.norm2(r)  ;
    ASSERT_EQ(alg.norm2(r), 0.);

    trace_mng->info() << "x=x+lambda*r";
    y = y + (lambda * x);
    // trace_mng->info()<<" NORME : "<<y.name()<<" = "<<alg.norm2(y)  ;
    ASSERT_EQ(alg.norm2(y), std::sqrt(2 * 1.5 * 1.5 + (global_size - 2) * 0.25));

    trace_mng->info() << "y=A*lambda*r";
    y = A * (lambda * x);
    // trace_mng->info()<<" NORME : "<<y.name()<<" = "<<alg.norm2(y)  ;
    ASSERT_EQ(alg.norm2(y), std::sqrt(0.5));

    {
      Alien::VectorWriter writer(y);
      for (Integer i = 0; i < local_size; ++i)
        writer[offset + i] = offset + i;
    }

    Real value1 = eval(dot(x, y));
    trace_mng->info() << " DOT(x,y) " << value1;
    ASSERT_EQ(value1, 45);

    Real value2 = eval(dot(lambda * x, y));
    trace_mng->info() << " DOT(lambda*x,y) " << value2;
    ASSERT_EQ(value2, 22.5);

    Real value3 = eval(dot(x, r + y));
    trace_mng->info() << " DOT(x,r+y) " << value3;
    ASSERT_EQ(value3, 45);

    Real value4 = eval(dot(x, A * y));
    trace_mng->info() << " DOT(x,A*y) " << value4;
    ASSERT_EQ(value4, 9);

    trace_mng->info() << "B=lambda*A";
    B = lambda * A;
    Real A_norme2 = alg_expr.norm2(A);
    Real B_norme2 = alg_expr.norm2(B);
    ASSERT_EQ(B_norme2, lambda * A_norme2);

    trace_mng->info() << "C=A+B";
    C = A + B;
    Real C_norme2 = alg_expr.norm2(C);
    ASSERT_EQ(C_norme2, (lambda + 1) * A_norme2);
  }

  ////////////////////////////////////////////////
  //
  // TESTS PIPELINE ENABLING ASYNCHRONISM
  {
    pipeline(vassign(y, A * x), vassign(r, y - A * x));
  }
}
