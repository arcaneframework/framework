// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#include <alien/ref/AlienRefSemantic.h>
#include <alien/utils/Precomp.h>
#include <cmath>
#include <gtest/gtest.h>

#include <alien/ref/AlienImportExport.h>
//#include <alien/ref/mv_expr/MVExpr.h>

#include <alien/kernels/sycl/SYCLPrecomp.h>
#include <alien/kernels/sycl/data/SYCLParallelEngine.h>
#include <alien/kernels/sycl/data/SYCLBEllPackMatrix.h>
#include <alien/kernels/sycl/data/SYCLVector.h>

#include <alien/kernels/sycl/data/HCSRMatrix.h>
#include <alien/kernels/sycl/data/HCSRVector.h>

#include <alien/handlers/scalar/sycl/VectorAccessorT.h>
#include <alien/handlers/scalar/sycl/MatrixProfiler.h>
#include <alien/handlers/scalar/sycl/ProfiledMatrixBuilderT.h>

#include <alien/kernels/sycl/algebra/SYCLLinearAlgebra.h>
#include <alien/kernels/simple_csr/algebra/SimpleCSRLinearAlgebra.h>

#include <Environment.h>


#include <alien/kernels/sycl/data/SYCLParallelEngineImplT.h>
#include <alien/handlers/scalar/sycl/VectorAccessorImplT.h>
#include <alien/handlers/scalar/sycl/ProfiledMatrixBuilderImplT.h>

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


// Tests the default c'tor.
TEST(TestSYCLMV, HCSRVector)
{
  using namespace Alien;
  Alien::ITraceMng* trace_mng = AlienTest::Environment::traceMng();
  Integer global_size = 1050;
  const Alien::Space s(global_size, "MySpace");
  Alien::MatrixDistribution mdist(s, s, AlienTest::Environment::parallelMng());
  Alien::VectorDistribution vdist(s, AlienTest::Environment::parallelMng());
  Alien::Vector x(vdist);
  Alien::Vector y(vdist);
  std::size_t local_size = vdist.localSize();
  auto offset = vdist.offset();


  Alien::SYCLParallelEngine engine;
  {
    auto x_acc = Alien::SYCL::VectorAccessorT<Real>(x);
    engine.submit([&](Alien::SYCLControlGroupHandler& cgh)
                  {
                     auto xv = x_acc.view(cgh) ;
                     cgh.parallel_for(engine.maxNumThreads(),
                                         [=](Alien::SYCLParallelEngine::Item<1> item)
                                         {
                                            auto index = item.get_id(0) ;
                                            auto id = item.get_id(0);
                                            for (std::size_t index = id; id < local_size; id += item.get_range()[0])
                                               xv[index] = 1.*index;
                                         });

                  }) ;

    auto y_acc = Alien::SYCL::VectorAccessorT<Real>(y);
    engine.submit([&](Alien::SYCLControlGroupHandler& cgh)
                  {
                    auto yv = y_acc.view(cgh) ;
                    auto xcv = x_acc.constView(cgh) ;
                    cgh.parallel_for(engine.maxNumThreads(),
                                         [=](Alien::SYCLParallelEngine::Item<1> item)
                                         {
                                            auto index = item.get_id(0) ;
                                            auto id = item.get_id(0);
                                            for (std::size_t index = id; id < local_size; id += item.get_range()[0])
                                              yv[index] = 2*xcv[index] ;
                                         });
                  }) ;

    Real norme_x = 0. ;
    Real norme_y = 0. ;
    auto xhv = x_acc.hostView() ;
    auto yhv = y_acc.hostView() ;
    for (std::size_t i = 0; i < local_size; ++i)
    {
      norme_x +=  xhv[i]* xhv[i] ;
      norme_y +=  yhv[i]* yhv[i] ;
    }
    trace_mng->info() << "NORME2 X : "<<norme_x ;
    trace_mng->info() << "NORME2 Y : "<<norme_y ;
    ASSERT_EQ(385323925, norme_x);
    ASSERT_EQ(1541295700, norme_y);
  }
}


// Tests the default c'tor.
TEST(TestSYCLMV, HCSRMatrix)
{
  using namespace Alien;
  Alien::ITraceMng* trace_mng = AlienTest::Environment::traceMng();
  Integer global_size = 1050;
  const Alien::Space s(global_size, "MySpace");
  Alien::MatrixDistribution mdist(s, s, AlienTest::Environment::parallelMng());
  Alien::VectorDistribution vdist(s, AlienTest::Environment::parallelMng());
  Alien::Matrix A(mdist); // A.setName("A") ;

  std::size_t local_size = vdist.localSize();
  auto offset = vdist.offset();

  Alien::SYCLParallelEngine engine;
  {
    Alien::SYCL::MatrixProfiler profiler(A);
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
    Alien::SYCL::ProfiledMatrixBuilder builder(A, Alien::ProfiledMatrixOptions::eResetValues);
    engine.submit([&](Alien::SYCLControlGroupHandler& cgh)
                  {
                    auto matrix_acc = builder.view(cgh) ;
                    cgh.parallel_for(engine.maxNumThreads(),
                                         [=](Alien::SYCLParallelEngine::Item<1> item)
                                         {
                                            auto index = item.get_id(0) ;
                                            auto id = item.get_id(0);
                                            for (auto index = id; id < local_size; id += item.get_range()[0])
                                            {
                                              Integer row = offset + index;
                                              matrix_acc[matrix_acc.entryIndex(row,row)] = 2.;
                                              if (row + 1 < global_size)
                                                matrix_acc[matrix_acc.entryIndex(row, row + 1)] = -1.;
                                              if (row - 1 >= 0)
                                                matrix_acc[matrix_acc.entryIndex(row, row - 1)] = -1.;
                                            }
                                         });
                  }) ;

    {
      Real norme_A = 0. ;
      auto hview = builder.hostView();
      for(std::size_t irow=0;irow<local_size;++irow)
      {
          for(auto k=hview.kcol(irow);k<hview.kcol(irow+1);++k)
          {
            norme_A += hview[k]*hview[k] ;
          }
      }
      trace_mng->info() << "NORME2 A : "<<norme_A ;
      ASSERT_EQ(6298, norme_A);
    }
  }
}

TEST(TestSYCLMV, HCSR2SYCLConverter)
{
  using namespace Alien;
  Alien::ITraceMng* trace_mng = AlienTest::Environment::traceMng();
  Integer global_size = 1050;
  const Alien::Space s(global_size, "MySpace");
  Alien::MatrixDistribution mdist(s, s, AlienTest::Environment::parallelMng());
  Alien::VectorDistribution vdist(s, AlienTest::Environment::parallelMng());
  Alien::Matrix A(mdist);
  Alien::Vector x(vdist);
  Alien::Vector y(vdist);

  auto local_size = vdist.localSize();
  auto offset = vdist.offset();

  Alien::SYCLParallelEngine engine;
  {
    Alien::SYCL::MatrixProfiler profiler(A);
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
    Alien::SYCL::ProfiledMatrixBuilder builder(A, Alien::ProfiledMatrixOptions::eResetValues);
    engine.submit([&](Alien::SYCLControlGroupHandler& cgh)
                  {
                    auto matrix_acc = builder.view(cgh) ;
                    cgh.parallel_for(engine.maxNumThreads(),
                                         [=](Alien::SYCLParallelEngine::Item<1> item)
                                         {
                                            auto index = item.get_id(0) ;
                                            auto id = item.get_id(0);
                                            for (auto index = id; id < local_size; id += item.get_range()[0])
                                            {
                                              Integer row = offset + index;
                                              matrix_acc[matrix_acc.entryIndex(row,row)] = 2.;
                                              if (row + 1 < global_size)
                                                matrix_acc[matrix_acc.entryIndex(row, row + 1)] = -1.;
                                              if (row - 1 >= 0)
                                                matrix_acc[matrix_acc.entryIndex(row, row - 1)] = -1.;
                                            }
                                         });
                  }) ;
  }
  {
      auto x_acc = Alien::SYCL::VectorAccessorT<Real>(x);
      engine.submit([&](Alien::SYCLControlGroupHandler& cgh)
                    {
                       auto xv = x_acc.view(cgh) ;
                       cgh.parallel_for(engine.maxNumThreads(),
                                           [=](Alien::SYCLParallelEngine::Item<1> item)
                                           {
                                              auto index = item.get_id(0) ;
                                              auto id = item.get_id(0);
                                              for (auto index = id; id < local_size; id += item.get_range()[0])
                                                 xv[index] = 1.*index;
                                           });

                    }) ;
  }

  {
    Alien::SYCLLinearAlgebra sycl_alg;

    trace_mng->info() << "TEST SPMV : y = A*x ";
    const auto& ma = A.impl()->get<Alien::BackEnd::tag::sycl>();

    const auto& vx = x.impl()->get<Alien::BackEnd::tag::sycl>();
    auto& vy = y.impl()->get<Alien::BackEnd::tag::sycl>(true);

    sycl_alg.mult(A, x, y);
    {
      Real norme_y = 0. ;
      Alien::LocalVectorReader reader(y);
      for (Integer i = 0; i < local_size; ++i) {
        norme_y += reader[i]*reader[i] ;
      }
      trace_mng->info() << "NORME2 Y=A*X : "<<norme_y ;
      ASSERT_EQ(1102501, norme_y);
     }
  }
}
