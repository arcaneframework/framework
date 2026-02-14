// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#include <boost/lexical_cast.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/cmdline.hpp>
#include <boost/program_options/variables_map.hpp>

#include <arccore/message_passing_mpi/StandaloneMpiMessagePassingMng.h>
#include <arccore/base/StringBuilder.h>

#include <alien/kernels/simple_csr/algebra/SimpleCSRLinearAlgebra.h>

#include <alien/kernels/sycl/SYCLPrecomp.h>
#include "alien/kernels/sycl/data/SYCLEnv.h"
#include "alien/kernels/sycl/data/SYCLEnvInternal.h"

#include <alien/kernels/sycl/data/SYCLBEllPackMatrix.h>
#include <alien/kernels/sycl/data/SYCLVector.h>
#include <alien/kernels/sycl/algebra/SYCLLinearAlgebra.h>

#include "alien/kernels/sycl/data/SYCLVectorInternal.h"
#include <alien/kernels/sycl/data/SYCLBEllPackInternal.h>
#include <alien/kernels/sycl/algebra/SYCLInternalLinearAlgebra.h>

#include "alien/kernels/sycl/data/SYCLEnv.h"
#include "alien/kernels/sycl/data/SYCLEnvInternal.h"
#include <alien/kernels/sycl/algebra/SYCLKernelInternal.h>

#include <alien/kernels/sycl/data/SYCLParallelEngine.h>

#include <alien/kernels/sycl/data/HCSRMatrix.h>
#include <alien/kernels/sycl/data/HCSRVector.h>

#include <alien/handlers/scalar/sycl/VectorAccessorT.h>
#include <alien/handlers/scalar/sycl/MatrixProfiler.h>
#include <alien/handlers/scalar/sycl/ProfiledMatrixBuilderT.h>

#include <alien/kernels/sycl/data/SYCLParallelEngineImplT.h>
#include <alien/handlers/scalar/sycl/VectorAccessorImplT.h>
#include <alien/handlers/scalar/sycl/ProfiledMatrixBuilderImplT.h>

#include <alien/kernels/sycl/algebra/SYCLLinearAlgebra.h>
#include <alien/kernels/simple_csr/algebra/SimpleCSRLinearAlgebra.h>

#include <alien/ref/AlienRefSemantic.h>

#include <alien/utils/StdTimer.h>

int main(int argc, char** argv)
{

  using namespace boost::program_options;
  options_description desc;
  // clang-format off
  desc.add_options()
      ("help", "produce help")
      ("size",    value<int>()->default_value(16),            "size") ;
  // clang-format on

  variables_map vm;
  store(parse_command_line(argc, argv, desc), vm);
  notify(vm);

  if (vm.count("help")) {
    std::cout << desc << "\n";
    return 1;
  }

  MPI_Init(&argc, &argv);
  auto* pm = Arccore::MessagePassing::Mpi::StandaloneMpiMessagePassingMng::create(MPI_COMM_WORLD);
  bool is_parallel = pm->commSize() > 1;

  auto* trace_mng = Arccore::arccoreCreateDefaultTraceMng();
  Alien::Integer my_rank = pm->commRank();
  Arccore::StringBuilder filename("sycl.log");
  Arccore::ReferenceCounter<Arccore::ITraceStream> ofile;
  if (pm->commSize() > 1) {
    filename += pm->commRank();
    ofile = Arccore::ITraceStream::createFileStream(filename.toString());
    trace_mng->setRedirectStream(ofile.get());
  }
  trace_mng->finishInitialize();

  Alien::setTraceMng(trace_mng);
  Alien::setVerbosityLevel(Alien::Verbosity::Debug);

  trace_mng->info() << "INFO START SYCL TEST";

  // clang-format off
  typedef Alien::StdTimer   TimerType;
  typedef TimerType::Sentry SentryType;
  // clang-format on

  TimerType timer;

  using namespace Alien;
  //Alien::ITraceMng* trace_mng = AlienTest::Environment::traceMng();

  // clang-format off
  Integer global_size = vm["size"].as<int>() ;
  // clang-format on

  const Alien::Space s(global_size, "MySpace");
  Alien::MatrixDistribution mdist(s, s, pm);
  Alien::VectorDistribution vdist(s, pm);
  Alien::Matrix A(mdist); // A.setName("A") ;
  Alien::Vector x(vdist); // x.setName("x") ;
  Alien::Vector y(vdist); // y.setName("y") ;
  Alien::Vector r(vdist); // r.setName("r") ;
  //Alien::Real lambda = 0.5;

  auto local_size = vdist.localSize();
  auto offset = vdist.offset();


  Alien::SYCLParallelEngine engine;
  {
    auto x_acc = Alien::SYCL::VectorAccessorT<Real>(x);
    engine.submit([&](Alien::SYCLControlGroupHandler& cgh)
                  {
                     auto xv = x_acc.view(cgh) ;
                     cgh.parallel_for(engine.maxNumThreads(),
                                         [=](Alien::SYCLParallelEngine::Item<1>::type item)
                                         {
                                            auto id = item.get_id(0);
                                            for (std::size_t index = id; id < std::size_t(local_size); id += item.get_range()[0])
                                               xv[index] = 1.*index;
                                         });

                  }) ;

    auto y_acc = Alien::SYCL::VectorAccessorT<Real>(y);
    engine.submit([&](Alien::SYCLControlGroupHandler& cgh)
                  {
                    auto yv = y_acc.view(cgh) ;
                    auto xcv = x_acc.constView(cgh) ;
                    cgh.parallel_for(engine.maxNumThreads(),
                                         [=](Alien::SYCLParallelEngine::Item<1>::type item)
                                         {
                                            auto index = item.get_id(0) ;
                                            auto id = item.get_id(0);
                                            for (std::size_t index = id; id < std::size_t(local_size); id += item.get_range()[0])
                                              yv[index] = 2*xcv[index] ;
                                         });
                  }) ;

    Real norme_x = 0. ;
    Real norme_y = 0. ;
    auto xhv = x_acc.hostView() ;
    auto yhv = y_acc.hostView() ;
    for (Alien::Integer i = 0; i < local_size; ++i)
    {
      norme_x +=  xhv[i]* xhv[i] ;
      norme_y +=  yhv[i]* yhv[i] ;
    }
    trace_mng->info() << "NORME2 X : "<<norme_x ;
    trace_mng->info() << "NORME2 Y : "<<norme_y ;
  }

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
                                         [=](Alien::SYCLParallelEngine::Item<1>::type item)
                                         {
                                            auto index = item.get_id(0) ;
                                            auto id = item.get_id(0);
                                            for (std::size_t index = id; id < std::size_t(local_size); id += item.get_range()[0])
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

    if(local_size<20)
    {
      Real norme_A = 0. ;
      auto hview = builder.hostView();
      for(Alien::Integer irow=0;irow<local_size;++irow)
      {
          trace_mng->info() <<" ROW ["<<irow<<"]:";
          for(auto k=hview.kcol(irow);k<hview.kcol(irow+1);++k)
          {
            norme_A += hview[k]*hview[k] ;
            trace_mng->info() <<"\t("<<irow<<","<<hview.col(k)<<","<<hview[k]<<")";
          }
      }
      trace_mng->info() << "NORME2 A : "<<norme_A ;
    }
    else
    {
      Real norme_A = 0. ;
      auto hview = builder.hostView();
      for(Alien::Integer irow=0;irow<local_size;++irow)
      {
          for(auto k=hview.kcol(irow);k<hview.kcol(irow+1);++k)
          {
            norme_A += hview[k]*hview[k] ;
          }
      }
      trace_mng->info() << "NORME2 A : "<<norme_A ;
    }
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
        trace_mng->info() << "Y[" << i << "]=" << reader[i];
      }
      trace_mng->info() << "NORME2 Y=A*X : "<<norme_y ;
     }
  }

  timer.printInfo(trace_mng->info().file(), "SYCL-BENCH");

  trace_mng->info() << "INFO FINALIZE SYCL TEST";

  MPI_Finalize();

  return 0;
}
