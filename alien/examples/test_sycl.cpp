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

#include <boost/lexical_cast.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/cmdline.hpp>
#include <boost/program_options/variables_map.hpp>

#include <arccore/message_passing_mpi/StandaloneMpiMessagePassingMng.h>
#include <arccore/base/StringBuilder.h>

#include <alien/kernels/simple_csr/algebra/SimpleCSRLinearAlgebra.h>

#include <alien/kernels/sycl/SYCLPrecomp.h>
#include <alien/kernels/sycl/data/SYCLBEllPackMatrix.h>
#include <alien/kernels/sycl/data/SYCLVector.h>
#include <alien/kernels/sycl/algebra/SYCLLinearAlgebra.h>
#include <alien/kernels/sycl/algebra/SYCLInternalLinearAlgebra.h>

#include "alien/kernels/sycl/data/SYCLEnv.h"
#include "alien/kernels/sycl/data/SYCLEnvInternal.h"
#include <alien/kernels/sycl/algebra/SYCLKernelInternal.h>

#include <alien/ref/AlienRefSemantic.h>

#include <alien/utils/StdTimer.h>

int main(int argc, char** argv)
{

  using namespace boost::program_options;
  options_description desc;
  // clang-format off
  desc.add_options()
      ("help", "produce help")
      ("size",    value<int>()->default_value(16),            "size")
      ("nb-test", value<int>()->default_value(1),             "nb tests")
      ("test",    value<std::string>()->default_value("all"), "test")
      ("dot-algo", value<int>()->default_value(0),            "dot algo choice") ;
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
  std::string test    = vm["test"].as<std::string>() ;
  int nb_test         = vm["nb-test"].as<int>() ;
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
      writer[i] = offset + i;
  }

  {
    Alien::LocalVectorWriter writer(r);
    for (Integer i = 0; i < local_size; ++i)
      writer[i] = 0.;
  }

  Alien::SimpleCSRLinearAlgebra csr_alg;
  Alien::SYCLLinearAlgebra sycl_alg;

  if ((test.compare("all") == 0) || (test.compare("copy") == 0)) {
    trace_mng->info() << "TEST COPY : r = y";
    {
      // clang-format off
      auto const& csr_y = y.impl()->get<Alien::BackEnd::tag::simplecsr>() ;
      auto&       csr_r = r.impl()->get<Alien::BackEnd::tag::simplecsr>() ;
      // clang-format on
      for (int i = 0; i < nb_test; ++i) {
        SentryType s(timer, "CSR-COPY");
        csr_alg.copy(y, r);
      }
    }
    {
      // clang-format off
      auto const& sycl_y = y.impl()->get<Alien::BackEnd::tag::sycl>() ;
      auto&       sycl_r = r.impl()->get<Alien::BackEnd::tag::sycl>() ;
      // clang-format on
      for (int i = 0; i < nb_test; ++i) {
        SentryType s(timer, "SYLC-COPY");
        sycl_alg.copy(y, r);
      }
    }
    {
      Alien::LocalVectorReader reader(r);
      for (Integer i = 0; i < std::min(10, local_size); ++i) {
        trace_mng->info() << "R[" << i << "]=" << reader[i];
      }
    }
  }

  if ((test.compare("all") == 0) || (test.compare("axpy") == 0)) {
    trace_mng->info() << "TEST AXPY : y += a*x ";
    {
      // clang-format off
      auto const& csr_x = x.impl()->get<Alien::BackEnd::tag::simplecsr>() ;
      auto&       csr_y = y.impl()->get<Alien::BackEnd::tag::simplecsr>() ;
      // clang-format on
      for (int i = 0; i < nb_test; ++i) {
        SentryType s(timer, "CSR-AXPY");
        csr_alg.axpy(1., x, y);
      }
    }
    {
      // clang-format off
      auto const& sycl_x = x.impl()->get<Alien::BackEnd::tag::sycl>() ;
      auto&       sycl_y = y.impl()->get<Alien::BackEnd::tag::sycl>() ;
      // clang-format on
      for (int i = 0; i < nb_test; ++i) {
        SentryType s(timer, "SYCL-AXPY");
        sycl_alg.axpy(1., x, y);
      }
    }

    {
      Alien::LocalVectorReader reader(y);
      for (Integer i = 0; i < std::min(10, local_size); ++i) {
        trace_mng->info() << "Y[" << i << "]=" << reader[i];
      }
    }
  }

  if ((test.compare("all") == 0) || (test.compare("dot") == 0)) {
    trace_mng->info() << "TEST DOT : dot(x,y) ";
    Real x_dot_y_ref = 0.;
    {
      Alien::LocalVectorReader reader_x(x);
      Alien::LocalVectorReader reader_y(y);
      for (Integer i = 0; i < local_size; ++i)
        x_dot_y_ref += reader_x[i] * reader_y[i];
      if (is_parallel)
        x_dot_y_ref = Arccore::MessagePassing::mpAllReduce(pm,
                                                           Arccore::MessagePassing::ReduceSum,
                                                           x_dot_y_ref);
    }
    Real x_dot_y = 0.;
    Real x_dot_y2 = 0.;
    Alien::SYCLInternalLinearAlgebra::FutureType f_x_dot_y(x_dot_y2);

    {
      auto const& csr_x = x.impl()->get<Alien::BackEnd::tag::simplecsr>();
      auto const& csr_y = y.impl()->get<Alien::BackEnd::tag::simplecsr>();

      for (int itest = 0; itest < nb_test; ++itest) {
        //std::cout<<itest<<"*";
        SentryType s(timer, "CSR-DOT");
        x_dot_y = csr_alg.dot(x, y);
      }
      //std::cout<<std::endl ;
    }
    {
      int dot_algo = vm["dot-algo"].as<int>();
      Alien::SYCLInternalLinearAlgebra internal_sycl_alg;
      internal_sycl_alg.setDotAlgo(dot_algo);
      auto const& sycl_x = x.impl()->get<Alien::BackEnd::tag::sycl>();
      auto const& sycl_y = y.impl()->get<Alien::BackEnd::tag::sycl>();

      for (int itest = 0; itest < nb_test; ++itest) {
        //std::cout<<itest<<"*";
        SentryType s(timer, "SYLC-DOT");
        x_dot_y = internal_sycl_alg.dot(sycl_x, sycl_y);
      }
      for (int itest = 0; itest < nb_test; ++itest) {
        //std::cout<<itest<<"*";
        SentryType s(timer, "SYLC-DOT-F");
        internal_sycl_alg.dot(sycl_x, sycl_y, f_x_dot_y);
      }

      //std::cout<<std::endl ;
    }
    trace_mng->info() << "SYCL DOT(X,Y) = " << x_dot_y << ", F-DOT = " << f_x_dot_y.get() << ", REF=" << x_dot_y_ref;
  }

  if ((test.compare("all") == 0) || (test.compare("mult") == 0)) {
    trace_mng->info() << "TEST SPMV : y = A*x ";
    {
      // clang-format off
      auto const& csr_A = A.impl()->get<Alien::BackEnd::tag::simplecsr>() ;
      auto const& csr_x = x.impl()->get<Alien::BackEnd::tag::simplecsr>() ;
      auto&       csr_y = y.impl()->get<Alien::BackEnd::tag::simplecsr>() ;
      // clang-format on
      for (int i = 0; i < nb_test; ++i) {
        SentryType s(timer, "CSR-SPMV");
        csr_alg.mult(A, x, y);
      }
    }
    {
      Alien::LocalVectorReader reader(y);
      for (Integer i = 0; i < std::min(10, local_size); ++i) {
        trace_mng->info() << my_rank << " CSR Y[" << i << "]=" << reader[i];
      }
    }
    {
      // clang-format off
      auto const& sycl_A = A.impl()->get<Alien::BackEnd::tag::sycl>() ;
      auto const& sycl_x = x.impl()->get<Alien::BackEnd::tag::sycl>() ;
      auto&       sycl_y = y.impl()->get<Alien::BackEnd::tag::sycl>() ;
      // clang-format on
      for (int i = 0; i < nb_test; ++i) {
        SentryType s(timer, "SYLC-SPMV");
        sycl_alg.mult(A, x, y);
      }
    }
    {
      Alien::LocalVectorReader reader(y);
      for (Integer i = 0; i < std::min(10, local_size); ++i) {
        trace_mng->info() << my_rank << " SYCL Y[" << i << "]=" << reader[i];
      }
    }
  }

  timer.printInfo(trace_mng->info().file(), "SYCL-BENCH");

  trace_mng->info() << "INFO FINALIZE SYCL TEST";

  MPI_Finalize();

  return 0;
}
