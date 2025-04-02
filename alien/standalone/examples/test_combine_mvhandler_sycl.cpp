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

#define ARCCORE_DEVICE_CODE
#include <arccore/message_passing_mpi/StandaloneMpiMessagePassingMng.h>
#include <arccore/base/StringBuilder.h>

#include <alien/distribution/MatrixDistribution.h>
#include <alien/distribution/VectorDistribution.h>
#include <alien/index_manager/IIndexManager.h>
#include <alien/index_manager/IndexManager.h>
#include <alien/index_manager/functional/AbstractItemFamily.h>
#include <alien/index_manager/functional/BasicIndexManager.h>
#include <alien/index_manager/functional/DefaultAbstractFamily.h>
#include <alien/ref/AlienRefSemantic.h>


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
#include <alien/handlers/scalar/sycl/CombineProfiledMatrixBuilderT.h>

#include <alien/kernels/sycl/data/SYCLParallelEngineImplT.h>
#include <alien/handlers/scalar/sycl/VectorAccessorImplT.h>
#include <alien/handlers/scalar/sycl/ProfiledMatrixBuilderImplT.h>
#include <alien/handlers/scalar/sycl/CombineProfiledMatrixBuilderImplT.h>

#include <alien/kernels/sycl/algebra/SYCLLinearAlgebra.h>
#include <alien/kernels/simple_csr/algebra/SimpleCSRLinearAlgebra.h>

#include <alien/ref/AlienRefSemantic.h>

#include <alien/utils/StdTimer.h>

struct Mesh
{
  int m_nx;
  int m_ny;
  int nodeLid(int i, int j) const {
    return j*m_nx+i ;
  }
  int cellLid(int i, int j) const {
    return j*(m_nx-1)+i ;
  }
  int nbNodes() const {
    return m_nx*m_ny ;
  }
  int nbCells() const {
    return (m_nx-1)*(m_ny-1) ;
  }

};

template<>
struct sycl::is_device_copyable<Mesh> : std::true_type {};

int main(int argc, char** argv)
{

  using namespace boost::program_options;
  options_description desc;
  // clang-format off
  desc.add_options()
      ("help", "produce help")
      ("nx",    value<int>()->default_value(4),            "nb node along X axe")
      ("ny",    value<int>()->default_value(4),            "nb nodes alog Y") ;
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

  auto comm_size = pm->commSize();
  auto comm_rank = pm->commRank();

  int local_nx = vm["nx"].as<int>() ;
  int ny       = vm["ny"].as<int>() ;
  int nx       = local_nx * comm_size ;

  auto node_global_size = nx * ny;
  auto cell_global_size = (nx - 1) * (ny - 1);

  auto mesh = Mesh{local_nx,ny} ;

  Alien::UniqueArray<Alien::Int64> node_uid;
  Alien::UniqueArray<Alien::Integer> node_lid;
  Alien::UniqueArray<Alien::Int64> cell_uid;
  Alien::UniqueArray<Alien::Integer> cell_lid;

  auto node_local_size = local_nx * ny;
  node_uid.reserve(node_local_size);
  node_lid.reserve(node_local_size);
  for (int i = 0; i < node_local_size; ++i) {
    node_uid.add(comm_rank * node_local_size + i);
    node_lid.add(i);
  }

  auto node_family = Alien::DefaultAbstractFamily(node_uid, pm);

  auto cell_local_size = local_nx * (ny - 1);
  if (comm_rank == comm_size - 1)
    cell_local_size = (local_nx-1) * (ny - 1);

  cell_uid.reserve(cell_local_size);
  cell_lid.reserve(cell_local_size);
  for (int i = 0; i < cell_local_size; ++i) {
    cell_uid.add(comm_rank * cell_local_size + i);
    cell_lid.add(i);
  }

  Alien::DefaultAbstractFamily cell_family(cell_uid, pm);

  Alien::IndexManager index_manager(pm);

  auto indexSetU = index_manager.buildScalarIndexSet("U", node_lid, node_family, 0);
  auto indexSetV = index_manager.buildScalarIndexSet("V", cell_lid, cell_family, 1);

  index_manager.prepare();

  auto allUIndex = index_manager.getIndexes(indexSetU) ;
  auto allVIndex = index_manager.getIndexes(indexSetV) ;

  auto space = Alien::Space(index_manager.globalSize(), "MySpace");

  auto mdist = Alien::MatrixDistribution(space, space, pm);
  auto vdist = Alien::VectorDistribution(space, pm);
  auto A = Alien::Matrix(mdist); // A.setName("A") ;

  auto local_size = vdist.localSize();
  auto offset = vdist.offset();

  {
    Alien::SYCL::MatrixProfiler profiler(A);
    for (Integer i = 0; i < nx; ++i) {
        for (Integer j = 0; j< ny; ++j) {
            Integer node_lid = mesh.nodeLid(i,j) ;
            Integer row = allUIndex[node_lid];
            profiler.addMatrixEntry(row, row);
            if(j<ny-1)
            {
              if(i<nx-1)
              {
                Integer cell_lid = mesh.cellLid(i,j) ;
                Integer col = allVIndex[cell_lid] ;
                profiler.addMatrixEntry(row, col);
              }
              if(i>0)
              {
                Integer cell_lid =  mesh.cellLid(i-1,j) ;
                Integer col = allVIndex[cell_lid] ;
                profiler.addMatrixEntry(row, col);
              }
            }
            if(j>0)
            {
              if(i<nx-1)
              {
                Integer cell_lid =  mesh.cellLid(i,j-1) ;
                Integer col = allVIndex[cell_lid] ;
                profiler.addMatrixEntry(row, col);
              }
              if(i>0)
              {
                Integer cell_lid =  mesh.cellLid(i-1,j-1) ;
                Integer col = allVIndex[cell_lid] ;
                profiler.addMatrixEntry(row, col);
              }
            }
        }
    }
    for (Integer i = 0; i < nx-1; ++i) {
        for (Integer j = 0; j< ny-1; ++j) {
            Integer cell_lid = mesh.cellLid(i,j) ;
            Integer row = allVIndex[cell_lid] ;
            profiler.addMatrixEntry(row, row);
            Integer col0 = allUIndex[mesh.nodeLid(i,j)] ;
            profiler.addMatrixEntry(row, col0);
            Integer col1 = allUIndex[mesh.nodeLid(i+1,j)] ;
            profiler.addMatrixEntry(row, col1);
            Integer col2 = allUIndex[mesh.nodeLid(i+1,j+1)] ;
            profiler.addMatrixEntry(row, col2);
            Integer col3 = allUIndex[mesh.nodeLid(i+1,j)] ;
            profiler.addMatrixEntry(row, col3);
        }
    }
  }

  Alien::UniqueArray<Alien::Integer> connection_offset(local_size+1);
  connection_offset.fill(0) ;
  for(int i=0;i<nx-1;++i)
    for(int j=0;j<ny-1;++j)
      {
        auto cell_id = mesh.cellLid(i,j) ;
        connection_offset[allVIndex[cell_id]] = 4 ;
      }
  {
    int offset = 0 ;
    for(int index=0;index<local_size;++index)
      {
        auto size = connection_offset[index] ;
        connection_offset[index] = offset ;
        offset += size ;
      }
    connection_offset[local_size] = offset ;
  }
  Alien::UniqueArray<Alien::Integer> connection_index(connection_offset[local_size]);
  {
    for(int i=0;i<nx-1;++i)
      for(int j=0;j<ny-1;++j)
        {
          auto cell_id = mesh.cellLid(i,j) ;
          auto offset =  connection_offset[allVIndex[cell_id]] ;
          {
            auto node_id = mesh.nodeLid(i,j) ;
            auto u_index = allUIndex[node_id] ;
            connection_index[offset+0] = u_index ;
          }
          {
            auto node_id = mesh.nodeLid(i+1,j) ;
            auto u_index = allUIndex[node_id] ;
            connection_index[offset+1] = u_index ;
          }
          {
            auto node_id = mesh.nodeLid(i+1,j+1) ;
            auto u_index = allUIndex[node_id] ;
            connection_index[offset+2] = u_index ;
          }
          {
            auto node_id = mesh.nodeLid(i,j+1) ;
            auto u_index = allUIndex[node_id] ;
            connection_index[offset+3] = u_index ;
          }
        }
  }
  Real cell_diag = 1. ;
  Real node_diag = 2. ;
  Real node_cell_off_diag = 0.1 ;
  Real cell_node_off_diag = 0.01 ;

  Alien::SYCLParallelEngine engine;
  {
    Alien::SYCL::CombineAddProfiledMatrixBuilder builder(A, Alien::ProfiledMatrixOptions::eResetValues);
    builder.setParallelAssembleStencil(4,connection_offset.view(),connection_index.view()) ;

    auto allUIndex_buffer = sycl::buffer<Integer,1>{allUIndex.data(),sycl::range(allUIndex.size())} ;
    auto allVIndex_buffer = sycl::buffer<Integer,1>{allVIndex.data(),sycl::range(allVIndex.size())} ;

    engine.submit([&](Alien::SYCLControlGroupHandler& cgh)
                  {
                    auto matrix_acc = builder.view(cgh) ;
                    auto allUIndex_acc = allUIndex_buffer.get_access<sycl::access::mode::read>(cgh.m_internal) ;
                    auto allVIndex_acc = allVIndex_buffer.get_access<sycl::access::mode::read>(cgh.m_internal) ;
                    cgh.parallel_for(nx,
                                     ny,
                                     [=](Alien::SYCLParallelEngine::Item<2>::type item)
                                     {
                                        auto i = item.get_id(0);
                                        auto j = item.get_id(1);
                                        auto node_id = mesh.nodeLid(i,j) ;
                                        auto row = allUIndex_acc[node_id] ;
                                        auto eii = matrix_acc.entryIndex(row,row) ;
                                        matrix_acc[eii] = node_diag ;

                                        if(j<ny-1)
                                        {
                                          if(i<nx-1)
                                          {
                                            Integer cell_lid = mesh.cellLid(i,j) ;
                                            Integer col = allVIndex_acc[cell_lid] ;
                                            matrix_acc[eii] += node_cell_off_diag ;
                                            auto eij = matrix_acc.entryIndex(row,col) ;
                                            matrix_acc[eij] = - node_cell_off_diag ;
                                            auto ejjk = matrix_acc.combineEntryIndex(row,col,col) ;
                                            matrix_acc.combine(ejjk, cell_diag) ;
                                            auto ejik = matrix_acc.combineEntryIndex(row,col,row) ;
                                            matrix_acc.combine(ejik, -cell_node_off_diag) ;
                                          }
                                          if(i>0)
                                          {
                                            Integer cell_lid =  mesh.cellLid(i-1,j) ;
                                            Integer col = allVIndex_acc[cell_lid] ;
                                            matrix_acc[eii] += node_cell_off_diag ;
                                            auto eij = matrix_acc.entryIndex(row,col) ;
                                            matrix_acc[eij] = - node_cell_off_diag ;
                                            auto ejjk = matrix_acc.combineEntryIndex(row,col,col) ;
                                            matrix_acc.combine(ejjk, cell_diag) ;
                                            auto ejik = matrix_acc.combineEntryIndex(row,col,row) ;
                                            matrix_acc.combine(ejik,- cell_node_off_diag) ;
                                          }
                                        }

                                        if(j>0)
                                        {
                                          if(i<nx-1)
                                          {
                                            Integer cell_lid =  mesh.cellLid(i,j-1) ;
                                            Integer col = allVIndex_acc[cell_lid] ;
                                            matrix_acc[eii] += node_cell_off_diag ;
                                            auto eij = matrix_acc.entryIndex(row,col) ;
                                            matrix_acc[eij] = - node_cell_off_diag ;
                                            auto ejjk = matrix_acc.combineEntryIndex(row,col,col) ;
                                            matrix_acc.combine(ejjk, cell_diag) ;
                                            auto ejik = matrix_acc.combineEntryIndex(row,col,row) ;
                                            matrix_acc.combine(ejik,- cell_node_off_diag) ;
                                          }
                                          if(i>0)
                                          {
                                            Integer cell_lid =  mesh.cellLid(i-1,j-1) ;
                                            Integer col = allVIndex_acc[cell_lid] ;
                                            matrix_acc[eii] += node_cell_off_diag ;
                                            auto eij = matrix_acc.entryIndex(row,col) ;
                                            matrix_acc[eij] = - node_cell_off_diag ;
                                            auto ejjk = matrix_acc.combineEntryIndex(row,col,col) ;
                                            matrix_acc.combine(ejjk, cell_diag) ;
                                            auto ejik = matrix_acc.combineEntryIndex(row,col,row) ;
                                            matrix_acc.combine(ejik,- cell_node_off_diag) ;
                                          }
                                        }

                                     });
                  }) ;
    builder.combine() ;

    if(local_size<20)
    {
      Real norme_A = 0. ;
      auto hview = builder.hostView();
      for(std::size_t irow=0;irow<local_size;++irow)
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
      for(std::size_t irow=0;irow<local_size;++irow)
      {
          for(auto k=hview.kcol(irow);k<hview.kcol(irow+1);++k)
          {
            norme_A += hview[k]*hview[k] ;
          }
      }
      trace_mng->info() << "NORME2 A : "<<norme_A ;
    }

  }

  timer.printInfo(trace_mng->info().file(), "SYCL-BENCH");

  trace_mng->info() << "INFO FINALIZE SYCL TEST";

  MPI_Finalize();

  return 0;
}
