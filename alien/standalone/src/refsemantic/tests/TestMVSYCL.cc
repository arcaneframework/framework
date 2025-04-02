// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#include <alien/ref/AlienRefSemantic.h>
#include <alien/utils/Precomp.h>
#include <cmath>
#include <gtest/gtest.h>

#include <alien/distribution/MatrixDistribution.h>
#include <alien/distribution/VectorDistribution.h>
#include <alien/index_manager/IIndexManager.h>
#include <alien/index_manager/IndexManager.h>
#include <alien/index_manager/functional/AbstractItemFamily.h>
#include <alien/index_manager/functional/BasicIndexManager.h>
#include <alien/index_manager/functional/DefaultAbstractFamily.h>

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
#include <alien/handlers/scalar/sycl/CombineProfiledMatrixBuilderT.h>

#include <alien/kernels/sycl/algebra/SYCLLinearAlgebra.h>
#include <alien/kernels/simple_csr/algebra/SimpleCSRLinearAlgebra.h>

#include <Environment.h>


#include <alien/kernels/sycl/data/SYCLParallelEngineImplT.h>
#include <alien/handlers/scalar/sycl/VectorAccessorImplT.h>
#include <alien/handlers/scalar/sycl/ProfiledMatrixBuilderImplT.h>
#include <alien/handlers/scalar/sycl/CombineProfiledMatrixBuilderImplT.h>

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
                                         [=](Alien::SYCLParallelEngine::Item<1>::type item)
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
                                         [=](Alien::SYCLParallelEngine::Item<1>::type item)
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
                                         [=](Alien::SYCLParallelEngine::Item<1>::type item)
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
                                         [=](Alien::SYCLParallelEngine::Item<1>::type item)
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
                                           [=](Alien::SYCLParallelEngine::Item<1>::type item)
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

TEST(TestSYCLMV, CombineAddBuilder)
{
  using namespace Alien;
  Alien::ITraceMng* trace_mng = AlienTest::Environment::traceMng();
  auto pm = AlienTest::Environment::parallelMng() ;

  auto comm_size = pm->commSize();
  auto comm_rank = pm->commRank();

  int ny       = 3 ;
  int local_nx = 3 ;
  int nx = local_nx * comm_size ;

  auto node_global_size = nx * ny;
  auto cell_global_size = (nx - 1) * (ny - 1);

  auto mesh = Mesh{nx,ny} ;

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
    {
      auto hview = builder.hostView();

      for(int i=0;i<nx;++i)
        for(int j=0;j<ny;++j)
        {
          auto node_id = mesh.nodeLid(i,j) ;
          auto row = allUIndex[node_id] ;
          if(j<ny-1)
          {
            if(i<nx-1)
            {
              Integer cell_lid = mesh.cellLid(i,j) ;
              Integer col = allVIndex[cell_lid] ;
              auto eij = hview.entryIndex(row,col) ;
              auto ejjk = hview.combineEntryIndex(row,col,col) ;
              auto ejik = hview.combineEntryIndex(row,col,row) ;
            }
          }
        }
    }
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

      ASSERT_DOUBLE_EQ(106.9212, norme_A);
    }
  }
}
