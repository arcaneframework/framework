// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#include <mpi.h>

#include <string>
#include <map>
#include <time.h>
#include <vector>
#include <fstream>

//#ifdef ALIEN_USE_SYCL
#define ARCCORE_DEVICE_CODE
//#endif

#include <arcane/ArcaneVersion.h>
#include <arcane/Timer.h>
#include <arcane/ItemPairGroup.h>
#include <arcane/mesh/ItemFamily.h>
#include <arcane/utils/PlatformUtils.h>
#include <arcane/utils/IMemoryInfo.h>
#include <arcane/utils/OStringStream.h>
#include <arcane/ITimeLoopMng.h>
#include <arccore/base/Span.h>


#include <alien/arcane_tools/accessors/ItemVectorAccessor.h>
#include <alien/core/block/VBlock.h>

#include "arcane/accelerator/core/IAcceleratorMng.h"
#include "arcane/accelerator/Reduce.h"
#include "arcane/accelerator/Accelerator.h"
#include "arcane/accelerator/RunCommandEnumerate.h"
#include "arcane/accelerator/NumArrayViews.h"
#include "arcane/accelerator/SpanViews.h"

#include <alien/arcane_tools/IIndexManager.h>
#include <alien/arcane_tools/indexManager/BasicIndexManager.h>
#include <alien/arcane_tools/indexManager/SimpleAbstractFamily.h>
#include <alien/arcane_tools/distribution/DistributionFabric.h>
#include <alien/arcane_tools/indexSet/IndexSetFabric.h>
#include <alien/arcane_tools/data/Space.h>

#include <alien/handlers/scalar/sycl/VectorAccessorT.h>
#include <alien/handlers/scalar/sycl/MatrixProfiler.h>
#include <alien/handlers/scalar/sycl/ProfiledMatrixBuilderT.h>
#include <alien/handlers/scalar/sycl/CombineProfiledMatrixBuilderT.h>

#include <alien/handlers/scalar/sycl/CombineProfiledMatrixBuilderImplT.h>
#include <alien/handlers/scalar/sycl/VectorAccessorImplT.h>
#include <alien/handlers/scalar/sycl/ProfiledMatrixBuilderImplT.h>

#include <alien/kernels/simple_csr/algebra/SimpleCSRLinearAlgebra.h>
#include <alien/kernels/simple_csr/algebra/SimpleCSRInternalLinearAlgebra.h>

#include <alien/ref/AlienRefSemantic.h>

#include <alien/kernels/redistributor/Redistributor.h>
#include <alien/ref/data/scalar/RedistributedVector.h>
#include <alien/ref/data/scalar/RedistributedMatrix.h>
#include <alien/ref/import_export/MatrixMarketSystemWriter.h>

#include <alien/expression/solver/SolverStater.h>

#include <alien/arcane_tools/accelerator/ArcaneParallelEngine.h>

#include <alien/arcane_tools/accelerator/ArcaneParallelEngineImplT.h>

#ifdef ALIEN_USE_PETSC
#include <alien/kernels/petsc/io/AsciiDumper.h>
#include <alien/kernels/petsc/algebra/PETScLinearAlgebra.h>
#endif

#ifdef ALIEN_USE_MTL4
#include <alien/kernels/mtl/algebra/MTLLinearAlgebra.h>
#endif
#ifdef ALIEN_USE_HTSSOLVER
#include <alien/kernels/hts/HTSBackEnd.h>
#include <alien/kernels/hts/data_structure/HTSMatrix.h>
#include <alien/kernels/hts/algebra/HTSLinearAlgebra.h>
#endif
#ifdef ALIEN_USE_TRILINOS
#include <alien/kernels/trilinos/TrilinosBackEnd.h>
#include <alien/kernels/trilinos/data_structure/TrilinosMatrix.h>
#include <alien/kernels/trilinos/algebra/TrilinosLinearAlgebra.h>
#endif

#ifdef ALIEN_USE_SYCL
#include <alien/kernels/sycl/SYCLPrecomp.h>

#include "alien/kernels/sycl/data/SYCLEnv.h"
#include "alien/kernels/sycl/data/SYCLEnvInternal.h"

#include <alien/kernels/sycl/data/SYCLBEllPackMatrix.h>
#include <alien/kernels/sycl/data/SYCLVector.h>
#include <alien/kernels/sycl/algebra/SYCLLinearAlgebra.h>

#include "alien/kernels/sycl/data/SYCLVectorInternal.h"
#include <alien/kernels/sycl/data/SYCLBEllPackInternal.h>
#include <alien/kernels/sycl/algebra/SYCLInternalLinearAlgebra.h>

#include "alien/kernels/sycl/data/SYCLSendRecvOp.h"
#include "alien/kernels/sycl/data/SYCLLUSendRecvOp.h"
#include <alien/kernels/sycl/algebra/SYCLKernelInternal.h>
#endif

#include <alien/expression/solver/ILinearSolver.h>
#include "AlienCoreSolverOptionTypes.h"

#include "AlienBenchModule.h"

#include <arcane/ItemPairGroup.h>
#include <arcane/IMesh.h>

#include <alien/core/impl/MultiVectorImpl.h>

#include <alien/expression/krylov/AlienKrylov.h>
#include <alien/utils/StdTimer.h>

using namespace Arcane;
using namespace Alien;

namespace ax = Arcane::Accelerator;
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
class Computer
{
public :
  bool m_homogeneous ;
  Real m_diag_coeff ;
  Real m_off_diag_coeff ;
  Real m_lambdax = 1.;
  Real m_lambday = 1.;
  Real m_lambdaz = 1.;
  Real m_alpha = 1.;
  Real m_sigma = 0.;

  Real dii(Integer ci) const {
    return m_diag_coeff ;
  }

  Real funcn(Real3 p) const
  {
    return p.x * p.x * p.y;
  }

  Real funck(Real3 p) const
  {
  #define PI 3.14159265358979323846264

    if (m_homogeneous)
      return m_off_diag_coeff;
    else
      return sycl::exp(-m_alpha * 0.5 * (1 + sycl::sin(2 * PI * p.x / m_lambdax))
          * (1 + sycl::sin(2 * PI * p.y / m_lambday)));
  }

  Real fij(Integer vi, Integer vj, Arcane::Real3 xi, Arcane::Real3 xj) const
  {
    if (vi == vj)
      return dii(vi);
    else {
      auto xij = xi + xj;
      xij /= 2.;

      return funck(xij);
    }
  }
} ;

template<>
struct sycl::is_device_copyable<Real3> : std::true_type {};

template<>
struct sycl::is_device_copyable<Computer> : std::true_type {};

//template<>
//struct sycl::is_device_copyable<ConstArrayView<Integer>> : std::true_type {};

void
AlienBenchModule::_testSYCLWithUSM( Timer& pbuild_timer,
                                    CellGroup& areaU,
                                    CellCellGroup& cell_cell_connection,
                                    CellCellGroup& all_cell_cell_connection,
                                    Arccore::UniqueArray<Arccore::Integer>& allUIndex,
                                    Alien::Vector& vectorB,
                                    Alien::Vector& vectorBB,
                                    Alien::Vector& vectorX,
                                    Alien::Vector& coordX,
                                    Alien::Vector& coordY,
                                    Alien::Vector& coordZ,
                                    Alien::Matrix& matrixA)
{

  {
    m_cell_cell_connection_offset.resize(areaU.own().size()+1) ;
    Integer offset = 0 ;
    Integer index = 0 ;
    ENUMERATE_ITEMPAIR(Cell, Cell, icell, cell_cell_connection)
    {
      m_cell_cell_connection_offset[index++] = offset ;
      offset += icell.subItems().count() ;
    }
    m_cell_cell_connection_offset[index] = offset ;
    m_cell_cell_connection_index.resize(offset) ;
    offset = 0 ;
    ENUMERATE_ITEMPAIR(Cell, Cell, icell, cell_cell_connection)
    {
      ENUMERATE_SUB_ITEM(Cell, isubcell, icell)
      {
        m_cell_cell_connection_index[offset++] = isubcell->localId() ;
      }
    }
  }

  //Arcane::NumArray<Arccore::Integer,MDDim1> accAllUIndex(allUIndex.constView()) ;
  Arccore::SmallSpan<const Int32> accAllUIndex = allUIndex ;//index_manager.getIndexes(indexSetU);
  //Arccore::SmallSpan<const Int32> cell_conn_lids = cell_cell_connection.itemGroup().view().localIds() ;
  //Arccore::SmallSpan<const Integer> cell_lids = areaU.view().localIds() ;
  //Arccore::SmallSpan<const Integer> own_cell_lids = areaU.own().view().localIds() ;

  Arccore::UniqueArray<Arccore::Integer> cell_conn_lids(platform::getDefaultDataAllocator()) ;
  cell_conn_lids = cell_cell_connection.itemGroup().view().localIds() ;
  Arccore::UniqueArray<Arccore::Integer> all_cell_conn_lids(platform::getDefaultDataAllocator()) ;
  all_cell_conn_lids = all_cell_cell_connection.itemGroup().view().localIds() ;

  Arccore::UniqueArray<Arccore::Integer> cell_lids(platform::getDefaultDataAllocator()) ;
  cell_lids = areaU.view().localIds() ;
  Arccore::UniqueArray<Arccore::Integer> own_cell_lids(platform::getDefaultDataAllocator()) ;
  own_cell_lids = areaU.own().view().localIds() ;

  ///////////////////////////////////////////////////////////////////////////
  //
  // VECTOR BUILDING AND FILLING
  //
  info() << "Building & initializing vector b";
  info() << "Space size = " << m_vdist.globalSize()
         << ", local size= " << m_vdist.localSize();

  Computer computer{m_homogeneous,
                    m_diag_coeff,
                    m_off_diag_coeff,
                    m_lambdax,
                    m_lambday,
                    m_lambdaz,
                    m_alpha,
                    m_sigma} ;
  {
    ENUMERATE_CELL (icell, areaU)
    {
      Real3 x;
      for (Arcane::Node node : icell->nodes()) {
        x += m_node_coord[node];
      }
      x /= icell->nbNode();
      m_cell_center[icell] = x;
      if(icell->isOwn())
        m_cell_is_own[icell] = 1 ;
      else
        m_cell_is_own[icell] = 0 ;
    }

    Alien::ParallelEngine engine(*m_default_queue) ;
    engine.submit([&](ControlGroupHandler& handler)
                  {
                    auto& command = handler.command() ;
                    auto in_cell_lids = ax::viewIn(command,cell_lids);
                    auto in_center = ax::viewIn(command,m_cell_center);
                    auto out_u = ax::viewOut(command,m_u);
                    auto out_k = ax::viewOut(command,m_k);

                    auto local_size = cell_lids.size() ;
                    handler.parallel_for(engine.maxNumThreads(),
                                         [=](Alien::ParallelEngine::Item<1>::type item)
                                         {
                                            auto id = item.get_id(0) ;
                                            for (auto index = id; index < local_size; index += item.get_range()[0])
                                            {
                                              auto vi   = CellLocalId(in_cell_lids[index]) ;
                                              //auto vi = in_cell_lids[index] ;
                                              auto x    = in_center[vi] ;
                                              out_u[vi] = computer.funcn(x) ;
                                              out_k[vi] = computer.funck(x) ;
                                            }
                                          });
                  }) ;
  }

  {
    auto vx_acc = Alien::SYCL::VectorAccessorT<Real>(vectorX);
    auto cx_acc = Alien::SYCL::VectorAccessorT<Real> (coordX);
    auto cy_acc = Alien::SYCL::VectorAccessorT<Real> (coordY);
    auto cz_acc = Alien::SYCL::VectorAccessorT<Real> (coordZ);

    Alien::ParallelEngine engine(*m_default_queue) ;

    engine.submit([&](ControlGroupHandler& handler)
                  {
                    auto& command         = handler.command() ;
                    auto in_cell_lids     = ax::viewIn(command,cell_lids);
                    //auto in_cell_lids     = ax::viewIn(command,own_cell_lids);
                    auto in_center        = ax::viewIn(command,m_cell_center);
                    auto in_is_own        = ax::viewIn(command,m_cell_is_own);
                    auto in_allUIndex     = ax::viewIn(command,accAllUIndex) ;
                    auto in_u             = ax::viewIn(command,m_u);

                    auto out_vx = vx_acc.view(handler) ;
                    auto out_cx = cx_acc.view(handler) ;
                    auto out_cy = cy_acc.view(handler) ;
                    auto out_cz = cz_acc.view(handler) ;

                    //auto local_size = own_cell_lids.size() ;
                    auto local_size = cell_lids.size() ;

                    //command << RUNCOMMAND_ENUMERATE(Cell,vi,areaU.own())

                    handler.parallel_for(engine.maxNumThreads(),
                                         [=](Alien::ParallelEngine::Item<1>::type item)
                                         {
                                            auto id = item.get_id(0) ;
                                            for (auto index = id; index < local_size; index += item.get_range()[0])
                                            {
                                              auto vi   = CellLocalId(in_cell_lids[index]) ;
                                              auto lid     = in_cell_lids[index] ;
                                              auto iIndex = in_allUIndex[lid];
                                              auto is_own = in_is_own[vi] ;
                                              auto xC     = in_center[vi] ;
                                              if(iIndex!=-1 && is_own==1)
                                              {
                                                out_vx[iIndex] = in_u[vi] ;
                                                out_cx[iIndex] = xC.x ;
                                                out_cy[iIndex] = xC.y ;
                                                out_cz[iIndex] = xC.z ;
                                              }
                                            } ;
                                         }) ;

                  }) ;
  }

  ///////////////////////////////////////////////////////////////////////////
  //
  // MATRIX BUILDING AND FILLING
  //
  {
      Timer::Sentry ts(&pbuild_timer);
      {
        Alien::SYCL::MatrixProfiler profiler(matrixA);
        ///////////////////////////////////////////////////////////////////////////
        //
        // DEFINE PROFILE
        //
        ENUMERATE_ITEMPAIR(Cell, Cell, icell, cell_cell_connection)
        {
          const Cell& cell = *icell;
          const Integer iIndex = allUIndex[cell.localId()];
          profiler.addMatrixEntry(iIndex, iIndex);
          ENUMERATE_SUB_ITEM(Cell, isubcell, icell)
          {
            const Cell& subcell = *isubcell;
            profiler.addMatrixEntry(iIndex, allUIndex[subcell.localId()]);
          }
        }
      }

      {
        Alien::SYCL::CombineMultProfiledMatrixBuilder builder(matrixA, Alien::ProfiledMatrixOptions::eResetValues);
        builder.setParallelAssembleStencil(1,m_cell_cell_connection_offset.view(),m_cell_cell_connection_index.view()) ;

        Alien::ParallelEngine engine(*m_default_queue) ;
        engine.submit([&](ControlGroupHandler& handler)
                      {
                        auto& command = handler.command() ;

                        auto in_allUIndex      = ax::viewIn(command,accAllUIndex) ;
                        auto in_conn_index     = ax::viewIn(command,m_cell_cell_connection_index) ;
                        auto in_conn_offset    = ax::viewIn(command,m_cell_cell_connection_offset) ;
                        auto in_cell_conn_lids = ax::viewIn(command,cell_conn_lids) ;
                        auto in_center         = ax::viewIn(command,m_cell_center);
                        auto in_is_own         = ax::viewIn(command,m_cell_is_own);

                        auto matrix_acc = builder.view(handler) ;
                        auto local_size = cell_cell_connection.itemGroup().size() ;
                        //command << RUNCOMMAND_ENUMERATE(Cell,vi,CellGroup(cell_cell_connection.itemGroup()))
                        //command << RUNCOMMAND_LOOP1(iter,cell_cell_connection.itemGroup().size())

                        handler.parallel_for(engine.maxNumThreads(),
                                             [=](Alien::ParallelEngine::Item<1>::type item)
                                             {
                                                auto id = item.get_id(0) ;
                                                for (auto index = id; index < local_size; index += item.get_range()[0])
                                                {
                                                  auto lid = in_cell_conn_lids[index] ;
                                                  auto vi = CellLocalId(in_cell_conn_lids[index]) ;
                                                  double diag = computer.dii(lid);

                                                  auto xi = in_center[vi] ;

                                                  Integer i = in_allUIndex[lid];
                                                  auto eii = matrix_acc.entryIndex(i, i) ;
                                                  matrix_acc[eii] += diag;

                                                  for(auto k=in_conn_offset[index];k<in_conn_offset[index+1];++k)
                                                  {
                                                    auto slid = in_conn_index[k] ;
                                                    auto svj = CellLocalId(slid) ;
                                                    //auto svj = in_conn_index[k] ;
                                                    auto xj = in_center[svj] ;
                                                    Integer j = in_allUIndex[slid];
                                                    auto eij = matrix_acc.entryIndex(i, j) ;

                                                    double off_diag = computer.fij(lid, slid,xi,xj);
                                                    matrix_acc[eii] += off_diag;
                                                    matrix_acc[eij] -= off_diag;
                                                  }
                                                } ;
                                              }) ;
                      }) ;
        /*
        {
          auto hview = builder.hostView();
          for(std::size_t index=0;index<cell_lids.size();++index)
          {
              auto i = allUIndex[cell_lids[index]] ;
              auto eii = hview.entryIndex(i,i) ;
              std::cout <<" ROW ["<<index<<"]: DIAG("<<cell_lids[index]<<","<<i<<","<<eii<<","<<hview[eii]<<") ";
              for(std::size_t k=m_cell_cell_connection_offset[index];k<m_cell_cell_connection_offset[index+1];++k)
              {
                  auto jindex = m_cell_cell_connection_index[k] ;
                  auto j =  allUIndex[cell_lids[jindex]] ;
                  auto eij =  hview.entryIndex(i,j) ;
                std::cout <<"("<<cell_lids[jindex]<<","<<j<<","<<eij<<","<<hview[eij]<<")";
              }
              std::cout<<std::endl ;
          }
        }*/
        if (options()->sigma() > 0.)
        {
          m_sigma = options()->sigma();
          Arcane::Real3 xCmax { 0.25, 0.25, 0.25 };
          Arcane::Real3 xCmin { 0.75, 0.75, 0.55 };
          auto epsilon = options()->epsilon() ;
          auto sigma = m_sigma ;
          engine.submit([&](ControlGroupHandler& handler)
                        {
                          auto& command = handler.command() ;

                          auto in_center         = ax::viewIn(command,m_cell_center);
                          auto in_is_own         = ax::viewIn(command,m_cell_is_own);
                          auto out_s             = ax::viewOut(command,m_s);

                          //auto in_cell_lids      = ax::viewIn(command,all_cell_lids);
                          auto in_conn_index     = ax::viewIn(command,m_cell_cell_connection_index) ;
                          auto in_conn_offset    = ax::viewIn(command,m_cell_cell_connection_offset) ;
                          auto in_cell_conn_lids = ax::viewIn(command,all_cell_conn_lids) ;
                          auto in_allUIndex      = ax::viewIn(command,accAllUIndex) ;

                          /*
                          auto in_cell_lids      = cell_lids_buffer.get_access<sycl::access::mode::read>(handler.m_internal) ;
                          auto in_conn_index     = cell_conn_index_buffer.get_access<sycl::access::mode::read>(handler.m_internal) ;
                          auto in_conn_offset    = cell_conn_offset_buffer.get_access<sycl::access::mode::read>(handler.m_internal) ;
                          auto in_cell_conn_lids = cell_conn_lids_buffer.get_access<sycl::access::mode::read>(handler.m_internal) ;
                          auto in_allUIndex      = allUIndex_buffer.get_access<sycl::access::mode::read>(handler.m_internal) ;

                          auto in_is_own = cell_is_own_buffer.get_access<sycl::access::mode::read>(handler.m_internal) ;
                          auto in_center = cell_center_buffer.get_access<sycl::access::mode::read>(handler.m_internal) ;
                          auto out_s     = s_buffer.get_access<sycl::access::mode::read_write>(handler.m_internal) ;
                          */

                          auto matrix_acc = builder.view(handler) ;
                          //command << RUNCOMMAND_ENUMERATE(Cell,vi,CellGroup(cell_cell_connection.itemGroup()))
                          //command << RUNCOMMAND_LOOP1(iter,cell_cell_connection.itemGroup().size())

                          auto local_size = all_cell_cell_connection.itemGroup().size() ;
                          handler.parallel_for(engine.maxNumThreads(),
                                               [=](Alien::ParallelEngine::Item<1>::type item)
                                               {
                                                  auto id = item.get_id(0) ;
                                                  for (auto index = id; index < local_size; index += item.get_range()[0])
                                                  {
                                                    auto lid = in_cell_conn_lids[index] ;
                                                    auto vi = CellLocalId(lid) ;

                                                    auto xC = in_center[vi];
                                                    auto xDmax = xC - xCmax;
                                                    auto xDmin = xC - xCmin;
                                                    out_s[vi] = 0.;
                                                    Integer i = in_allUIndex[lid];
                                                    auto eii = matrix_acc.entryIndex(i, i) ;
                                                    if (xDmax.normL2() < epsilon) {
                                                      out_s[vi] = 1.;
                                                      if (in_is_own[vi]==1)
                                                        matrix_acc[eii] = sigma;
                                                      for(auto k = in_conn_offset[index];k<in_conn_offset[index+1];++k)
                                                      {
                                                        auto slid = in_conn_index[k] ;
                                                        auto svi = CellLocalId(slid) ;
                                                        if(in_is_own[svi]==1) {
                                                            Integer j = in_allUIndex[slid];
                                                            auto ejik = matrix_acc.combineEntryIndex(i, j, i) ;
                                                            //matrix_acc[eji] = 0.;
                                                            matrix_acc.combine(ejik,0.) ;
                                                        }
                                                      }
                                                    }
                                                    if (xDmin.normL2() < epsilon) {
                                                      out_s[vi] = -1.;
                                                      //Integer i = in_allUIndex[lid];

                                                      if (in_is_own[vi])
                                                        matrix_acc[eii] = 1. / sigma;
                                                      for(auto k = in_conn_offset[index];k<in_conn_offset[index+1];++k)
                                                      {
                                                        auto slid = in_conn_index[k] ;
                                                        auto svi = CellLocalId(slid) ;
                                                        if(in_is_own[svi]==1) {
                                                            Integer j = in_allUIndex[slid];
                                                            auto ejik = matrix_acc.combineEntryIndex(i, j, i) ;
                                                            //matrix_acc[eji] = 0.;
                                                            matrix_acc.combine(ejik,0.) ;
                                                        }
                                                      }
                                                    }
                                                  } ;
                                               }) ;
                        }) ;
            builder.combine() ;
          }
        builder.finalize() ;
      }
  }
  {
    info()<<"COMPUTE SYCL NORME B";
    Alien::SYCLLinearAlgebra syclAlg;
    syclAlg.mult(matrixA, vectorX, vectorB);
    syclAlg.mult(matrixA, vectorX, vectorBB);
    Real normeb = syclAlg.norm2(vectorB);
    std::cout << "sycl ||b||=" << normeb<<std::endl ;
  }
}


void
AlienBenchModule::_testSYCL(Timer& pbuild_timer,
                            CellGroup& areaU,
                            CellCellGroup& cell_cell_connection,
                            CellCellGroup& all_cell_cell_connection,
                            Arccore::UniqueArray<Arccore::Integer>& allUIndex,
                            Alien::Vector& vectorB,
                            Alien::Vector& vectorBB,
                            Alien::Vector& vectorX,
                            Alien::Vector& coordX,
                            Alien::Vector& coordY,
                            Alien::Vector& coordZ,
                            Alien::Matrix& matrixA)
{

  {
    m_cell_cell_connection_offset.resize(areaU.own().size()+1) ;
    Integer offset = 0 ;
    Integer index = 0 ;
    ENUMERATE_ITEMPAIR(Cell, Cell, icell, cell_cell_connection)
    {
      m_cell_cell_connection_offset[index++] = offset ;
      offset += icell.subItems().count() ;
    }
    m_cell_cell_connection_offset[index] = offset ;
    m_cell_cell_connection_index.resize(offset) ;
    offset = 0 ;
    ENUMERATE_ITEMPAIR(Cell, Cell, icell, cell_cell_connection)
    {
      ENUMERATE_SUB_ITEM(Cell, isubcell, icell)
      {
        m_cell_cell_connection_index[offset++] = isubcell->localId() ;
      }
    }
  }

  //Arcane::NumArray<Arccore::Integer,MDDim1> accAllUIndex = allUIndex ;
  Arccore::SmallSpan<const Int32> accAllUIndex = allUIndex ;//index_manager.getIndexes(indexSetU);
  //Arccore::SmallSpan<const Int32> cell_conn_lids = cell_cell_connection.itemGroup().view().localIds() ;
  //Arccore::SmallSpan<const Integer> cell_lids = areaU.view().localIds() ;
  //Arccore::SmallSpan<const Integer> own_cell_lids = areaU.own().view().localIds() ;
  Arccore::UniqueArray<Arccore::Integer> cell_conn_lids(platform::getDefaultDataAllocator()) ;
  cell_conn_lids = cell_cell_connection.itemGroup().view().localIds() ;
  Arccore::UniqueArray<Arccore::Integer> all_cell_conn_lids(platform::getDefaultDataAllocator()) ;
  all_cell_conn_lids = all_cell_cell_connection.itemGroup().view().localIds() ;
  Arccore::UniqueArray<Arccore::Integer> cell_lids(platform::getDefaultDataAllocator()) ;
  cell_lids = areaU.view().localIds() ;
  Arccore::UniqueArray<Arccore::Integer> own_cell_lids(platform::getDefaultDataAllocator()) ;
  own_cell_lids = areaU.own().view().localIds() ;

  ///////////////////////////////////////////////////////////////////////////
  //
  // VECTOR BUILDING AND FILLING
  //
  info() << "Building & initializing vector b";
  info() << "Space size = " << m_vdist.globalSize()
         << ", local size= " << m_vdist.localSize();

  sycl::buffer<Integer,1> cell_lids_buffer(cell_lids.data(),sycl::range(cell_lids.size())) ;
  sycl::buffer<Integer,1> allUIndex_buffer(allUIndex.data(),sycl::range(allUIndex.size())) ;
  sycl::buffer<Integer,1> cell_conn_offset_buffer(m_cell_cell_connection_offset.data(),sycl::range(m_cell_cell_connection_offset.size())) ;
  sycl::buffer<Integer,1> cell_conn_index_buffer(m_cell_cell_connection_index.data(),sycl::range(m_cell_cell_connection_index.size())) ;
  sycl::buffer<Integer,1> cell_conn_lids_buffer(cell_conn_lids.data(),sycl::range(cell_conn_lids.size())) ;
  sycl::buffer<Integer,1> all_cell_conn_lids_buffer(all_cell_conn_lids.data(),sycl::range(all_cell_conn_lids.size())) ;
  sycl::buffer<Int16,1>   cell_is_own_buffer(m_cell_is_own.asArray().data(),sycl::range(m_cell_is_own.asArray().size())) ;
  sycl::buffer<double,1>  u_buffer(m_u.asArray().data(),sycl::range(m_u.asArray().size())) ;
  sycl::buffer<double,1>  k_buffer(m_k.asArray().data(),sycl::range(m_k.asArray().size())) ;
  sycl::buffer<double,1>  s_buffer(m_s.asArray().data(),sycl::range(m_s.asArray().size())) ;
  sycl::buffer<Real3,1>   cell_center_buffer(m_cell_center.asArray().data(),sycl::range(m_cell_center.asArray().size())) ;

  Computer computer{m_homogeneous,
                    m_diag_coeff,
                    m_off_diag_coeff,
                    m_lambdax,
                    m_lambday,
                    m_lambdaz,
                    m_alpha,
                    m_sigma} ;
  {
    ENUMERATE_CELL (icell, areaU)
    {
      Real3 x;
      for (Arcane::Node node : icell->nodes()) {
        x += m_node_coord[node];
      }
      x /= icell->nbNode();
      m_cell_center[icell] = x;
      if(icell->isOwn())
        m_cell_is_own[icell] = 1 ;
      else
        m_cell_is_own[icell] = 0 ;
    }

    Alien::ParallelEngine engine(*m_default_queue) ;
    engine.submit([&](ControlGroupHandler& handler)
                  {
                    auto& command = handler.command() ;
                    /*
                    auto in_cell_lids = ax::viewIn(command,cell_lids);
                    auto in_center = ax::viewIn(command,m_cell_center);
                    auto out_u = ax::viewOut(command,m_u);
                    auto out_k = ax::viewOut(command,m_k);
                    */
                    auto in_cell_lids = cell_lids_buffer.get_access<sycl::access::mode::read>(handler.m_internal) ;
                    auto in_center = cell_center_buffer.get_access<sycl::access::mode::read>(handler.m_internal) ;
                    auto out_u = u_buffer.get_access<sycl::access::mode::read_write>(handler.m_internal) ;
                    auto out_k = k_buffer.get_access<sycl::access::mode::read_write>(handler.m_internal) ;

                    auto local_size = cell_lids.size() ;
                    handler.parallel_for(engine.maxNumThreads(),
                                         [=](Alien::ParallelEngine::Item<1>::type item)
                                         {
                                            auto id = item.get_id(0) ;
                                            for (auto index = id; index < local_size; index += item.get_range()[0])
                                            {
                                              //auto vi   = CellLocalId(in_cell_lids[index]) ;
                                              auto vi = in_cell_lids[index] ;
                                              auto x    = in_center[vi] ;
                                              out_u[vi] = computer.funcn(x) ;
                                              out_k[vi] = computer.funck(x) ;
                                            }
                                          });
                  }) ;
  }

  {
    auto vx_acc = Alien::SYCL::VectorAccessorT<Real>(vectorX);
    auto cx_acc = Alien::SYCL::VectorAccessorT<Real> (coordX);
    auto cy_acc = Alien::SYCL::VectorAccessorT<Real> (coordY);
    auto cz_acc = Alien::SYCL::VectorAccessorT<Real> (coordZ);

    Alien::ParallelEngine engine(*m_default_queue) ;

    engine.submit([&](ControlGroupHandler& handler)
                  {
                    auto& command         = handler.command() ;
                    /*
                    auto in_cell_lids     = ax::viewIn(command,cell_lids);
                    auto in_center        = ax::viewIn(command,m_cell_center);
                    auto in_is_own        = ax::viewIn(command,m_cell_is_own);
                    auto in_allUIndex     = ax::viewIn(command,accAllUIndex) ;
                    auto in_u             = ax::viewIn(command,m_u);
                    */

                    auto in_cell_lids = cell_lids_buffer.get_access<sycl::access::mode::read>(handler.m_internal) ;
                    auto in_allUIndex = allUIndex_buffer.get_access<sycl::access::mode::read>(handler.m_internal) ;
                    auto in_is_own = cell_is_own_buffer.get_access<sycl::access::mode::read>(handler.m_internal) ;
                    auto in_center = cell_center_buffer.get_access<sycl::access::mode::read>(handler.m_internal) ;
                    auto in_u = u_buffer.get_access<sycl::access::mode::read>(handler.m_internal) ;

                    auto out_vx = vx_acc.view(handler) ;
                    auto out_cx = cx_acc.view(handler) ;
                    auto out_cy = cy_acc.view(handler) ;
                    auto out_cz = cz_acc.view(handler) ;

                    //auto local_size = own_cell_lids.size() ;
                    auto local_size = cell_lids.size() ;

                    //command << RUNCOMMAND_ENUMERATE(Cell,vi,areaU.own())

                    handler.parallel_for(engine.maxNumThreads(),
                                         [=](Alien::ParallelEngine::Item<1>::type item)
                                         {
                                            auto id = item.get_id(0) ;
                                            for (auto index = id; index < local_size; index += item.get_range()[0])
                                            {
                                              //auto vi   = CellLocalId(in_cell_lids[index]) ;
                                              auto vi     = in_cell_lids[index] ;
                                              auto iIndex = in_allUIndex[vi];
                                              auto is_own = in_is_own[vi] ;
                                              auto xC     = in_center[vi] ;
                                              if(iIndex!=-1 && is_own==1)
                                              {
                                                out_vx[iIndex] = in_u[vi] ;
                                                out_cx[iIndex] = xC.x ;
                                                out_cy[iIndex] = xC.y ;
                                                out_cz[iIndex] = xC.z ;
                                              }
                                            } ;
                                         }) ;

                  }) ;
  }

  ///////////////////////////////////////////////////////////////////////////
  //
  // MATRIX BUILDING AND FILLING
  //
  {
      Timer::Sentry ts(&pbuild_timer);
      {
        Alien::SYCL::MatrixProfiler profiler(matrixA);
        ///////////////////////////////////////////////////////////////////////////
        //
        // DEFINE PROFILE
        //
        ENUMERATE_ITEMPAIR(Cell, Cell, icell, cell_cell_connection)
        {
          const Cell& cell = *icell;
          const Integer iIndex = allUIndex[cell.localId()];
          profiler.addMatrixEntry(iIndex, iIndex);
          ENUMERATE_SUB_ITEM(Cell, isubcell, icell)
          {
            const Cell& subcell = *isubcell;
            profiler.addMatrixEntry(iIndex, allUIndex[subcell.localId()]);
          }
        }
      }

      {
        Alien::SYCL::CombineMultProfiledMatrixBuilder builder(matrixA, Alien::ProfiledMatrixOptions::eResetValues);
        builder.setParallelAssembleStencil(1,m_cell_cell_connection_offset.view(),m_cell_cell_connection_index.view()) ;

        Alien::ParallelEngine engine(*m_default_queue) ;
        engine.submit([&](ControlGroupHandler& handler)
                      {
                        auto& command = handler.command() ;
                        /*
                        auto in_allUIndex      = ax::viewIn(command,accAllUIndex) ;
                        auto in_conn_index     = ax::viewIn(command,m_cell_cell_connection_index) ;
                        auto in_conn_offset    = ax::viewIn(command,m_cell_cell_connection_offset) ;
                        auto in_cell_conn_lids = ax::viewIn(command,cell_conn_lids) ;
                        auto in_center         = ax::viewIn(command,m_cell_center);
                        auto in_is_own         = ax::viewIn(command,m_cell_is_own);
                        */

                        //auto in_cell_lids      = cell_lids_buffer.get_access<sycl::access::mode::read>(handler.m_internal) ;
                        auto in_conn_index     = cell_conn_index_buffer.get_access<sycl::access::mode::read>(handler.m_internal) ;
                        auto in_conn_offset    = cell_conn_offset_buffer.get_access<sycl::access::mode::read>(handler.m_internal) ;
                        auto in_cell_conn_lids = cell_conn_lids_buffer.get_access<sycl::access::mode::read>(handler.m_internal) ;
                        auto in_allUIndex      = allUIndex_buffer.get_access<sycl::access::mode::read>(handler.m_internal) ;

                        auto in_is_own = cell_is_own_buffer.get_access<sycl::access::mode::read>(handler.m_internal) ;
                        auto in_center = cell_center_buffer.get_access<sycl::access::mode::read>(handler.m_internal) ;

                        auto matrix_acc = builder.view(handler) ;
                        auto local_size = cell_cell_connection.itemGroup().size() ;
                        //command << RUNCOMMAND_ENUMERATE(Cell,vi,CellGroup(cell_cell_connection.itemGroup()))
                        //command << RUNCOMMAND_LOOP1(iter,cell_cell_connection.itemGroup().size())

                        handler.parallel_for(engine.maxNumThreads(),
                                             [=](Alien::ParallelEngine::Item<1>::type item)
                                             {
                                                auto id = item.get_id(0) ;
                                                for (auto index = id; index < local_size; index += item.get_range()[0])
                                                {
                                                  auto vi = in_cell_conn_lids[index] ;
                                                  //auto vi = CellLocalId(in_cell_conn_lids[index]) ;
                                                  double diag = computer.dii(vi);

                                                  auto xi = in_center[vi] ;

                                                  Integer i = in_allUIndex[vi];
                                                  auto eii = matrix_acc.entryIndex(i, i) ;
                                                  matrix_acc[eii] += diag;

                                                  for(auto k=in_conn_offset[index];k<in_conn_offset[index+1];++k)
                                                  {
                                                    auto svj = in_conn_index[k] ;
                                                    //auto svj = CellLocalId(slid) ;
                                                    auto xj = in_center[svj] ;
                                                    Integer j = in_allUIndex[svj];
                                                    auto eij = matrix_acc.entryIndex(i, j) ;

                                                    double off_diag = computer.fij(vi, svj,xi,xj);
                                                    matrix_acc[eii] += off_diag;
                                                    matrix_acc[eij] -= off_diag;
                                                  }
                                                } ;
                                              }) ;
                      }) ;
        /*
        {
          auto hview = builder.hostView();
          for(std::size_t index=0;index<cell_lids.size();++index)
          {
              auto i = allUIndex[cell_lids[index]] ;
              auto eii = hview.entryIndex(i,i) ;
              std::cout <<" ROW ["<<index<<"]: DIAG("<<cell_lids[index]<<","<<i<<","<<eii<<","<<hview[eii]<<") ";
              for(std::size_t k=m_cell_cell_connection_offset[index];k<m_cell_cell_connection_offset[index+1];++k)
              {
                  auto jindex = m_cell_cell_connection_index[k] ;
                  auto j =  allUIndex[cell_lids[jindex]] ;
                  auto eij =  hview.entryIndex(i,j) ;
                std::cout <<"("<<cell_lids[jindex]<<","<<j<<","<<eij<<","<<hview[eij]<<")";
              }
              std::cout<<std::endl ;
          }
        }*/
        if (options()->sigma() > 0.)
        {
          m_sigma = options()->sigma();
          Arcane::Real3 xCmax { 0.25, 0.25, 0.25 };
          Arcane::Real3 xCmin { 0.75, 0.75, 0.55 };
          auto epsilon = options()->epsilon() ;
          auto sigma = m_sigma ;
          engine.submit([&](ControlGroupHandler& handler)
                        {
                          auto& command = handler.command() ;
                          /*
                          auto in_cell_center = ax::viewIn(command,m_cell_center);
                          auto in_cell_is_own = ax::viewIn(command,m_cell_is_own);
                          auto out_s   = ax::viewOut(command,m_s);
                          auto in_conn_index = ax::viewIn(command,m_cell_cell_connection_index) ;
                          auto in_conn_offset = ax::viewIn(command,m_cell_cell_connection_offset) ;
                          auto in_cell_conn_lids = viewIn(command,cell_conn_lids) ;
                          auto in_allUIndex = ax::viewIn(command,accAllUIndex) ;
                          */

                          //auto in_cell_lids      = cell_lids_buffer.get_access<sycl::access::mode::read>(handler.m_internal) ;
                          auto in_conn_index     = cell_conn_index_buffer.get_access<sycl::access::mode::read>(handler.m_internal) ;
                          auto in_conn_offset    = cell_conn_offset_buffer.get_access<sycl::access::mode::read>(handler.m_internal) ;
                          auto in_cell_conn_lids = all_cell_conn_lids_buffer.get_access<sycl::access::mode::read>(handler.m_internal) ;

                          auto in_allUIndex      = allUIndex_buffer.get_access<sycl::access::mode::read>(handler.m_internal) ;

                          auto in_is_own = cell_is_own_buffer.get_access<sycl::access::mode::read>(handler.m_internal) ;
                          auto in_center = cell_center_buffer.get_access<sycl::access::mode::read>(handler.m_internal) ;
                          auto out_s     = s_buffer.get_access<sycl::access::mode::read_write>(handler.m_internal) ;

                          auto matrix_acc = builder.view(handler) ;
                          //command << RUNCOMMAND_ENUMERATE(Cell,vi,CellGroup(cell_cell_connection.itemGroup()))
                          //command << RUNCOMMAND_LOOP1(iter,cell_cell_connection.itemGroup().size())

                          auto local_size = all_cell_cell_connection.itemGroup().size() ;
                          handler.parallel_for(engine.maxNumThreads(),
                                               [=](Alien::ParallelEngine::Item<1>::type item)
                                               {
                                                  auto id = item.get_id(0) ;
                                                  for (auto index = id; index < local_size; index += item.get_range()[0])
                                                  {
                                                    auto vi = in_cell_conn_lids[index] ;

                                                    auto xC = in_center[vi];
                                                    auto xDmax = xC - xCmax;
                                                    auto xDmin = xC - xCmin;
                                                    out_s[vi] = 0.;
                                                    Integer i = in_allUIndex[vi];
                                                    auto eii = matrix_acc.entryIndex(i, i) ;
                                                    if (xDmax.normL2() < epsilon) {
                                                      out_s[vi] = 1.;
                                                      if (in_is_own[vi]==1)
                                                        matrix_acc[eii] = sigma;
                                                      for(auto k = in_conn_offset[index];k<in_conn_offset[index+1];++k)
                                                      {
                                                        auto svj = in_conn_index[k] ;
                                                        if(in_is_own[svj]==1) {
                                                            Integer j = in_allUIndex[svj];
                                                            auto ejik = matrix_acc.combineEntryIndex(i, j, i) ;
                                                            //matrix_acc[eji] = 0.;
                                                            matrix_acc.combine(ejik,0.) ;
                                                        }
                                                      }
                                                    }
                                                    if (xDmin.normL2() < epsilon) {
                                                      out_s[vi] = -1.;
                                                      Integer i = in_allUIndex[vi];

                                                      if (in_is_own[vi])
                                                        matrix_acc[eii] = 1. / sigma;
                                                      for(auto k = in_conn_offset[index];k<in_conn_offset[index+1];++k)
                                                      {
                                                        auto svj = in_conn_index[k] ;
                                                        if(in_is_own[svj]==1) {
                                                            Integer j = in_allUIndex[svj];
                                                            auto ejik = matrix_acc.combineEntryIndex(i, j, i) ;
                                                            //matrix_acc[eji] = 0.;
                                                            matrix_acc.combine(ejik,0.) ;
                                                        }
                                                      }
                                                    }
                                                  } ;
                                               }) ;
                        }) ;
            builder.combine() ;
          }
        builder.finalize() ;
      }
  }
  
  {
    info()<<"COMPUTE SYCL NORME B";
    std::cout << "COMPUTE SYCL NORME B"<<std::endl ;
    Alien::SYCLLinearAlgebra syclAlg;
    syclAlg.mult(matrixA, vectorX, vectorB);
    syclAlg.mult(matrixA, vectorX, vectorBB);
    Real normeb = syclAlg.norm2(vectorB);
    std::cout << "||b||=" << normeb<<std::endl ;
    {
      SYCLInternalLinearAlgebra alg ;

      auto&  sycl_x = vectorX.impl()->get<Alien::BackEnd::tag::sycl>(true) ;
      alg.assign(sycl_x,0.) ;
    }
  }
}
/*---------------------------------------------------------------------------*/
