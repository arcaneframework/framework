#include <mpi.h>

#include <string>
#include <map>
#include <time.h>
#include <vector>
#include <fstream>
#include <boost/timer.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/cmdline.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/timer.hpp>
#include <boost/lexical_cast.hpp>

#include <arcane/ArcaneVersion.h>
#include <arcane/Timer.h>
#include <arcane/ItemPairGroup.h>
#include <arcane/mesh/ItemFamily.h>
#include <arcane/utils/PlatformUtils.h>
#include <arcane/utils/IMemoryInfo.h>
#include <arcane/utils/OStringStream.h>
#include <arcane/ITimeLoopMng.h>
#include <alien/arcane_tools/accessors/ItemVectorAccessor.h>
#include <alien/core/block/VBlock.h>

#include <alien/arcane_tools/IIndexManager.h>
#include <alien/arcane_tools/indexManager/BasicIndexManager.h>
#include <alien/arcane_tools/indexManager/SimpleAbstractFamily.h>
#include <alien/arcane_tools/distribution/DistributionFabric.h>
#include <alien/arcane_tools/indexSet/IndexSetFabric.h>

#include <alien/kernels/simple_csr/algebra/SimpleCSRLinearAlgebra.h>


#include <alien/AlienRefSemantic.h>

#include <alien/kernels/redistributor/Redistributor.h>
#include <alien/data/scalar/RedistributedVector.h>
#include <alien/data/scalar/RedistributedMatrix.h>

#ifdef ALIEN_USE_PETSC
#include <alien/Kernels/PETSc/IO/AsciiDumper.h>
#include <alien/Kernels/PETSc/Algebra/PETScLinearAlgebra.h>
#endif
#ifdef ALIEN_USE_MTL4
#include <alien/Kernels/MTL/Algebra/MTLLinearAlgebra.h>
#endif
#ifdef ALIEN_USE_HTSSOLVER
#include <alien/Kernels/HTS/HTSBackEnd.h>
#include <alien/Kernels/HTS/DataStructure/HTSMatrix.h>
#include <alien/Kernels/HTS/Algebra/HTSLinearAlgebra.h>
#endif
#ifdef ALIEN_USE_TRILINOS
#include <alien/Kernels/Trilinos/TrilinosBackEnd.h>
#include <alien/Kernels/Trilinos/DataStructure/TrilinosMatrix.h>
#include <alien/Kernels/Trilinos/Algebra/TrilinosLinearAlgebra.h>
#endif

#include <alien/expression/solver/ILinearSolver.h>

#include "AlienBenchModule.h"

#include <arcane/ItemPairGroup.h>
#include <arcane/IMesh.h>

#include <alien/core/impl/MultiVectorImpl.h>

using namespace Arcane;
using namespace Alien;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AlienBenchModule::init()
{
  Alien::setTraceMng(traceMng());
  Alien::setVerbosityLevel(Alien::Verbosity::Debug);
  m_parallel_mng = subDomain()->parallelMng();
  m_homogeneous = options()->homogeneous() ;
  m_diag_coeff  = options()->diagonalCoefficient();
  m_lambdax     = options()->lambdax() ;
  m_lambday     = options()->lambday() ;
  m_lambdaz     = options()->lambdaz() ;
  m_alpha       = options()->alpha() ;

  Alien::ILinearSolver * solver = options()->linearSolver();
  solver->init() ;
}

/*---------------------------------------------------------------------------*/

void
AlienBenchModule::
test()
{
  Timer pbuild_timer(subDomain(),"PBuildPhase",Timer::TimerReal);
  Timer psolve_timer(subDomain(),"PSolvePhase",Timer::TimerReal);
  Timer rbuild_timer(subDomain(),"RBuildPhase",Timer::TimerReal);
  Timer rsolve_timer(subDomain(),"RSolvePhase",Timer::TimerReal);

  ItemGroup areaU = allCells();
  CellCellGroup cell_cell_connection(areaU.own(),areaU,m_stencil_kind);
  CellCellGroup all_cell_cell_connection(areaU,areaU,m_stencil_kind);

  Alien::ArcaneTools::BasicIndexManager index_manager(m_parallel_mng);
  index_manager.setTraceMng(traceMng());

  auto indexSetU = index_manager.buildScalarIndexSet("U", areaU);
  index_manager.prepare();

  ///////////////////////////////////////////////////////////////////////////
  //
  // CREATE Space FROM IndexManger
  // CREATE MATRIX ASSOCIATED TO Space
  // CREATE VECTORS ASSOCIATED TO Space
  //

  // Accès à l'indexation
  Arccore::UniqueArray<Arccore::Integer> allUIndex = index_manager.getIndexes(indexSetU);

  m_mdist = Alien::ArcaneTools::createMatrixDistribution(&index_manager, parallelMng());
  m_vdist = Alien::ArcaneTools::createVectorDistribution(&index_manager, parallelMng());

  info()<<"GLOBAL SIZE : "<<m_vdist.globalSize();
  Alien::Space space(m_vdist.globalSize(), "TestSpace");


  Alien::Vector vectorB(m_vdist);
  Alien::Vector vectorBB(m_vdist);
  Alien::Vector vectorX(m_vdist);

  ///////////////////////////////////////////////////////////////////////////
  //
  // VECTOR BUILDING AND FILLING
  //
  info() << "Building & initializing vector b";
  info() << "Space size = " << m_vdist.globalSize() << ", local size= " << m_vdist.localSize();

  ENUMERATE_CELL(icell,areaU)
  {
    Real3 x ;
    ENUMERATE_NODE(inode,icell->nodes())
    {
      x += m_node_coord[*inode] ;
    }
    x /= icell->nbNode() ;
    m_cell_center[icell] = x ;
    m_u[icell] = funcn(x) ;
    m_k[icell] = funck(x) ;
  }
  {
    // Builder du vecteur
    Alien::VectorWriter writer(vectorX);
    ENUMERATE_CELL(icell,areaU.own())
    {
      const Integer iIndex = allUIndex[icell->localId()];
      writer[iIndex] = m_u[icell] ;
    }
  }



  Alien::Matrix matrixA(m_mdist); // local matrix for exact measure without side effect (however, you can reuse a matrix with several builder)


    ///////////////////////////////////////////////////////////////////////////
    //
    // MATRIX BUILDING AND FILLING
    //
    {
      Timer::Sentry ts(&pbuild_timer);
      Alien::MatrixProfiler profiler(matrixA);
      ///////////////////////////////////////////////////////////////////////////
      //
      // DEFINE PROFILE
      //
      ENUMERATE_ITEMPAIR(Cell,Cell,icell,cell_cell_connection)
      {
        const Cell & cell = *icell;
        const Integer iIndex = allUIndex[cell.localId()];
        profiler.addMatrixEntry(iIndex, allUIndex[cell.localId()]);
        ENUMERATE_SUB_ITEM(Cell,isubcell,icell)
        {
          const Cell& subcell = *isubcell;
          profiler.addMatrixEntry(iIndex,allUIndex[subcell.localId()]);
        }
      }
    }
    {
      Timer::Sentry ts(&pbuild_timer);
      Alien::ProfiledMatrixBuilder builder(matrixA, Alien::ProfiledMatrixOptions::eResetValues);
      ENUMERATE_ITEMPAIR(Cell, Cell, icell, cell_cell_connection)
      {
          const Cell & cell = *icell;
          double diag = dii(cell) ;

          Integer i = allUIndex[cell.localId()];
          builder(i,i) += diag;
          ENUMERATE_SUB_ITEM(Cell, isubcell, icell)
          {
            const Cell& subcell = *isubcell;
            double off_diag = fij(cell,subcell) ;
            builder(i,i) += off_diag;
            Integer j = allUIndex[subcell.localId()];
            builder(i,j) -= off_diag;
          }
      }
      if(options()->sigma()>0.)
      {
        m_sigma = options()->sigma() ;
        auto xCmax = Real3{0.25,0.25,0.25} ;
        auto xCmin = Real3{0.75,0.75,0.55} ;
        ENUMERATE_ITEMPAIR(Cell, Cell, icell, all_cell_cell_connection)
        {
            const Cell & cell = *icell;
            Real3 xC = m_cell_center[icell] ;
            Real3 xDmax = xC - xCmax ;
            Real3 xDmin = xC - xCmin ;
            m_s[cell] = 0 ;
            if(xDmax.abs() < options()->epsilon())
            {
              m_s[cell] = 1. ;
              Integer i = allUIndex[cell.localId()];
              info()<<"MATRIX TRANSFO SIGMAMAX "<<i;
              if(cell.isOwn())
                builder(i,i) = m_sigma ;
              ENUMERATE_SUB_ITEM(Cell, isubcell, icell)
              {
                const Cell& subcell = *isubcell;
                if(subcell.isOwn())
                {
                  Integer j = allUIndex[subcell.localId()];
                  builder(j,i) = 0.;
                }
              }
            }
            if(xDmin.abs() < options()->epsilon())
            {
              m_s[cell] = -1. ;
              Integer i = allUIndex[cell.localId()];
              info()<<"MATRIX TRANSFO SIGMA MIN"<<i;

              if(cell.isOwn())
                builder(i,i) = 1./m_sigma ;
              ENUMERATE_SUB_ITEM(Cell, isubcell, icell)
              {
                const Cell& subcell = *isubcell;
                if(subcell.isOwn())
                {
                  Integer j = allUIndex[subcell.localId()];
                  builder(j,i) = 0.;
                }
              }
            }
          }
      }

      builder.finalize();
    }

    {
      Alien::SimpleCSRLinearAlgebra csrAlg;
      csrAlg.mult(matrixA,vectorX,vectorB);
      csrAlg.mult(matrixA,vectorX,vectorBB);
      Real normeb = csrAlg.norm2(vectorB) ;
      info()<<"||b||="<<normeb;
    }
#ifdef ALIEN_USE_HTSSOLVER
    /*{
      info()<<"HTS";
      Alien::HTSLinearAlgebra htsAlg;
      htsAlg.mult(matrixA,vectorX,vectorB);
      htsAlg.mult(matrixA,vectorX,vectorBB);
      Real normeb = htsAlg.norm2(vectorB) ;
      info()<<"||b||="<<normeb;
    }*/
#endif
#ifdef ALIEN_USE_TRILINOS
    {
      info()<<"Trilinos";
      Alien::TrilinosLinearAlgebra tpetraAlg;
      tpetraAlg.mult(matrixA,vectorX,vectorB);
      tpetraAlg.mult(matrixA,vectorX,vectorBB);
      Real normeb = tpetraAlg.norm2(vectorB) ;
      info()<<"||b||="<<normeb;
      //tpetraAlg.dump(matrixA,"MatrixA.txt") ;
      //tpetraAlg.dump(vectorB,"vectorB.txt") ;
      //tpetraAlg.dump(vectorBB,"vectorBB.txt") ;
      //tpetraAlg.dump(vectorX,"vectorX.txt") ;
    }
#endif

    if(options()->unitRhs())
    {
        Alien::LocalVectorWriter v(vectorBB);
        for(Integer i=0;i<v.size();++i)
          v[i] = 1.;
    }

    if(options()->zeroRhs())
    {
        Alien::LocalVectorWriter v(vectorB);
        for(Integer i=0;i<v.size();++i)
          v[i] = 0.;
    }

    ///////////////////////////////////////////////////////////////////////////
    //
    // RESOLUTION
    //
    {
      Alien::ILinearSolver * solver = options()->linearSolver();
      solver->init() ;




      if (not solver->hasParallelSupport() and m_parallel_mng->commSize() > 1)
      {
        info() << "Current solver has not a parallel support for solving linear system : skip it";
      }
      else
      {
        Integer nb_resolutions = options()->nbResolutions() ;
        for(Integer i=0;i<nb_resolutions;++i)
        {
          if(i>0)  // i=0, matrix allready filled
          {
            Timer::Sentry ts(&pbuild_timer);
            Alien::ProfiledMatrixBuilder builder(matrixA, Alien::ProfiledMatrixOptions::eResetValues);
            ENUMERATE_ITEMPAIR(Cell, Cell, icell, cell_cell_connection)
            {
                const Cell & cell = *icell;
                double diag = dii(cell) ;

                Integer i = allUIndex[cell.localId()];
                builder(i,i) += diag;
                ENUMERATE_SUB_ITEM(Cell, isubcell, icell)
                {
                  const Cell& subcell = *isubcell;
                  double off_diag = fij(cell,subcell) ;
                  builder(i,i) += off_diag;
                  Integer j = allUIndex[subcell.localId()];
                  builder(i,j) -= off_diag;
                }
            }
            if(options()->sigma()>0.)
            {
              m_sigma = options()->sigma() ;
              auto xCmax = Real3{0.25,0.25,0.25} ;
              auto xCmin = Real3{0.75,0.75,0.55} ;
              ENUMERATE_ITEMPAIR(Cell, Cell, icell, all_cell_cell_connection)
              {
                  const Cell & cell = *icell;
                  Real3 xC = m_cell_center[icell] ;
                  Real3 xDmax = xC - xCmax ;
                  Real3 xDmin = xC - xCmin ;
                  m_s[cell] = 0 ;
                  if(xDmax.abs() < options()->epsilon())
                  {
                    m_s[cell] = 1. ;
                    Integer i = allUIndex[cell.localId()];
                    info()<<"MATRIX TRANSFO SIGMAMAX "<<i;
                    if(cell.isOwn())
                      builder(i,i) = m_sigma ;
                    ENUMERATE_SUB_ITEM(Cell, isubcell, icell)
                    {
                      const Cell& subcell = *isubcell;
                      if(subcell.isOwn())
                      {
                        Integer j = allUIndex[subcell.localId()];
                        builder(j,i) = 0.;
                      }
                    }
                  }
                  if(xDmin.abs() < options()->epsilon())
                  {
                    m_s[cell] = -1. ;
                    Integer i = allUIndex[cell.localId()];
                    info()<<"MATRIX TRANSFO SIGMA MIN"<<i;

                    if(cell.isOwn())
                      builder(i,i) = 1./m_sigma ;
                    ENUMERATE_SUB_ITEM(Cell, isubcell, icell)
                    {
                      const Cell& subcell = *isubcell;
                      if(subcell.isOwn())
                      {
                        Integer j = allUIndex[subcell.localId()];
                        builder(j,i) = 0.;
                      }
                    }
                  }
                }
            }

            builder.finalize();
          }

          // Réinitialisation de vectorX
          //if(i>0)  // i=0, vector allready filled
          {
            Alien::LocalVectorReader reader(vectorBB);
            Alien::LocalVectorWriter vb(vectorB);
            Alien::LocalVectorWriter vx(vectorX);
            for(Integer i=0;i<m_vdist.localSize();++i)
            {
              vx[i] = 0. ;
              vb[i] = reader[i];
            }
          }

          Timer::Sentry ts(&psolve_timer);
          solver->solve(matrixA,vectorB,vectorX) ;
        }
        Alien::SolverStatus status = solver->getStatus();
        if(status.succeeded)
        {

          Alien::VectorReader reader(vectorX);
          ENUMERATE_CELL(icell,areaU.own())
          {
            const Integer iIndex = allUIndex[icell->localId()];
            m_x[icell] = reader[iIndex] ;
          }

          SimpleCSRLinearAlgebra alg;
          Alien::Vector vectorR(m_vdist);
          alg.mult(matrixA,vectorX,vectorR);
          alg.axpy(-1.,vectorB,vectorR) ;
          Real res = alg.norm2(vectorR) ;
          info()<<"RES : "<<res;
        }
        solver->getSolverStat().print(Universe().traceMng(), status, "Linear Solver : ") ;
      }
      solver->end() ;
    }

    if(options()->redistribution() && m_parallel_mng->commSize()>1)
    {
      info()<<"Test REDISTRIBUTION";

      {
        Alien::LocalVectorWriter v(vectorX);
        for(Integer i=0;i<v.size();++i)
          v[i] = 0;
      }
      Alien::Vector vectorR(m_vdist);

      bool keep_proc = false ;
      if(m_parallel_mng->commRank()==0)
        keep_proc=true ;

      rbuild_timer.start();
      Alien::Redistributor redist(matrixA.distribution().globalRowSize(), m_parallel_mng->messagePassingMng(), keep_proc);

      Alien::RedistributedMatrix Aa(matrixA, redist);
      Alien::RedistributedVector bb(vectorB, redist);
      Alien::RedistributedVector xx(vectorX, redist);
      Alien::RedistributedVector rr(vectorR, redist);
      rbuild_timer.stop() ;
      if(keep_proc)
      {
        auto solver = options()->linearSolver() ;
        //solver->updateParallelMng(Aa.distribution().parallelMng());
        solver->init();
        {
          Timer::Sentry ts(&rsolve_timer);
          solver->solve(Aa,bb,xx);
        }

        Alien::SimpleCSRLinearAlgebra alg;

        alg.mult(Aa,xx,rr);
        alg.axpy(-1.,bb,rr) ;
        Real res = alg.norm2(rr) ;
        info()<<"REDISTRIBUTION RES : "<<res;
      }

    }
  info()<<"===================================================";
  info()<<"BENCH INFO :";
  info()<<" PBUILD    :"<<pbuild_timer.totalTime();
  info()<<" PSOLVE    :"<<psolve_timer.totalTime();
  info()<<" RBUILD    :"<<rbuild_timer.totalTime();
  info()<<" RSOLVE    :"<<rsolve_timer.totalTime();
  info()<<"===================================================";

  subDomain()->timeLoopMng()->stopComputeLoop(true);
}

/*---------------------------------------------------------------------------*/

Real
AlienBenchModule::
funcn(Real3 p) const
{
  return p.x * p.x * p.y;
}

Real
AlienBenchModule::
funck(Real3 p) const
{
#define PI 3.14159265358979323846264

  if(m_homogeneous)
    return m_off_diag_coeff ;
  else
    return std::exp(-m_alpha*0.5*(1+std::sin(2*PI*p.x/m_lambdax))*(1+std::sin(2*PI*p.y/m_lambday)));
}


Real
AlienBenchModule::
dii(const Cell & ci) const
{
    return m_diag_coeff;
}

Real
AlienBenchModule::
fij(const Cell & ci, const Cell & cj) const
{
  if (ci == cj)
    return dii(ci);
  else
  {
    Real3 xi = m_cell_center[ci] ;
    Real3 xj = m_cell_center[cj] ;
    Real3 xij = xi + xj ;
    xij /= 2. ;

    return funck(xij);
  }
}

/*---------------------------------------------------------------------------*/


ARCANE_REGISTER_MODULE_ALIENBENCH(AlienBenchModule);
