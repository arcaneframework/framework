// -*- C++ -*-
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#define MPICH_SKIP_MPICXX 1
#include "mpi.h"

#include <vector>
#include <list>

#include "alien/kernels/composyx/ComposyxPrecomp.h"

#include <alien/data/Space.h>
#include <alien/expression/solver/ILinearSolver.h>
#include <alien/expression/solver/ILinearAlgebra.h>
#include <alien/expression/solver/SolverStat.h>
#include <alien/expression/solver/SolverStater.h>

#include <alien/kernels/simple_csr/SimpleCSRPrecomp.h>
#include <alien/kernels/simple_csr/algebra/SimpleCSRLinearAlgebra.h>
#include <alien/kernels/simple_csr/SimpleCSRMatrix.h>

//#include <alien/kernels/composyx/algebra/ComposyxLinearAlgebra.h>
#include <alien/core/impl/MultiMatrixImpl.h>
#include <alien/core/impl/MultiVectorImpl.h>

//#include <alien/kernels/composyx/algebra/ComposyxLinearAlgebra.h>
//#include <alien/kernels/composyx/linear_solver/ComposyxInternalLinearSolver.h>
#include <alien/core/backend/LinearSolverT.h>
#include <alien/core/block/ComputeBlockOffsets.h>
#include <ALIEN/axl/ComposyxSolver_IOptions.h>

#include <alien/kernels/composyx/data_structure/ComposyxMatrixImplT.h>
#include <alien/kernels/composyx/data_structure/ComposyxVectorImplT.h>

#include <alien/kernels/composyx/linear_solver/ComposyxInternalSolver.h>
/*---------------------------------------------------------------------------*/



BEGIN_COMPOSYXINTERNAL_NAMESPACE

bool ComposyxEnv::m_is_initialized = false;
ComposyxEnv* ComposyxEnv::m_instance = nullptr ;

template <>
int
ComposyxEnv::getEnv(std::string const& key, int default_value)
{
  const char* arch_str = ::getenv(key.c_str());
  if (arch_str)
    return atoi(arch_str);
  return default_value;
}

void
ComposyxEnv::initialize(IMessagePassingMng* parallel_mng)
{
  if (m_is_initialized)
    return;
#ifdef ALIEN_USE_COMPOSYX
  composyx::MMPI::init();
#endif
  m_is_initialized = true;

  if(m_instance)
    return ;

  m_instance = new ComposyxEnv ;
  m_instance->init(parallel_mng) ;
  m_is_initialized = true ;
}

void
ComposyxEnv::finalize()
{
  delete m_instance ;
  m_instance = nullptr ;
#ifdef ALIEN_USE_COMPOSYX
  composyx::MMPI::finalize();
#endif
  m_is_initialized = false;
}

//template class MatrixInternal<double>;

END_COMPOSYXINTERNAL_NAMESPACE


namespace Alien {


/*---------------------------------------------------------------------------*/
ComposyxInternalSolver::ComposyxInternalSolver(
    IMessagePassingMng* parallel_mng, IOptionsComposyxSolver* options)
: m_parallel_mng(parallel_mng)
, m_options(options)
, m_stater(this)
{
  ComposyxInternal::ComposyxEnv::initialize(parallel_mng) ;
}

ComposyxInternalSolver::~ComposyxInternalSolver()
{

}

void
ComposyxInternalSolver::init(int argc, char const** argv)
{
}

void
ComposyxInternalSolver::init()
{
  alien_info([&] { cout() << "ComposyxSolver::init"; });

  m_output_level = m_options->outputLevel();
  bool verbose = (m_parallel_mng->commRank()== 0) &&  m_output_level>0 ;
  m_internal.reset(new InternalType()) ;
  m_internal->init(m_options->maxIterationNum(),
                   m_options->stopCriteriaValue(),
                   m_options->solver().localstr(),
                   m_options->preconditioner().localstr(),
                   verbose) ;
}

void
ComposyxInternalSolver::end()
{
  if (m_output_level > 0)
    internalPrintInfo();
  m_internal.reset() ;
}

#ifdef ALIEN_USE_COMPOSYX


bool
ComposyxInternalSolver::solve( ComposyxMatrixType const& A,
                               ComposyxVectorType const& b,
                               ComposyxVectorType& x)
{
  if (m_output_level > 0)
    alien_info([&] { cout() << "ComposyxSolver::solve"; });

  {
    Alien::BaseSolverStater::Sentry s(m_init_solver_time);
    if(A.isParallel())
    {
        x.internal()->init(b.internal()->parallelMng(),
                           b.internal()->ndofs(),
                           m_internal->solve(A.internal()->getMatrix(),b.internal()->getVector())) ;
    }
    else
    {
        x.internal()->init(b.internal()->parallelMng(),
                           b.internal()->ndofs(),
                           m_internal->solve(A.internal()->getLocMatrix(),b.internal()->getLocVector())) ;
    }
    m_status.iteration_count = m_internal->getNiter(A.isParallel()) ;
    //sol.display_centralized("X found");
  }


  m_status.succeeded = m_status.iteration_count < m_options->maxIterationNum();

  return m_status.succeeded;
}

#endif // ALIEN_USE_Composyx

bool
ComposyxInternalSolver::solve(IMatrix const& A, IVector const& b, IVector& x)
{
  using namespace Alien;

#ifdef ALIEN_USE_COMPOSYX
  SolverStatSentry<ComposyxInternalSolver> sentry(m_stater, BaseSolverStater::ePrepare);
  auto const& matrix = A.impl()->get<BackEnd::tag::composyx>();
  auto const& rhs    = b.impl()->get<BackEnd::tag::composyx>();
  auto&       sol    = x.impl()->get<BackEnd::tag::composyx>(true);
  sentry.release();

  SolverStatSentry<ComposyxInternalSolver> sentry2(m_stater, BaseSolverStater::eSolve);
  return solve(matrix, rhs, sol);
#else
  return false ;
#endif
}
/*---------------------------------------------------------------------------*/

void
ComposyxInternalSolver::internalPrintInfo() const
{
  m_stat.print(const_cast<ITraceMng*>(traceMng()), m_status,
      format("Linear Solver : {0}", "ComposyxSolver"));
  if (m_output_level > 0)
    alien_info([&] {
      cout() << "INIT SOLVER TIME : " << m_init_solver_time;
      cout() << "ITER SOLVER TIME : " << m_iter_solver_time;
    });
}



ILinearSolver*
ComposyxInternalSolverFactory(IMessagePassingMng* p_mng, IOptionsComposyxSolver* options)
{
  return new ComposyxInternalSolver(p_mng, options);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
}
