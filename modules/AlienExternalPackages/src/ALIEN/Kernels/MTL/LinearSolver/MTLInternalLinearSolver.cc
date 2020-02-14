#include <ALIEN/Kernels/MTL/LinearSolver/MTLInternalLinearSolver.h>

#include <ALIEN/Alien-ExternalPackagesPrecomp.h>

#include <ALIEN/Expression/Solver/SolverStats/SolverStater.h>
#include <ALIEN/Core/Backend/LinearSolverT.h>

#include <boost/numeric/mtl/mtl.hpp>
#include <boost/numeric/itl/itl.hpp>

#include <ALIEN/Kernels/MTL/DataStructure/MTLVector.h>
#include <ALIEN/Kernels/MTL/DataStructure/MTLMatrix.h>
#include <ALIEN/Kernels/MTL/MTLBackEnd.h>

#include <ALIEN/Kernels/MTL/Algebra/MTLLinearAlgebra.h>
#include <ALIEN/Data/Space.h>

#include <ALIEN/axl/MTLLinearSolver_IOptions.h>

/*---------------------------------------------------------------------------*/

#ifdef USE_PMTL4
mtl::par::environment * m_global_environment = NULL;
#endif /* USE_PMTL4 */


/*---------------------------------------------------------------------------*/
namespace Alien {

template class ALIEN_EXTERNALPACKAGES_EXPORT LinearSolver<BackEnd::tag::mtl>;

MTLInternalLinearSolver::MTLInternalLinearSolver(
    Arccore::MessagePassing::IMessagePassingMng* parallel_mng,
    IOptionsMTLLinearSolver* options)
: m_initialized(false)
, m_solver_option(MTLOptionTypes::BiCGStab)
, m_preconditioner_option(MTLOptionTypes::ILU0PC)
, m_max_iteration(0)
, m_precision(0)
, m_output_level(0)
  , m_parallel_mng(parallel_mng)
  , m_options(options)
{
  ;
}

/*---------------------------------------------------------------------------*/

MTLInternalLinearSolver::
~MTLInternalLinearSolver()
{
  ;
}

/*---------------------------------------------------------------------------*/

void 
MTLInternalLinearSolver::
init()
{
  m_stater.reset();
  m_stater.startInitializationMeasure();

#ifdef USE_PMTL4
  int argc = 0 ;
  char** argv = NULL ;
  if(m_global_environment==NULL)
    // m_global_environment = new mtl::par::environment(argc,argv) ;
#endif /* USE_PMTL4 */

  m_max_iteration = m_options->maxIterationNum() ;
  m_precision = m_options->stopCriteriaValue() ;
  m_solver_option = m_options->solver() ;
  m_preconditioner_option = m_options->preconditioner() ;

  m_initialized = true ;
}

/*---------------------------------------------------------------------------*/

void
MTLInternalLinearSolver::updateParallelMng(
    Arccore::MessagePassing::IMessagePassingMng* pm)
{
  m_parallel_mng = pm;
}

/*---------------------------------------------------------------------------*/

const Alien::SolverStatus&
MTLInternalLinearSolver::
getStatus() const
{
  return m_status;
}

/*---------------------------------------------------------------------------*/

bool
MTLInternalLinearSolver::
solve(MTLMatrix const& matrix, MTLVector const& rhs, MTLVector& sol)
{
  using namespace Alien;

  const MatrixInternal::MTLMatrixType & A = matrix.internal()->m_internal;
  const VectorInternal::MTLVectorType & b = rhs.internal()->m_internal;
  VectorInternal::MTLVectorType & x = sol.internal()->m_internal;

  return _solve(A,b,x) ;
}

/*---------------------------------------------------------------------------*/

std::shared_ptr<ILinearAlgebra>
MTLInternalLinearSolver::
algebra() const
{
  return std::shared_ptr<ILinearAlgebra>(new Alien::MTLLinearAlgebra());
}

/*---------------------------------------------------------------------------*/

bool
MTLInternalLinearSolver::
_solve(MatrixInternal::MTLMatrixType const& matrix, 
       VectorInternal::MTLVectorType const& rhs, 
       VectorInternal::MTLVectorType & x)
{
#ifdef EXPORT
  mtl::io::matrix_market_ostream ifile("matrix.txt") ;
  ifile<<matrix;
  ifile.close();
  ofstream rhsfile("rhs.txt") ;
  rhsfile<<rhs ;
  ofstream xfile("x.txt") ;
  xfile<<x ;
  //exit(0) ;
#endif /* EXPORT */

  try { 

    switch(m_solver_option)
    {
    case MTLOptionTypes::BiCGStab :
      {
	itl::basic_iteration<double> iter(rhs,m_max_iteration,m_precision);
	//itl::noisy_iteration<double> iter(rhs,m_max_iteration,m_precision);
	switch(m_preconditioner_option)
	{
	case MTLOptionTypes::NonePC:
	  {
	    itl::pc::identity<MatrixInternal::MTLMatrixType> P(matrix);
	    itl::bicgstab(matrix,x,rhs,P,iter);
	  }
	  break ;
	case MTLOptionTypes::DiagPC:
	  {
	    itl::pc::diagonal<MatrixInternal::MTLMatrixType> P(matrix);
	    itl::bicgstab(matrix,x,rhs,P,iter);
	  }
	  break ;
	case MTLOptionTypes::ILU0PC:
	  {
	    itl::pc::ilu_0<MatrixInternal::MTLMatrixType, float> P(matrix);
	    itl::bicgstab(matrix,x,rhs,P,iter);
	  }
	  break ;
	case MTLOptionTypes::ILUTPC:
	case MTLOptionTypes::SSORPC:
	  {
	    alien_fatal([&] {
	      cout() << "Preconditioner not available";
	    });
	  }
	  break ;
	}
	m_status.iteration_count = iter.iterations() ;
	m_status.residual = iter.resid() ;
	if(m_status.iteration_count>=m_max_iteration)
	  m_status.succeeded = false ;
	else
	  m_status.succeeded = true ;
      }
      break ;
#ifdef MTL_HAS_UMFPACK
    case MTLOptionTypes::LU :
      {
	mtl::matrix::umfpack::solver<MatrixInternal::MTLMatrixType> solver(matrix);
	solver(x,rhs) ;
	m_status.succeeded = true ;
      }
      break ;
#endif /* MTL_HAS_UMFPACK */
    default :
      {
        alien_fatal([&] {
          cout() << "Solver option not available";
        });
      }
      break ;
    }

  } catch (mtl::logic_error & e) {
    throw Arccore::FatalErrorException(A_FUNCINFO,
        Arccore::String::format(
            "MTL Login Error Exception catched while solving linear system : {0}",
            e.what()));
  } catch (mtl::runtime_error & e) {
    throw Arccore::FatalErrorException(A_FUNCINFO,
        Arccore::String::format(
            "MTL Runtime Error Exception catched while solving linear system : {0}",
            e.what()));
  } catch (mtl::index_out_of_range & e) {
    throw Arccore::FatalErrorException(A_FUNCINFO,
        Arccore::String::format(
            "MTL Out of range Exception catched while solving linear system : {0}",
            e.what()));
  }

#ifdef EXPORT
  mtl::io::matrix_market_ostream ifile("matrix.txt") ;
  ifile<<matrix;
  ifile.close();
  ofstream rhsfile("rhs.txt") ;
  rhsfile<<rhs ;
  rhsfile.close() ;
  ofstream solfile("sol.txt") ;
  solfile<<x ;
  solfile.close() ;
  //exit(0) ;
#endif

  return m_status.succeeded  ;
}

/*---------------------------------------------------------------------------*/

void 
MTLInternalLinearSolver::
end()
{
  if(m_output_level>0)
    internalPrintInfo() ;
}

/*---------------------------------------------------------------------------*/

bool 
MTLInternalLinearSolver::
hasParallelSupport() const
{
#ifdef USE_PMTL4
  return true;
#else /* USE_PMTL4 */
  return false ;
#endif /* USE_PMTL4 */
}

/*---------------------------------------------------------------------------*/

void
MTLInternalLinearSolver::
internalPrintInfo() const
{
  m_stater.print(const_cast<Arccore::ITraceMng*>(traceMng()), m_status,
      Arccore::String::format("Linear Solver : {0}", "MTLLinearSolver"));
}

IInternalLinearSolver<MTLMatrix, MTLVector>*
MTLInternalLinearSolverFactory(
    Arccore::MessagePassing::IMessagePassingMng* p_mng, IOptionsMTLLinearSolver* options)
{
  return new MTLInternalLinearSolver(p_mng, options);
}

} // namespace Alien
