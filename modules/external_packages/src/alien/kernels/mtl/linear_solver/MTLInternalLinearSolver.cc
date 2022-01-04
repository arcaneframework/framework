#include <alien/kernels/mtl/linear_solver/MTLInternalLinearSolver.h>

#include <alien/AlienExternalPackagesPrecomp.h>

#include <alien/expression/solver/SolverStater.h>
#include <alien/core/backend/LinearSolverT.h>

#include <alien/core/backend/SolverFabricRegisterer.h>

#include <boost/numeric/mtl/mtl.hpp>
#include <boost/numeric/itl/itl.hpp>

#include <alien/kernels/mtl/data_structure/MTLVector.h>
#include <alien/kernels/mtl/data_structure/MTLMatrix.h>
#include <alien/kernels/mtl/MTLBackEnd.h>

#include <alien/kernels/mtl/algebra/MTLLinearAlgebra.h>
#include <alien/data/Space.h>


#include <alien/kernels/mtl/linear_solver/MTLOptionTypes.h>
#include <ALIEN/axl/MTLLinearSolver_IOptions.h>
#include <alien/kernels/mtl/linear_solver/arcane/MTLLinearSolverService.h>
#include <ALIEN/axl/MTLLinearSolver_IOptions.h>
#include <ALIEN/axl/MTLLinearSolver_StrongOptions.h>

/*---------------------------------------------------------------------------*/

#ifdef USE_PMTL4
mtl::par::environment* m_global_environment = NULL;
#endif /* USE_PMTL4 */

/*---------------------------------------------------------------------------*/
namespace Alien {

template class ALIEN_EXTERNAL_PACKAGES_EXPORT LinearSolver<BackEnd::tag::mtl>;

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

MTLInternalLinearSolver::~MTLInternalLinearSolver()
{
  ;
}

/*---------------------------------------------------------------------------*/

void
MTLInternalLinearSolver::init()
{
  m_stater.reset();
  m_stater.startInitializationMeasure();

#ifdef USE_PMTL4
  int argc = 0;
  char** argv = NULL;
  if (m_global_environment == NULL)
// m_global_environment = new mtl::par::environment(argc,argv) ;
#endif /* USE_PMTL4 */

    m_max_iteration = m_options->maxIterationNum();
  m_precision = m_options->stopCriteriaValue();
  m_solver_option = m_options->solver();
  m_preconditioner_option = m_options->preconditioner();

  m_initialized = true;
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
MTLInternalLinearSolver::getStatus() const
{
  return m_status;
}

/*---------------------------------------------------------------------------*/

bool
MTLInternalLinearSolver::solve(
    MTLMatrix const& matrix, MTLVector const& rhs, MTLVector& sol)
{
  using namespace Alien;

  const MatrixInternal::MTLMatrixType& A = matrix.internal()->m_internal;
  const VectorInternal::MTLVectorType& b = rhs.internal()->m_internal;
  VectorInternal::MTLVectorType& x = sol.internal()->m_internal;

  return _solve(A, b, x);
}

/*---------------------------------------------------------------------------*/

std::shared_ptr<ILinearAlgebra>
MTLInternalLinearSolver::algebra() const
{
  return std::shared_ptr<ILinearAlgebra>(new Alien::MTLLinearAlgebra());
}

/*---------------------------------------------------------------------------*/

bool
MTLInternalLinearSolver::_solve(MatrixInternal::MTLMatrixType const& matrix,
    VectorInternal::MTLVectorType const& rhs, VectorInternal::MTLVectorType& x)
{
#ifdef EXPORT
  mtl::io::matrix_market_ostream ifile("matrix.txt");
  ifile << matrix;
  ifile.close();
  ofstream rhsfile("rhs.txt");
  rhsfile << rhs;
  ofstream xfile("x.txt");
  xfile << x;
// exit(0) ;
#endif /* EXPORT */

  try {

    switch (m_solver_option) {
    case MTLOptionTypes::BiCGStab: {
      itl::basic_iteration<double> iter(rhs, m_max_iteration, m_precision);
      // itl::noisy_iteration<double> iter(rhs,m_max_iteration,m_precision);
      switch (m_preconditioner_option) {
      case MTLOptionTypes::NonePC: {
        itl::pc::identity<MatrixInternal::MTLMatrixType> P(matrix);
        itl::bicgstab(matrix, x, rhs, P, iter);
      } break;
      case MTLOptionTypes::DiagPC: {
        itl::pc::diagonal<MatrixInternal::MTLMatrixType> P(matrix);
        itl::bicgstab(matrix, x, rhs, P, iter);
      } break;
      case MTLOptionTypes::ILU0PC: {
        itl::pc::ilu_0<MatrixInternal::MTLMatrixType, float> P(matrix);
        itl::bicgstab(matrix, x, rhs, P, iter);
      } break;
      case MTLOptionTypes::ILUTPC:
      case MTLOptionTypes::SSORPC: {
        alien_fatal([&] { cout() << "Preconditioner not available"; });
      } break;
      }
      m_status.iteration_count = iter.iterations();
      m_status.residual = iter.resid();
      if (m_status.iteration_count >= m_max_iteration)
        m_status.succeeded = false;
      else
        m_status.succeeded = true;
    } break;
    case MTLOptionTypes::CG: {
      std::cout<<"MTL4 CG"<<std::endl ;
      //itl::basic_iteration<double> iter(rhs, m_max_iteration, m_precision);
      itl::noisy_iteration<double> iter(rhs,m_max_iteration,m_precision);
      switch (m_preconditioner_option) {
      case MTLOptionTypes::NonePC: {
        itl::pc::identity<MatrixInternal::MTLMatrixType> P(matrix);
        itl::cg(matrix, x, rhs, P, iter);
      } break;
      case MTLOptionTypes::DiagPC: {
        itl::pc::diagonal<MatrixInternal::MTLMatrixType> P(matrix);
        itl::cg(matrix, x, rhs, P, iter);
      } break;
      case MTLOptionTypes::SSORPC: {
        alien_fatal([&] { cout() << "Preconditioner not available"; });
      } break;
      }
      m_status.iteration_count = iter.iterations();
      m_status.residual = iter.resid();
      std::cout<<"NB ITER : "<<m_status.iteration_count<<" residual="<<m_status.residual<<std::endl;
      if (m_status.iteration_count >= m_max_iteration)
        m_status.succeeded = false;
      else
        m_status.succeeded = true;
    } break;
#ifdef MTL_HAS_UMFPACK
    case MTLOptionTypes::LU: {
      mtl::matrix::umfpack::solver<MatrixInternal::MTLMatrixType> solver(matrix);
      solver(x, rhs);
      m_status.succeeded = true;
    } break;
#endif /* MTL_HAS_UMFPACK */
    default: {
      alien_fatal([&] { cout() << "Solver option not available"; });
    } break;
    }

  } catch (mtl::logic_error& e) {
    throw Arccore::FatalErrorException(A_FUNCINFO,
        Arccore::String::format(
            "MTL Login Error Exception catched while solving linear system : {0}",
            e.what()));
  } catch (mtl::runtime_error& e) {
    throw Arccore::FatalErrorException(A_FUNCINFO,
        Arccore::String::format(
            "MTL Runtime Error Exception catched while solving linear system : {0}",
            e.what()));
  } catch (mtl::index_out_of_range& e) {
    throw Arccore::FatalErrorException(A_FUNCINFO,
        Arccore::String::format(
            "MTL Out of range Exception catched while solving linear system : {0}",
            e.what()));
  }

#ifdef EXPORT
  mtl::io::matrix_market_ostream ifile("matrix.txt");
  ifile << matrix;
  ifile.close();
  ofstream rhsfile("rhs.txt");
  rhsfile << rhs;
  rhsfile.close();
  ofstream solfile("sol.txt");
  solfile << x;
  solfile.close();
// exit(0) ;
#endif

  return m_status.succeeded;
}

/*---------------------------------------------------------------------------*/

void
MTLInternalLinearSolver::end()
{
  if (m_output_level > 0)
    internalPrintInfo();
}

/*---------------------------------------------------------------------------*/

bool
MTLInternalLinearSolver::hasParallelSupport() const
{
#ifdef USE_PMTL4
  return true;
#else /* USE_PMTL4 */
  return false;
#endif /* USE_PMTL4 */
}

/*---------------------------------------------------------------------------*/

void
MTLInternalLinearSolver::internalPrintInfo() const
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

template<>
class SolverFabric<Alien::BackEnd::tag::mtl>
: public ISolverFabric
{
public :
   BackEndId backend() const {
     return "mtl4" ;
  }

  void
  add_options(CmdLineOptionDescType& cmdline_options) const
  {
    using namespace boost::program_options;
    options_description desc("MTL4 options");
    desc.add_options()("mtl4-solver", value<std::string>()->default_value("bicgs"),"solver algo name : bicgstab")
                      ("mtl4-precond", value<std::string>()->default_value("none"),"preconditioner ilu diag none");

    cmdline_options.add(desc) ;
  }

  template<typename OptionT>
  Alien::ILinearSolver* _create(OptionT const& options,Alien::IMessagePassingMng* pm) const
  {
    double tol = get<double>(options,"tol");
    int max_iter = get<int>(options,"max-iter");

    std::string solver_type_s = get<std::string>(options,"mtl4-solver");
    MTLOptionTypes::eSolver solver_type =
        OptionsMTLLinearSolverUtils::stringToSolverEnum(solver_type_s);
    std::string precond_type_s = get<std::string>(options,"mtl4-precond");
    MTLOptionTypes::ePreconditioner precond_type =
        OptionsMTLLinearSolverUtils::stringToPreconditionerEnum(precond_type_s);
    // options
    auto solver_options = std::make_shared<StrongOptionsMTLLinearSolver>(
        MTLLinearSolverOptionsNames::_outputLevel = get<int>(options,"output-level"),
        MTLLinearSolverOptionsNames::_maxIterationNum = max_iter,
        MTLLinearSolverOptionsNames::_stopCriteriaValue = tol,
        MTLLinearSolverOptionsNames::_preconditioner = precond_type,
        MTLLinearSolverOptionsNames::_solver = solver_type);
    // service
    return new Alien::MTLLinearSolverService(pm, solver_options);

  }

  Alien::ILinearSolver* create(CmdLineOptionType const& options,Alien::IMessagePassingMng* pm) const
  {
    return _create(options,pm) ;
  }

  Alien::ILinearSolver* create(JsonOptionType const& options,Alien::IMessagePassingMng* pm) const
  {
    return _create(options,pm) ;
  }
};

typedef SolverFabric<Alien::BackEnd::tag::mtl> MTL4SolverFabric ;
REGISTER_SOLVER_FABRIC(MTL4SolverFabric);

} // namespace Alien
