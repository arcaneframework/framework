
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include <alien/kernels/mtl/linear_solver/arcane/MTLLinearSolverService.h>
#include <ALIEN/axl/MTLLinearSolver_StrongOptions.h>

namespace Alien {

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifdef USE_PMTL4
// mtl::par::environment * m_global_environment = NULL;
#endif /* USE_PMTL4 */

/*---------------------------------------------------------------------------*/
#ifdef ALIEN_USE_ARCANE
MTLLinearSolverService::MTLLinearSolverService(const Arcane::ServiceBuildInfo& sbi)
: ArcaneMTLLinearSolverObject(sbi)
, LinearSolver<BackEnd::tag::mtl>(
      sbi.subDomain()->parallelMng()->messagePassingMng(), options())
{
  ;
}
#endif

MTLLinearSolverService::MTLLinearSolverService(
    Arccore::MessagePassing::IMessagePassingMng* parallel_mng,
    std::shared_ptr<IOptionsMTLLinearSolver> _options)
: ArcaneMTLLinearSolverObject(_options)
, LinearSolver<BackEnd::tag::mtl>(parallel_mng, options())
{
  ;
}

/*---------------------------------------------------------------------------*/

MTLLinearSolverService::~MTLLinearSolverService()
{
  ;
}

/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE_MTLLINEARSOLVER(MTLSolver, MTLLinearSolverService);

} // namespace Alien

REGISTER_STRONG_OPTIONS_MTLLINEARSOLVER();
