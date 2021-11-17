#include <alien/expression/solver/ILinearSolver.h>

#include "petsc.h"
#include "PETScConfig.h"
#include "PETScInitType.h"

namespace Alien {
void
PETScInitType::apply(const PETScConfig* iksp, KSP& ksp,
    const PETScInternalLinearSolver::InitType::eInit type)
{
  switch (type) {
  case PETScInternalLinearSolver::InitType::User:
    iksp->checkError("Init user guess", KSPSetInitialGuessNonzero(ksp, PETSC_TRUE));
    break;
  case PETScInternalLinearSolver::InitType::Zero:
    iksp->checkError("Init zero guess", KSPSetInitialGuessNonzero(ksp, PETSC_FALSE));
    break;
  case PETScInternalLinearSolver::InitType::Knoll:
    iksp->checkError("Init Knoll guess", KSPSetInitialGuessKnoll(ksp, PETSC_TRUE));
    break;
  }
}

} // namespace Alien
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
