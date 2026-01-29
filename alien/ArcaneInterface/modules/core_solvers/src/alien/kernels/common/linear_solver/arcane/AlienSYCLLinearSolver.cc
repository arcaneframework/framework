// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

/**
 * Interface du service de résolution de système linéaire
 */
#include "alien/AlienLegacyConfig.h"

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

#include <alien/kernels/common/linear_solver/arcane/AlienSYCLLinearSolver.h>
#include <ALIEN/axl/AlienCoreSolver_StrongOptions.h>

namespace Alien {

/** Constructeur de la classe */

#ifdef ALIEN_USE_ARCANE
AlienSYCLLinearSolver::AlienSYCLLinearSolver(const Arcane::ServiceBuildInfo& sbi)
: ArcaneAlienCoreSolverObject(sbi)
, Alien::AlienCoreSYCLLinearSolver(
      sbi.subDomain()->parallelMng()->messagePassingMng(), options())
{
  Alien::setTraceMng(Arcane::TraceAccessor::traceMng());
}
#endif

AlienSYCLLinearSolver::AlienSYCLLinearSolver(
    Arccore::MessagePassing::IMessagePassingMng* parallel_mng,
    std::shared_ptr<IOptionsAlienCoreSolver> _options)
: ArcaneAlienCoreSolverObject(_options)
, Alien::AlienCoreSYCLLinearSolver(parallel_mng, options())
{
}

/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE_ALIENCORESOLVER(AlienCoreSYCLSolver, AlienSYCLLinearSolver);

} // namespace Alien

//REGISTER_STRONG_OPTIONS_ALIENCORESOLVER();
