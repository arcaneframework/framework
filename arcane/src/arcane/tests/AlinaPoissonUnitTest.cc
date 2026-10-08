// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AlinaPoissonUnitTest.cc                                     (C) 2000-2026 */
/*                                                                           */
/* Alina test using Poisson problem.                                         */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/ITraceMng.h"

#include "arcane/core/BasicUnitTest.h"
#include "arcane/core/FactoryService.h"
#include "arcane/core/IMesh.h"
#include "arcane/core/IParallelMng.h"

#include "arccore/message_passing/IMessagePassingMng.h"

#include "arccore/alina/AlinaLib.h"
#include "arccore/alina/PoissonProblemGenerator.h"

#include "arcane/tests/ArcaneTestGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
extern "C++" ARCANE_STD_EXPORT void
_testRedisAdapter(ITraceMng* tm);
}

namespace ArcaneTest
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arcane;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief ItemVector test service
 */
class AlinaPoissonUnitTest
: public BasicUnitTest
{

 public:

  explicit AlinaPoissonUnitTest(const ServiceBuildInfo& cb);

 public:

  void initializeTest() override {}
  void executeTest() override;

 private:
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE(AlinaPoissonUnitTest,
                        ServiceProperty("AlinaPoissonUnitTest", ST_CaseOption),
                        ARCANE_SERVICE_INTERFACE(IUnitTest));

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AlinaPoissonUnitTest::
AlinaPoissonUnitTest(const ServiceBuildInfo& mb)
: BasicUnitTest(mb)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AlinaPoissonUnitTest::
executeTest()
{
  using namespace Arcane::Alina;
  using namespace Arcane::AlinaLib;

  IParallelMng* pm = mesh()->parallelMng();
  ITraceMng* tm = pm->traceMng();
  tm->info() << "DO_TEST_ALINA";
  IMessagePassingMng* mpm = pm->messagePassingMng();

  int comm_rank = mpm->commRank();
  int comm_size = mpm->commSize();

  const Int32 n = 32;

  // For 32 bit indexing
  using ColumnType = Int32;
  // For 64 bit indexing
  // using ColumnType = Int64;

  std::vector<ColumnType> ptr;
  std::vector<ColumnType> col;
  std::vector<double> val;
  std::vector<double> rhs;

  Int32 chunk = PoissonProblemGenerator::createDistributedMatrix(comm_rank, comm_size, n, 1, ptr, col, val, rhs);

  // Setup
  AlinaSolverParameters prm(mpm, pm->traceMng());
  prm.setSolverPreconditioner(AlinaLib::eAlinaPreconditionerType::AMG);
  prm.setSolverType(AlinaLib::eAlinaSolverType::ConjugateGradient);

  UniqueArray<double> x(rhs.size(), 0.0);

  // Solve
  Real t0 = platform::getRealTime();
  {
    AlinaCSRMatrixView matrix_view(chunk, ptr.data(), col.data(), val.data());
    AlinaDistributedSolver solver(prm, matrix_view);
    SmallSpan<const double> rhs_view(rhs.data(), rhs.size());
    AlinaConvergenceInfo cnv = solver.solve(rhs_view, x.smallSpan());

    info() << "Iterations: " << cnv.iterations
           << " Error:      " << cnv.residual;
  }
  Real t1 = platform::getRealTime();
  tm->info() << "AlinaTime=" << (t1 - t0);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace ArcaneTest

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
