// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CaseFunctionTesterModule.cc                                 (C) 2000-2023 */
/*                                                                           */
/* Module de test des 'CaseFunction'.                                        */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/StandardCaseFunction.h"
#include "arcane/core/ITimeLoopMng.h"
#include "arcane/core/TimeLoopEntryPointInfo.h"
#include "arcane/core/TimeLoop.h"

#include "arcane/tests/CaseFunctionTester_axl.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace ArcaneTest
{
using namespace Arcane;
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Module de test des 'ICaseFunction'.
 */
class CaseFunctionTesterModule
: public ArcaneCaseFunctionTesterObject
{
 public:

  explicit CaseFunctionTesterModule(const ModuleBuildInfo& mbi);

 public:

  static void staticInitialize(ISubDomain* sd);

 public:

  VersionInfo versionInfo() const override { return Arcane::VersionInfo(0, 1, 0); }

 public:

  void init();
  void loop();

 private:
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_DEFINE_STANDARD_MODULE(CaseFunctionTesterModule, CaseFunctionTester);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{
  class StandardFuncTest
  : public StandardCaseFunction
  , public IBinaryMathFunctor<Real, Real3, Real>
  {
   public:

    StandardFuncTest(const CaseFunctionBuildInfo& bi)
    : StandardCaseFunction(bi)
    {}

   public:

    virtual IBinaryMathFunctor<Real, Real3, Real>* getFunctorRealReal3ToReal()
    {
      return this;
    }
    virtual Real apply(Real r, Real3 r3)
    {
      return r + r3.normL2();
    }
  };

} // namespace

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CaseFunctionTesterModule::
CaseFunctionTesterModule(const ModuleBuildInfo& mbi)
: ArcaneCaseFunctionTesterObject(mbi)
{
  addEntryPoint(this, "Init",
                &CaseFunctionTesterModule::init, IEntryPoint::WInit);
  addEntryPoint(this, "Loop",
                &CaseFunctionTesterModule::loop);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseFunctionTesterModule::
staticInitialize(ISubDomain* sd)
{
  ITimeLoopMng* tlm = sd->timeLoopMng();
  ITimeLoop* time_loop = tlm->createTimeLoop("CaseFunctionTester");

  {
    List<TimeLoopEntryPointInfo> clist;
    clist.add(TimeLoopEntryPointInfo("CaseFunctionTester.Init"));
    time_loop->setEntryPoints(ITimeLoop::WInit, clist);
  }

  {
    List<TimeLoopEntryPointInfo> clist;
    clist.add(TimeLoopEntryPointInfo("CaseFunctionTester.Loop"));
    time_loop->setEntryPoints(ITimeLoop::WComputeLoop, clist);
  }

  {
    StringList clist;
    clist.add("CaseFunctionTester");
    time_loop->setRequiredModulesName(clist);
  }

  tlm->registerTimeLoop(time_loop);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseFunctionTesterModule::
loop()
{
  if (m_global_iteration() > 10)
    subDomain()->timeLoopMng()->stopComputeLoop(true);

  // Temps de début d'itération auquel sont calculées les valeurs des fonctions
  Real global_time = m_global_old_time();
  Int32 global_iter = m_global_iteration();

  {
    Real v1 = options()->realTimeMultiply2.value();
    Real expected_v1 = global_time * 2.0;
    info() << "Function: real-time-multiply-2: " << v1;
    if (!math::isNearlyEqual(v1, expected_v1))
      ARCANE_FATAL("Bad (1) value v={0} expected={1}", v1, expected_v1);
  }
  {
    int v1 = options()->intIterMultiply3.value();
    int expected_v1 = global_iter * 3;
    info() << "Function: int-iter-multiply-3: " << v1;
    if (v1 != expected_v1)
      ARCANE_FATAL("Bad (2) value v={0} expected={1}", v1, expected_v1);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseFunctionTesterModule::
init()
{
  m_global_deltat.assign(1.5);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace ArcaneTest

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
