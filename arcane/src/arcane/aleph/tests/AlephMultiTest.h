// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AlephMultiTest.h                                                 (C) 2013 */
/*                                                                           */
/*---------------------------------------------------------------------------*/
#ifndef ALEPH_MULTI_TEST_H
#define ALEPH_MULTI_TEST_H

ARCANETEST_BEGIN_NAMESPACE
using namespace Arcane;
class AlephSolver;

// ****************************************************************************
// * Aleph multi-solver test
// ****************************************************************************
class AlephMultiTest
: public ArcaneAlephMultiTestObject
{
 public:
  struct SolverBuildInfo
  {
   public:
    SolverBuildInfo()
    : m_number_of_resolution_per_solvers(0)
    , m_underliying_solver(0)
    , m_number_of_core(0)
    {}

    Integer m_number_of_resolution_per_solvers;
    Integer m_underliying_solver;
    Integer m_number_of_core;
  };

  AlephMultiTest(const ModuleBuildInfo&);
  ~AlephMultiTest(void);
  void init(void);
  void compute(void);

 private:
  AlephFactory* m_aleph_factory;
  UniqueArray<AlephSolver*> m_global_aleph_solver;
  UniqueArray<SolverBuildInfo> m_solvers_build_info;
  UniqueArray<AlephSolver*> m_posted_solvers;
};

ARCANETEST_END_NAMESPACE
#endif
