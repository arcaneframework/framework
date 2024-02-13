// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#ifndef ALIENBENCHMODULE_H
#define ALIENBENCHMODULE_H

#include "AlienBench_axl.h"

#include <arcane/random/Uniform01.h>
#include <arcane/random/LinearCongruential.h>

class MemoryAllocationTracker;

using namespace Arcane;

class AlienBenchModule : public ArcaneAlienBenchObject
{
 public:
  //! Constructor
  AlienBenchModule(const Arcane::ModuleBuildInfo& mbi)
  : ArcaneAlienBenchObject(mbi)
  , m_uniform(m_generator)
  {
  }

  //! Destructor
  virtual ~AlienBenchModule(){};

 public:
  //! Initialization
  void init();
  //! Run the test
  void test();

 private:
 private:
  Real funcn(Real3 x) const;
  Real funck(Real3 x) const;
  Real dii(const Cell& ci) const;
  Real fij(const Cell& ci, const Cell& cj) const;

  eItemKind m_stencil_kind = Arcane::IK_Face;
  bool m_homogeneous = false;
  Real m_diag_coeff = 0.;
  Real m_off_diag_coeff = 0.5;
  Real m_lambdax = 1.;
  Real m_lambday = 1.;
  Real m_lambdaz = 1.;
  Real m_alpha = 1.;
  Real m_sigma = 0.;
  IParallelMng* m_parallel_mng = nullptr;

  Arcane::CellGroup m_areaU;
  Arcane::random::MinstdRand m_generator;
  mutable Arcane::random::Uniform01<Arcane::random::MinstdRand> m_uniform;

  Alien::MatrixDistribution m_mdist;
  Alien::VectorDistribution m_vdist;
};

#endif
