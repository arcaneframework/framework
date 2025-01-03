// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AlephTestModule.h                                           (C) 2000-2024 */
/*                                                                           */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_TESTS_ALEPHTESTMODULE_H
#define ARCANE_TESTS_ALEPHTESTMODULE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/aleph/tests/AlephTest.h"
#include "arcane/aleph/tests/AlephTestModule_axl.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace ArcaneTest
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arcane;

/*!
 * \brief Module de test pour Aleph.
 */
class AlephTestModule
: public ArcaneAlephTestModuleObject
{
 public:

  explicit AlephTestModule(const ModuleBuildInfo&);
  ~AlephTestModule();
  void init();
  void compute();
  void postSolver(const Integer);

 private:

  void initAmrRefineMesh(Integer nb_to_refine);
  void initAlgebra();

 public:

  static Real geoFaceSurface(Face, VariableItemReal3&);

 public:

  Integer m_total_nb_cell;
  Integer m_local_nb_cell;
  UniqueArray<Integer> m_rows_nb_element;
  UniqueArray<AlephInt> m_vector_indexs;
  UniqueArray<Real> m_vector_zeroes;
  UniqueArray<Real> m_vector_rhs;
  UniqueArray<Real> m_vector_lhs;
  AlephKernel* m_aleph_kernel;
  IAlephFactory* m_aleph_factory;
  AlephParams* m_aleph_params;
  UniqueArray<AlephMatrix*> m_aleph_mat;
  UniqueArray<AlephVector*> m_aleph_rhs;
  UniqueArray<AlephVector*> m_aleph_sol;
  Integer m_get_solution_idx;
  Integer m_fake_nb_iteration;
  Real m_fake_residual_norm[4];
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace ArcaneTest

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
