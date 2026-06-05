// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ErrorEstimate.h                                             (C) 2000-2022 */
/*                                                                           */
/* Service of analytical solutions used to estimate AMR error.               */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_TEST_AMR_ERRORESTIMATE_H
#define ARCANE_TEST_AMR_ERRORESTIMATE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Real3.h"

#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ErrorEstimate
{
  enum NormType
  {
    L2 = 0,
    L_INF = 1,
    INVALID_NORM = 10
  };

 public:

  /**
   * Attach an arbitrary function calculating
   * the exact value of the solution at a given point.
   */
  void attachExactValue(Real fptr(const Real3& p));

  /**
   * Attach an arbitrary function calculating
   * the exact value of the solution gradient at a given point.
   */
  void attachExactGradient(Real3 fptr(const Real3& p));

  /**
   * Attach an arbitrary function calculating
   * the exact value of the solution Hessian at a given point.
   */
  void attachExactHessian(Real3x3 fptr(const Real3& p));

  /**
   * Calculation and storage of the solution error e = u-u_h,
   * of the gradient grad(e) = grad(u) - grad(u_h), and also the Hessian
   * grad(grad(e)) = grad(grad(u)) - grad(grad(u_h)).
   */
  void computeSol(RealArray& sol, IMesh* mesh);

  void computeGlobalError();
  void computeError(RealArray& error, IMesh* mesh);
  void errorToFlagConverter(RealArray& error_per_cell, const Real& refine_frac,
                            const Real& coarsen_frac, const Integer& max_level, IMesh* mesh);
  /**
   * L2 error.
   * Note: error is not calculated,
   * you must call compute_error() first.
   */
  Real l2Error();

  /**
   * LInf error.
   * Note: error is not calculated,
   * you must call compute_error() first.
   */
  Real lInfError();

  /**
   * This method returns the error in the requested norm.
   *  Note: error is not calculated,
   * you must call compute_error() first.
   */
  Real errorNorm(const NormType& norm);

 private:

  /**
   * Function pointer to a function provided by the user
   * This calculates the exact value of the solution.
   */
  Real (*m_exact_value)(const Real3& p) = nullptr;

  /**
   * Function pointer to a function provided by the user
   * This calculates the exact derivative of the solution.
   */
  Real3 (*m_exact_gradient)(const Real3& p) = nullptr;

  /**
   * Function pointer to a function provided by the user
   * This calculates the exact second derivative of the solution.
   */
  Real3x3 (*m_exact_hessian)(const Real3& p) = nullptr;

  /**
   * Calculates the error on the solution and its derivatives for a scalar system.
   * It can be used to solve vector systems.
   */
  void _computeError(RealArray& error_vals, IMesh* mesh);

  /**
   * Vector dedicated to storing the global error.
   */
  Real3 m_error_vals;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
