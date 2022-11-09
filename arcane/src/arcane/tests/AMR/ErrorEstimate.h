// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ErrorEstimate.hc                                            (C) 2000-2022 */
/*                                                                           */
/* Service de solutions analytiques utilisees pour estimer l'erreur AMR.     */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_TEST_AMR_ERRORESTIMATE_H
#define ARCANE_TEST_AMR_ERRORESTIMATE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Real3.h"

#include "arcane/ArcaneTypes.h"

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
   * Attacher une fonction arbitraire calculant
   * la valeur exacte de la solution dans un point donné.
   */
  void attachExactValue(Real fptr(const Real3& p));

  /**
   * Attacher une fonction arbitraire calculant
   * la valeur exacte du gradient de la solution dans un point donné.
   */
  void attachExactGradient(Real3 fptr(const Real3& p));

  /**
   * Attacher une fonction arbitraire calculant
   * la valeur exacte du hessien de la solution dans un point donné.
   */
  void attachExactHessian(Real3x3 fptr(const Real3& p));

  /**
   * Calcul et stockage de l'erreur de la solution e = u-u_h,
   * du gradient grad(e) = grad(u) - grad(u_h), et aussi le hessien
   * grad(grad(e)) = grad(grad(u)) - grad(grad(u_h)).
   */
  void computeSol(RealArray& sol, IMesh* mesh);

  void computeGlobalError();
  void computeError(RealArray& error, IMesh* mesh);
  void errorToFlagConverter(RealArray& error_per_cell, const Real& refine_frac,
                            const Real& coarsen_frac, const Integer& max_level, IMesh* mesh);
  /**
   * erreur L2.
   * Note: pas de calcul de l'erreur,
   * il faut appeler d'abord le compute_error()
   */
  Real l2Error();

  /**
   * erreur LInf.
   * Note: pas de calcul de l'erreur,
   * il faut appeler d'abord le compute_error()
   */
  Real lInfError();

  /**
   * Cette methode retourne l'erreur dans la norme demandée.
   *  Note: pas de calcul de l'erreur,
   * il faut appeler d'abord le compute_error()
   */
  Real errorNorm(const NormType& norm);

 private:

  /**
   * Function pointer à une fonction fournit par l'utilisateur
   * Celle-ci calcule la valeur exacte de la solution.
   */
  Real (*m_exact_value)(const Real3& p) = nullptr;

  /**
   * Function pointer à une fonction fournit par l'utilisateur
   * Celle-ci calcule la dérivée exacte de la solution.
   */
  Real3 (*m_exact_gradient)(const Real3& p) = nullptr;

  /**
   * Function pointer à une fonction fournit par l'utilisateur
   * Celle-ci calcule la dérivéee seconde exacte de la solution.
   */
  Real3x3 (*m_exact_hessian)(const Real3& p) = nullptr;

  /**
   * Calcul l'erreur sur la solution et ses dérivées pour un système scalaire.
   * elle peut être utilisée pour résoudre des systèmes vectoriels
   */
  void _computeError(RealArray& error_vals, IMesh* mesh);

  /**
   * Vecteur propre au stockage de l'erreur globale.
   */
  Real3 m_error_vals;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
