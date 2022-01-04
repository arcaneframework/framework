// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IStandardFunction.h                                         (C) 2000-2011 */
/*                                                                           */
/* Interface d'une fonction standard.                                        */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ISTANDARDFUNCTION_H
#define ARCANE_ISTANDARDFUNCTION_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/IMathFunctor.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface gérant une fonction standard.
 *
 * Ces fonctions peuvent être associées à une option du jeu de données.
 *
 * Les fonctions standards ont les prototypes suivants possibles:
 * - f(Real,Real) -> Real
 * - f(Real,Real3) -> Real
 * - f(Real,Real) -> Real3
 * - f(Real,Real3) -> Real3
 *
 */
class ARCANE_CORE_EXPORT IStandardFunction
{
 public:

  virtual ~IStandardFunction(){}

 public:
  
  virtual IBinaryMathFunctor<Real,Real,Real>* getFunctorRealRealToReal() =0;
  virtual IBinaryMathFunctor<Real,Real3,Real>* getFunctorRealReal3ToReal() =0;
  virtual IBinaryMathFunctor<Real,Real,Real3>* getFunctorRealRealToReal3() =0;
  virtual IBinaryMathFunctor<Real,Real3,Real3>* getFunctorRealReal3ToReal3() =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

