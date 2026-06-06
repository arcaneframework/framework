// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IStandardFunction.h                                         (C) 2000-2025 */
/*                                                                           */
/* Interface of a standard function.                                         */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ISTANDARDFUNCTION_H
#define ARCANE_CORE_ISTANDARDFUNCTION_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/IMathFunctor.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Interface managing a standard function.
 *
 * These functions can be associated with a dataset option.
 *
 * Standard functions have the following possible prototypes:
 * - f(Real,Real) -> Real
 * - f(Real,Real3) -> Real
 * - f(Real,Real) -> Real3
 * - f(Real,Real3) -> Real3
 *
 */
class ARCANE_CORE_EXPORT IStandardFunction
{
 public:

  virtual ~IStandardFunction() = default;

 public:

  virtual IBinaryMathFunctor<Real, Real, Real>* getFunctorRealRealToReal() = 0;
  virtual IBinaryMathFunctor<Real, Real3, Real>* getFunctorRealReal3ToReal() = 0;
  virtual IBinaryMathFunctor<Real, Real, Real3>* getFunctorRealRealToReal3() = 0;
  virtual IBinaryMathFunctor<Real, Real3, Real3>* getFunctorRealReal3ToReal3() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
