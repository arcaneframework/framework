// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* StandardCaseFunction.h                                      (C) 2000-2025 */
/*                                                                           */
/* Class managing a standard dataset function.                               */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_STANDARDCASEFUNCTION_H
#define ARCANE_CORE_STANDARDCASEFUNCTION_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/CaseFunction.h"
#include "arcane/core/IStandardFunction.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Class managing a standard dataset function.
 *
 * \ingroup CaseOption
 *
 * This class must be inherited and the derived class must override
 * one of the methods that returns an IFunctionWithArgAndReturn2. Without
 * overriding, all these methods return a null pointer. The derived
 * class is not required to override all the getFunctor* methods
 * but can limit itself to overriding only those it wishes.
 */
class ARCANE_CORE_EXPORT StandardCaseFunction
: public CaseFunction
, public IStandardFunction
{
 public:

  //! Constructs a dataset function.
  explicit StandardCaseFunction(const CaseFunctionBuildInfo& info);
  ~StandardCaseFunction() override;

 private:

  void value(Real param, Real& v) const override;
  void value(Real param, Integer& v) const override;
  void value(Real param, bool& v) const override;
  void value(Real param, String& v) const override;
  void value(Real param, Real3& v) const override;
  void value(Integer param, Real& v) const override;
  void value(Integer param, Integer& v) const override;
  void value(Integer param, bool& v) const override;
  void value(Integer param, String& v) const override;
  void value(Integer param, Real3& v) const override;

 public:

  // NOTE: The 'virtual' keyword is necessary for SWIG
  // because this class has multiple inheritance and SWIG only takes
  // the first one and therefore does not see the following methods
  // as virtual.
  virtual IBinaryMathFunctor<Real, Real, Real>* getFunctorRealRealToReal() override;
  virtual IBinaryMathFunctor<Real, Real3, Real>* getFunctorRealReal3ToReal() override;
  virtual IBinaryMathFunctor<Real, Real, Real3>* getFunctorRealRealToReal3() override;
  virtual IBinaryMathFunctor<Real, Real3, Real3>* getFunctorRealReal3ToReal3() override;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
