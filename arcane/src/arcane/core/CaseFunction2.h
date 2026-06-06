// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CaseFunction2.h                                             (C) 2000-2023 */
/*                                                                           */
/* Data set function with explicit value type.                               */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_CASEFUNCTION2_H
#define ARCANE_CORE_CASEFUNCTION2_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Real3.h"
#include "arcane/core/CaseFunction.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Implementation of CaseFunction allowing direct return
 * of the value associated with a parameter without passing through a reference.
 *
 * This is primarily used to simplify C# extensions by avoiding
 * the different overloads of value().
 *
 * To use this class, you must implement the 'valueAs*' methods
 * for the two argument types \a Integer and \a Real and for the
 * different possible return types.
 */
class ARCANE_CORE_EXPORT CaseFunction2
: public CaseFunction
{
 public:

  //! Constructs a data set function.
  explicit CaseFunction2(const CaseFunctionBuildInfo& cfbi)
  : CaseFunction(cfbi)
  {}

 protected:

  void value(Real param, Real& v) const override
  {
    v = valueAsReal(param);
  }
  void value(Real param, Integer& v) const override
  {
    v = valueAsInteger(param);
  }
  void value(Real param, bool& v) const override
  {
    v = valueAsBool(param);
  }
  void value(Real param, String& v) const override
  {
    v = valueAsString(param);
  }
  void value(Real param, Real3& v) const override
  {
    v = valueAsReal3(param);
  }
  void value(Integer param, Real& v) const override
  {
    v = valueAsReal(param);
  }
  void value(Integer param, Integer& v) const override
  {
    v = valueAsInteger(param);
  }
  void value(Integer param, bool& v) const override
  {
    v = valueAsBool(param);
  }
  void value(Integer param, String& v) const override
  {
    v = valueAsString(param);
  }
  void value(Integer param, Real3& v) const override
  {
    v = valueAsReal3(param);
  }

 public:

  virtual Real valueAsReal(Real param) const = 0;
  virtual Integer valueAsInteger(Real param) const = 0;
  virtual bool valueAsBool(Real param) const = 0;
  virtual String valueAsString(Real param) const = 0;
  virtual Real3 valueAsReal3(Real param) const = 0;

  virtual Real valueAsReal(Integer param) const = 0;
  virtual Integer valueAsInteger(Integer param) const = 0;
  virtual bool valueAsBool(Integer param) const = 0;
  virtual String valueAsString(Integer param) const = 0;
  virtual Real3 valueAsReal3(Integer param) const = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
