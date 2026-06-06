// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IVariableComputeFunction.h                                  (C) 2000-2025 */
/*                                                                           */
/* Interface of the functor class for recalculating a variable.              */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IVARIABLECOMPUTEFUNCTION_H
#define ARCANE_CORE_IVARIABLECOMPUTEFUNCTION_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcaneGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Interface of the functor class for recalculating a variable.
 */
class IVariableComputeFunction
{
 public:

  virtual ~IVariableComputeFunction() = default; //!< Frees resources

 public:

  //! Executes the calculation function
  virtual void execute() = 0;

  //! Trace information of the calculation function definition
  virtual const TraceInfo& traceInfo() const = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
