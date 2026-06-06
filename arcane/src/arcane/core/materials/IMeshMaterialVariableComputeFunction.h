// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMeshMaterialVariableComputeFunction.h                      (C) 2000-2022 */
/*                                                                           */
/* Interface of the functor class for recalculating a variable.              */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_MATERIALS_IMESHMATERIALVARIABLECOMPUTEFUNCTION_H
#define ARCANE_CORE_MATERIALS_IMESHMATERIALVARIABLECOMPUTEFUNCTION_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/materials/MaterialsCoreGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 \brief Interface of the functor class for recalculating a variable.
 */
class ARCANE_CORE_EXPORT IMeshMaterialVariableComputeFunction
{
 public:

  virtual ~IMeshMaterialVariableComputeFunction(){} //!< Releases resources

 public:

  //! Executes the calculation function
  virtual void execute(IMeshMaterial* mat) =0;

  //! Trace information for the calculation function definition
  virtual const TraceInfo& traceInfo() const =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
