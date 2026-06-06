// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VariableComputeFunction.h                                   (C) 2000-2025 */
/*                                                                           */
/* Functor class for variable recalculation.                                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_VARIABLECOMPUTEFUNCTION_H
#define ARCANE_CORE_VARIABLECOMPUTEFUNCTION_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Functor.h"
#include "arcane/utils/TraceInfo.h"

#include "arcane/core/IVariableComputeFunction.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Interface of the variable recalculation functor class.
 */
class VariableComputeFunction
: public IVariableComputeFunction
{
 public:

  template <typename ClassType>
  VariableComputeFunction(ClassType* instance, void (ClassType::*func)())
  : m_functor(new FunctorT<ClassType>(instance, func))
  {
  }
  template <typename ClassType>
  VariableComputeFunction(ClassType* instance, void (ClassType::*func)(), const TraceInfo& tinfo)
  : m_functor(new FunctorT<ClassType>(instance, func))
  , m_trace_info(tinfo)
  {
  }

  //! Releases resources
  ~VariableComputeFunction() override
  {
    delete m_functor;
  }

 public:

  //! Executes the calculation function
  void execute() override { m_functor->executeFunctor(); }

  const TraceInfo& traceInfo() const override { return m_trace_info; }

 private:

  IFunctor* m_functor = nullptr;
  TraceInfo m_trace_info;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
