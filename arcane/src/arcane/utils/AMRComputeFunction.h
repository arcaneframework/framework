// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AMRComputeFunction.h                                        (C) 2000-2017 */
/*                                                                           */
/* Variable transport functor class.                                         */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_AMRCOMPUTEFUNCTION_H
#define ARCANE_AMRCOMPUTEFUNCTION_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/AMRTransportFunctor.h"

#include "arcane/core/Item.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 \brief Interface for the CellVariable transport functor class.
 */
class AMRComputeFunction
: public IAMRTransportFunctor
{
 public:

  template <typename ClassType>
  AMRComputeFunction(ClassType* instance, void (ClassType::*func)(Array<ItemInternal*>&, AMROperationType))
  : m_functor(new AMRTransportFunctorT<ClassType>(instance, func))
  {
  }
  template <typename ClassType>
  AMRComputeFunction(ClassType* instance, void (ClassType::*func)(Array<Cell>&, AMROperationType))
  : m_functor(new AMRTransportFunctorT<ClassType>(instance, func))
  {
  }
  virtual ~AMRComputeFunction() { delete m_functor; } //!< Releases resources

 public:

  //! Executes the calculation function
  virtual void executeFunctor(Array<ItemInternal*>& cells, AMROperationType op)
  {
    m_functor->executeFunctor(cells, op);
  }
  //! Executes the calculation function
  virtual void executeFunctor(Array<Cell>& cells, AMROperationType op)
  {
    m_functor->executeFunctor(cells, op);
  }

 private:

  IAMRTransportFunctor* m_functor;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
