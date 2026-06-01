// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* FunctorAMRTansport.h                                        (C) 2000-2022 */
/*                                                                           */
/* Functor with two arguments.                                               */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_FUNCTORAMRTRANSPORT_H
#define ARCANE_UTILS_FUNCTORAMRTRANSPORT_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/IAMRTransportFunctor.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Functor associated with a method of a class \a T.
 */
template <typename ClassType>
class AMRTransportFunctorT
: public IAMRTransportFunctor
{
 public:

  typedef void (ClassType::*FuncPtr)(Array<ItemInternal*>&, AMROperationType); //!< Type of the method pointer
  typedef void (ClassType::*FuncPtr2)(Array<Cell>&, AMROperationType); //!< Type of the method pointer
 public:

  //! Constructor
  AMRTransportFunctorT(ClassType* object, FuncPtr funcptr)
  : m_object(object)
  , m_function(funcptr)
  {}

  AMRTransportFunctorT(ClassType* object, FuncPtr2 funcptr2)
  : m_object(object)
  , m_function2(funcptr2)
  {}

 protected:

  //! Executes the associated method
  void executeFunctor(Array<ItemInternal*>& old_cells, AMROperationType op)
  {
    (m_object->*m_function)(old_cells, op);
  }
  //! Executes the associated method
  void executeFunctor(Array<Cell>& old_cells, AMROperationType op)
  {
    (m_object->*m_function2)(old_cells, op);
  }

 private:

  ClassType* m_object = nullptr; //!< Associated object.
  FuncPtr m_function = nullptr; //!< Pointer to the associated method.
  FuncPtr2 m_function2 = nullptr; //!< Pointer to the associated method.
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
