// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* FunctorAMRTansport.h                                       (C) 2000-2010 */
/*                                                                           */
/* Fonctor avec deux arguments.                                                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_FUNCTORAMRTRANSPORT_H
#define ARCANE_UTILS_FUNCTORAMRTRANSPORT_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/IAMRTransportFunctor.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Functor associé à une méthode d'une classe \a T.
 */
template<typename ClassType>
class AMRTransportFunctorT
: public IAMRTransportFunctor
{
 public:
	
  typedef void (ClassType::*FuncPtr)(Array<ItemInternal*>& , AMROperationType); //!< Type du pointeur sur la méthode
  typedef void (ClassType::*FuncPtr2)(Array<Cell>& , AMROperationType); //!< Type du pointeur sur la méthode
 public:
	
  //! Constructeur
  AMRTransportFunctorT(ClassType* object,FuncPtr funcptr)
  : m_object(object), m_function(funcptr) {}

  AMRTransportFunctorT(ClassType* object,FuncPtr2 funcptr2)
  : m_object(object), m_function2(funcptr2) {}

 protected:

  //! Exécute la méthode associé
  void executeFunctor(Array<ItemInternal*>& old_cells,AMROperationType op)
  {
    (m_object->*m_function)(old_cells,op);
  }
  //! Exécute la méthode associé
  void executeFunctor(Array<Cell>& old_cells,AMROperationType op)
  {
    (m_object->*m_function2)(old_cells,op);
  }
  
 private:

  ClassType* m_object; //!< Objet associé.
  FuncPtr m_function; //!< Pointeur vers la méthode associée.
  FuncPtr2 m_function2; //!< Pointeur vers la méthode associée.
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

