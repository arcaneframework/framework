// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Functor.h                                                   (C) 2000-2025 */
/*                                                                           */
/* Utility classes for managing functors.                                    */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_FUNCTOR_H
#define ARCCORE_BASE_FUNCTOR_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/IFunctor.h"

#include <functional>

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
template<typename T>
class FunctorT
: public IFunctor
{
 public:
	
  typedef void (T::*FuncPtr)(); //!< Type of the method pointer

 public:
	
  //! Constructor
  FunctorT(T* object,FuncPtr funcptr)
  : m_function(funcptr), m_object(object){}

  ~FunctorT() override { }

 protected:

  //! Executes the associated method
  void executeFunctor() override
  {
    (m_object->*m_function)();
  }

 private:
  FuncPtr m_function; //!< Pointer to the associated method.
  T* m_object; //!< Associated object.
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Functor associated with a method of a class \a T.
 */
template<typename ClassType,typename ArgType>
class FunctorWithArgumentT
: public IFunctorWithArgumentT<ArgType>
{
 public:
	
  typedef void (ClassType::*FuncPtr)(ArgType); //!< Type of the method pointer

 public:
	
  //! Constructor
  FunctorWithArgumentT(ClassType* object,FuncPtr funcptr)
  : m_object(object), m_function(funcptr) {}

 protected:

  //! Executes the associated method
  void executeFunctor(ArgType arg)
  {
    (m_object->*m_function)(arg);
  }
  
 private:

  ClassType* m_object; //!< Associated object.
  FuncPtr m_function; //!< Pointer to the associated method.
};


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Generic Functor using the std::function class.
 */
template<typename ArgType>
class StdFunctorWithArgumentT
: public IFunctorWithArgumentT<ArgType>
{
 public:
	
  //! Constructor
  StdFunctorWithArgumentT(const std::function<void(ArgType)>& function)
  : m_function(function) {}

 public:

 protected:

  //! Executes the associated method
  void executeFunctor(ArgType arg)
  {
    m_function(arg);
  }
  
 private:

  std::function<void(ArgType)> m_function;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
