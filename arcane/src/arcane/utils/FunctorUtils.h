// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* FunctorUtils.h                                              (C) 2000-2018 */
/*                                                                           */
/* Utility functions for functors.                                           */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_FUNCTORUTILS_H
#define ARCANE_UTILS_FUNCTORUTILS_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/Functor.h"
#include "arcane/utils/UtilsTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace functor
{
//! Specialization for a lambda function with one argument
template<typename LambdaType,typename A1> StdFunctorWithArgumentT<A1>
make(const LambdaType& f,void (LambdaType::*)(A1 r) const)
{
  return StdFunctorWithArgumentT<A1>(f);
}

/*!
 * \brief Creates and returns a functor for the lambda function \a f.
 *
 * The lambda \a f must correspond to a function with the
 * prototype void(ArgType).
 * The returned object is of type IFunctorWithArgumentT<ArgType> with
 * \a ArgType being the only parameter of \a f.
 */
template<typename LambdaType> auto
make(const LambdaType& f) -> decltype(make(f,&LambdaType::operator()))
{
  return make(f,&LambdaType::operator());
}

//! Specialization for a lambda function with one argument
template<typename LambdaType,typename A1>  StdFunctorWithArgumentT<A1>*
makePointer(const LambdaType& f,void (LambdaType::*)(A1 r) const)
{
  return new StdFunctorWithArgumentT<A1>(f);
}

/*!
 * \brief Creates and returns a pointer to a functor for the lambda function \a f.
 *
 * The lambda \a f must correspond to a function with the
 * prototype void(ArgType).
 * The returned pointer is of type IFunctorWithArgumentT<ArgType> with
 * \a ArgType being the only parameter of \a f.
 * The returned pointer must be destroyed by the delete operator.
 */
template<typename LambdaType> auto
makePointer(const LambdaType& f) -> decltype(makePointer(f,&LambdaType::operator()))
{
  return makePointer(f,&LambdaType::operator());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Applies a functor derived from a lambda function to a given method.
 *
 * This function allows a lambda function to be applied directly
 * to a method that takes an instance of
 * IFunctorWithArgumentT.
 *
 * For example, if we have the following class:
 *
 * \code
 * class A { void visit(IFunctorWithArgumentT<int>*); };
 * \encode
 *
 * It can be used as follows:
 *
 * \code
 * void f()
 * {
 *   A a;
 *   auto f = [](int x) { std::cout << "X=" << x << '\n'; };
 *
 *   functor::apply(&a,&A::visit,f);
 * }
 * \endcode
 */
template<typename LambdaType,typename T,typename ArgType> void
apply(T* x,void (T::*ptr)(IFunctorWithArgumentT<ArgType>*),const LambdaType& f)
{
  auto xstr = make(f);
  (x->*ptr)(&xstr);
}
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
