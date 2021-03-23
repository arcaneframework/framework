// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* FunctorUtils.h                                              (C) 2000-2018 */
/*                                                                           */
/* Fonctions utilitaires pour les fonctors.                                  */
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
//! Spécialisation pour une fonction lamba avec un argument
template<typename LambdaType,typename A1> StdFunctorWithArgumentT<A1>
make(const LambdaType& f,void (LambdaType::*)(A1 r) const)
{
  return StdFunctorWithArgumentT<A1>(f);
}

/*!
 * \brief Créé et retourne un fonctor pour la fonction lambda \a f.
 *
 * La lambda \a f doit correspondre à une fonction avec le
 * prototype void(ArgType).
 * L'objet retourné est de type IFunctorWithArgumentT<ArgType> avec
 * \a ArgType le seul paramètre de \a f.
 */
template<typename LambdaType> auto
make(const LambdaType& f) -> decltype(make(f,&LambdaType::operator()))
{
  return make(f,&LambdaType::operator());
}

//! Spécialisation pour une fonction lamba avec un argument
template<typename LambdaType,typename A1>  StdFunctorWithArgumentT<A1>*
makePointer(const LambdaType& f,void (LambdaType::*)(A1 r) const)
{
  return new StdFunctorWithArgumentT<A1>(f);
}

/*!
 * \brief Créé et retourne un pointeur pour un fonctor pour la fonction lambda \a f.
 *
 * La lambda \a f doit correspondre à une fonction avec le
 * prototype void(ArgType).
 * Le pointeur retourné est de type IFunctorWithArgumentT<ArgType> avec
 * \a ArgType le seul paramètre de \a f.
 * Le pointeur retourné doit être détruit par l'opérateur delete.
 */
template<typename LambdaType> auto
makePointer(const LambdaType& f) -> decltype(makePointer(f,&LambdaType::operator()))
{
  return makePointer(f,&LambdaType::operator());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Applique un functor issu d'une lambda fonction sur une méthode donnée.
 *
 * Cette fonction permet d'appliquer directement une lambda fonction
 * sur une méthode qui prend en paramètre une instance de
 * IFunctorWithArgumentT.
 *
 * Par exemple, si on a la classe suivante:
 *
 * \code
 * class A { void visit(IFunctorWithArgumentT<int>*); };
 * \encode
 *
 * On peut l'utiliser comme suit:
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

