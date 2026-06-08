// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IFunctor.h                                                  (C) 2000-2025 */
/*                                                                           */
/* Interface of a functor.                                                   */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_IFUNCTOR_H
#define ARCCORE_BASE_IFUNCTOR_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ArccoreGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Interface of a functor.
 * \ingroup Core
 */
class ARCCORE_BASE_EXPORT IFunctor
{
 public:

  //! Releases resources
  virtual ~IFunctor() {}

 public:

  //! Executes the associated method
  virtual void executeFunctor() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Interface of a functor with an argument but without a return value
 */
template <typename ArgType>
class IFunctorWithArgumentT
{
 public:

  //! Releases resources
  virtual ~IFunctorWithArgumentT() {}

 protected:

  IFunctorWithArgumentT() {}
  IFunctorWithArgumentT(const IFunctorWithArgumentT<ArgType>& rhs) = default;
  IFunctorWithArgumentT<ArgType>& operator=(const IFunctorWithArgumentT<ArgType>& rhs) = default;

 public:

  //! Executes the associated method
  virtual void executeFunctor(ArgType arg) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Interface of a functor with 2 arguments and a return value.
 */
template <typename ReturnType, typename Arg1, typename Arg2>
class IFunctorWithArgAndReturn2
{
 public:

  virtual ~IFunctorWithArgAndReturn2() {}

 public:

  virtual ReturnType executeFunctor(Arg1 a1, Arg2 a2) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
