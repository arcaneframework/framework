// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IFunctor.h                                                  (C) 2000-2025 */
/*                                                                           */
/* Interface d'un fonctor.                                                   */
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
 * \brief Interface d'un fonctor.
 * \ingroup Core
 */
class ARCCORE_BASE_EXPORT IFunctor
{
 public:
	
  //! Libère les ressources
  virtual ~IFunctor(){}

 public:

  //! Exécute la méthode associé
  virtual void executeFunctor() =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface d'un fonctor avec argument mais sans valeur de retour
 */
template<typename ArgType>
class IFunctorWithArgumentT
{
 public:
	
  //! Libère les ressources
  virtual ~IFunctorWithArgumentT() {}

 protected:

  IFunctorWithArgumentT() {}
  IFunctorWithArgumentT(const IFunctorWithArgumentT<ArgType>& rhs) = default;
  IFunctorWithArgumentT<ArgType>& operator=(const IFunctorWithArgumentT<ArgType>& rhs) = default;

 public:

  //! Exécute la méthode associé
  virtual void executeFunctor(ArgType arg) =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface d'un fonctor avec 2 arguments et une valeur de retour.
 */
template<typename ReturnType,typename Arg1,typename Arg2>
class IFunctorWithArgAndReturn2
{
 public:

  virtual ~IFunctorWithArgAndReturn2(){}

 public:

  virtual ReturnType executeFunctor(Arg1 a1,Arg2 a2) =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

