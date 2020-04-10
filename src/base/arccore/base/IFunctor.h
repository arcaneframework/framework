// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2020 IFPEN-CEA
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IFunctor.h                                                  (C) 2000-2018 */
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

namespace Arccore
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

