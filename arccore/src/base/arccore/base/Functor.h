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
/* Functor.h                                                   (C) 2000-2018 */
/*                                                                           */
/* Classes utilitaires pour gérer des fonctors.                              */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_FUNCTOR_H
#define ARCCORE_BASE_FUNCTOR_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/IFunctor.h"

#include <functional>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Functor associé à une méthode d'une classe \a T.
 */
template<typename T>
class FunctorT
: public IFunctor
{
 public:
	
  typedef void (T::*FuncPtr)(); //!< Type du pointeur sur la méthode

 public:
	
  //! Constructeur
  FunctorT(T* object,FuncPtr funcptr)
  : m_function(funcptr), m_object(object){}

  ~FunctorT() override { }

 protected:

  //! Exécute la méthode associé
  void executeFunctor() override
  {
    (m_object->*m_function)();
  }

 private:
  FuncPtr m_function; //!< Pointeur vers la méthode associée.
  T* m_object; //!< Objet associé.
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Functor associé à une méthode d'une classe \a T.
 */
template<typename ClassType,typename ArgType>
class FunctorWithArgumentT
: public IFunctorWithArgumentT<ArgType>
{
 public:
	
  typedef void (ClassType::*FuncPtr)(ArgType); //!< Type du pointeur sur la méthode

 public:
	
  //! Constructeur
  FunctorWithArgumentT(ClassType* object,FuncPtr funcptr)
  : m_object(object), m_function(funcptr) {}

 protected:

  //! Exécute la méthode associé
  void executeFunctor(ArgType arg)
  {
    (m_object->*m_function)(arg);
  }
  
 private:

  ClassType* m_object; //!< Objet associé.
  FuncPtr m_function; //!< Pointeur vers la méthode associée.
};


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Functor générique utilisant la classe std::function.
 */
template<typename ArgType>
class StdFunctorWithArgumentT
: public IFunctorWithArgumentT<ArgType>
{
 public:
	
  //! Constructeur
  StdFunctorWithArgumentT(const std::function<void(ArgType)>& function)
  : m_function(function) {}

 public:

 protected:

  //! Exécute la méthode associé
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

