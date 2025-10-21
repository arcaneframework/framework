// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Observer.h                                                  (C) 2000-2025 */
/*                                                                           */
/* Observateur.                                                              */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_OBSERVER_H
#define ARCCORE_BASE_OBSERVER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/IObserver.h"
#include "arccore/base/IObservable.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Classe de base d'un observateur.
 */
class ARCCORE_BASE_EXPORT AbstractObserver
: public IObserver
{
 public:

  AbstractObserver() = default;
  ~AbstractObserver() override;

 public:

  //! S'attache à l'observable \a obs
  void attachToObservable(IObservable* obs) override;

  //! Se détache de l'observable
  void detach() override;

 private:

  IObservable* m_observable = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup Core
 * \brief Observateur pour un type T
 */
template<class T>
class ObserverT
: public AbstractObserver
{
 public:
	
  typedef void (T::*Func0Ptr)(); //!< Type du pointeur sur la méthode
  typedef void (T::*Func1Ptr)(const IObservable&); //!< Type du pointeur sur la méthode

 public:
	
  //! Constructeur
  ObserverT(T* object,Func1Ptr funcptr)
  : m_object(object), m_function0(nullptr), m_function1(funcptr) {}
  ObserverT(T* object,Func0Ptr funcptr)
  : m_object(object), m_function0(funcptr), m_function1(nullptr) {}

 public:
  

 public:

  //! Exécute la méthode associé
  void observerUpdate(IObservable* iob) override
  {
    if (m_function1)
      (m_object->*m_function1)(*iob);
    if (m_function0)
      (m_object->*m_function0)();
  }

 private:

  T* m_object;    //!< Objet associé.
  Func0Ptr m_function0; //!< Pointeur vers la méthode associée.
  Func1Ptr m_function1; //!< Pointeur vers la méthode associée.
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

