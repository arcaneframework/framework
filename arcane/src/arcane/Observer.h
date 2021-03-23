// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Observer.h                                                  (C) 2000-2009 */
/*                                                                           */
/* Observateur.                                                              */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_OBSERVER_H
#define ARCANE_OBSERVER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/IObserver.h"
#include "arcane/IObservable.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IObservable;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Classe de base d'un observateur.
 *
 */
class ARCANE_CORE_EXPORT AbstractObserver
: public IObserver
{
 public:

  AbstractObserver() : m_observable(0) {}
  virtual ~AbstractObserver();

 public:
  
  //! S'attache à l'observable \a obs
  virtual void attachToObservable(IObservable* obs);

  //! Se détache de l'observable
  virtual void detach();

 public:

  virtual void observerUpdate(IObservable* iob) =0;

 private:

  IObservable* m_observable;
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
  : m_object(object), m_function0(0), m_function1(funcptr) {}
  ObserverT(T* object,Func0Ptr funcptr)
  : m_object(object), m_function0(funcptr), m_function1(0) {}

 public:
  

 public:

  //! Exécute la méthode associé
  virtual void observerUpdate(IObservable* iob)
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

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

