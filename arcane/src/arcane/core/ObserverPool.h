// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ObserverPool.h                                              (C) 2000-2025 */
/*                                                                           */
/* Liste d'observateurs.                                                     */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_OBSERVERPOOL_H
#define ARCANE_CORE_OBSERVERPOOL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/List.h"

#include "arcane/core/IObservable.h"
#include "arcane/core/Observer.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Liste d'observateurs
 *
 * Cette classe permet de gérer une liste d'observateurs et d'assurer
 * leur destruction lorsque l'objet cible de l'observateur est détruit.
 */
class ARCANE_CORE_EXPORT ObserverPool
{
 public:

  typedef Collection<IObserver*> ObserverCollection;

 public:

  //! Constructeur
  ObserverPool() {}
  ~ObserverPool(); //!< Libère les ressources

 public:

  //! Ajoute un observateur
  template <class T> inline void
  addObserver(T* obj, void (T::*func)(const IObservable&), IObservable* oba)
  {
    IObserver* obs = new ObserverT<T>(obj, func);
    oba->attachObserver(obs);
    m_observers.add(obs);
  }

  //! Ajoute un observateur
  template <class T> inline void
  addObserver(T* obj, void (T::*func)(), IObservable* oba)
  {
    IObserver* obs = new ObserverT<T>(obj, func);
    oba->attachObserver(obs);
    m_observers.add(obs);
  }

  //! Liste des observateurs
  ObserverCollection observers() { return m_observers; }

  //! Suppression des observateurs (detache aussi par la meme occasion)
  void detachAll();

 private:

  List<IObserver*> m_observers; //!< Liste des observateurs
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

