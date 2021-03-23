// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Observable.h                                                (C) 2000-2016 */
/*                                                                           */
/* Observable.                                                               */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_OBSERVABLE_H
#define ARCANE_OBSERVABLE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Array.h"

#include "arcane/IObservable.h"
#include "arcane/IObserver.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IObserver;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup Core
 * \brief Classe de base d'un observable.
 *
 * Un observable ne peut pas être copié.
 */
class ARCANE_CORE_EXPORT Observable
: public IObservable
{
 public:

  virtual ~Observable(); //!< Libère les ressources

 public:

  Observable() : m_is_destroyed(false) {}

 public:

  Observable(const Observable& rhs) = delete;
  void operator=(const Observable& rhs) = delete;

 public:

  void destroy() override;
  void attachObserver(IObserver* obs) override;
  void detachObserver(IObserver* obs) override;
  void notifyAllObservers() override;
  bool hasObservers() const override;
  bool isDestroyed() const override;
  void detachAllObservers() override;

 protected:

  void _detachAllObservers();

 private:

  bool m_is_destroyed;
  UniqueArray<IObserver*> m_observers;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Observable qui appelle automatiquement
 * IObservable::detachAllObservers() dans le destructeur.
 */
class ARCANE_CORE_EXPORT AutoDetachObservable
: public Observable
{
 public:
  AutoDetachObservable() : Observable(){}
  ~AutoDetachObservable();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

