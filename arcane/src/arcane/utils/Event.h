// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Event.h                                                     (C) 2000-2025 */
/*                                                                           */
/* Gestionnaires d'évènements.                                               */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_EVENT_H
#define ARCANE_UTILS_EVENT_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/UtilsTypes.h"
#include "arcane/utils/Array.h"

#include <functional>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Classe de base d'un handler d'évènement.
 */
class ARCANE_UTILS_EXPORT EventObservableBase
{
  friend EventObserverBase;
  class Impl;

 public:

  EventObservableBase();
  virtual ~EventObservableBase();

 public:

  EventObservableBase(const EventObservableBase&) = delete;
  EventObservableBase(EventObservableBase&&) = delete;
  EventObservableBase& operator=(const EventObservableBase&) = delete;
  EventObservableBase& operator=(EventObservableBase&&) = delete;

 public:

  bool hasObservers() const { return !m_observers_array.empty(); }
  void detachAllObservers();

 protected:

  void _attachObserver(EventObserverBase* obs, bool is_auto_destroy);
  void _detachObserver(EventObserverBase* obs);
  ConstArrayView<EventObserverBase*> _observers() const
  {
    return m_observers_array;
  }

 private:

  Impl* m_p = nullptr;
  UniqueArray<EventObserverBase*> m_observers_array;

 private:

  void _rebuildObserversArray();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Classe de base d'un observateur d'évènement.
 *
 * L'ajoute ou la suppression d'un observeur se fait via les opérateurs
 * EventObservable::operator+=() et EventObservable::operator-=().
 */
class ARCANE_UTILS_EXPORT EventObserverBase
{
  friend class EventObservableBase;

 public:

  EventObserverBase() = default;
  virtual ~EventObserverBase() ARCANE_NOEXCEPT_FALSE;

 protected:

  void _notifyDetach();
  void _notifyAttach(EventObservableBase* obs);

 private:

  EventObservableBase* m_observable = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup Utils
 * \brief Observateur d'évènements.
 */
template <typename... Args>
class EventObserver
: public EventObserverBase
{
 public:

  typedef EventObservable<Args...> ObservableType;

 public:

  EventObserver() {}
  EventObserver(const std::function<void(Args...)>& func)
  : m_functor(func)
  {}
  EventObserver(std::function<void(Args...)>&& func)
  : m_functor(func)
  {}
  void observerUpdate(Args... args)
  {
    if (m_functor)
      m_functor(args...);
  }

 private:

  std::function<void(Args...)> m_functor;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Conserve des références d'observateurs.
 */
class ARCANE_UTILS_EXPORT EventObserverPool
{
 public:

  ~EventObserverPool();

 public:

  //! Ajoute l'observateur \a x
  void add(EventObserverBase* x);
  //! Supprime tous les observateurs associés à cette instance.
  void clear();

 private:

  UniqueArray<EventObserverBase*> m_observers;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup Utils
 * \brief Classe de base d'un handler d'évènement.
 *
 * Les instances de cette classe ne peuvent pas être copiées.
 *
 * Cette classe permet d'enregistrer une liste d'observateurs qui peuvent
 * être notifiés lors de l'appel à notify(). \a Args contient
 * la liste des paramètres de notification.
 *
 * Il est possible d'ajouter un observateur via la méthode attach(). Si
 * l'observateur est une fonction lambda, il est nécessaire de
 * spécifier une instance de EventObserverPool pour gérer la durée
 * de vie de la lambda, qui sera alors la même que celle de
 * l'EventObserverPool associé.
 *
 \code
 * //! Evènement appelant une méthode void f(int,double):
 * EventObservable<int,double> observable;
 * EventObserverPool pool;
 * observable.attach(pool,[](int a,double b) { ... });
 * std::function<int,double> f2(...);
 * auto o = new EventObserver<int,double>(f2);
 * observable.attach(o);
 * observable.notify(1,3.2);
 \endcode
 *
 */
template <typename... Args>
class EventObservable
: public EventObservableBase
{
 public:

  typedef EventObserver<Args...> ObserverType;

 public:

  EventObservable() = default;

 public:

  /*!
   * \brief Attache l'observateur \a o à cet observable.
   *
   * Une exception est levée si l'observateur est déjà attaché à un observable.
   */
  void attach(ObserverType* o) { _attachObserver(o, false); }
  /*!
   * \brief Détache l'observateur \a o de cet observable.
   *
   * Une exception est levée si l'observateur n'est pas attaché à cet observable.
   */
  void detach(ObserverType* o) { _detachObserver(o); }

  /*!
   * \brief Ajoute un observateur utilisant la lambda \a lambda
   * et conserve une référence dans \a pool.
   */
  template <typename Lambda>
  void attach(EventObserverPool& pool, const Lambda& lambda)
  {
    auto x = new ObserverType(lambda);
    _attachObserver(x, false);
    pool.add(x);
  }

  //! Appelle les observeurs associés à cet observable.
  void notify(Args... args)
  {
    if (!hasObservers())
      return;
    for (auto o : _observers())
      ((ObserverType*)o)->observerUpdate(args...);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup Utils
 * \brief Classe gérant les observateurs associés à un évènement.
 * \sa EventObservable
 */
template <typename... Args>
class EventObservableView
{
 public:

  typedef EventObserver<Args...> ObserverType;

 public:

  explicit EventObservableView(EventObservable<Args...>& v)
  : m_observable_ref(v)
  {}

 public:

  /*!
   * \brief Attache l'observateur \a o à cet observable.
   *
   * Une exception est levée si l'observateur est déjà attaché à un observable.
   */
  void attach(ObserverType* o) { m_observable_ref.attach(o); }
  /*!
   * \brief Détache l'observateur \a o de cet observable.
   *
   * Une exception est levée si l'observateur n'est pas attaché à cet observable.
   */
  void detach(ObserverType* o) { m_observable_ref.detach(o); }

  /*!
   * \brief Ajoute un observateur utilisant la lambda \a lambda
   * et conserve une référence dans \a pool.
   */
  template <typename Lambda>
  void attach(EventObserverPool& pool, const Lambda& lambda)
  {
    m_observable_ref.attach(pool, lambda);
  }

 private:

  EventObservable<Args...>& m_observable_ref;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
