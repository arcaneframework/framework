// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Event.h                                                     (C) 2000-2025 */
/*                                                                           */
/* Event Handlers.                                                           */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_COMMON_EVENT_H
#define ARCCORE_COMMON_EVENT_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/common/Array.h"

#include <functional>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Base class for an event handler.
 */
class ARCCORE_COMMON_EXPORT EventObservableBase
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
 * \brief Base class for an event observer.
 *
 * Adding or removing an observer is done via the operators
 * EventObservable::operator+=() and EventObservable::operator-=().
 */
class ARCCORE_COMMON_EXPORT EventObserverBase
{
  friend class EventObservableBase;

 public:

  EventObserverBase() = default;
  virtual ~EventObserverBase() ARCCORE_NOEXCEPT_FALSE;

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
 * \brief Event observer.
 */
template <typename... Args>
class EventObserver
: public EventObserverBase
{
 public:

  using ObservableType = EventObservable<Args...>;

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
 * \brief Stores references to observers.
 */
class ARCCORE_COMMON_EXPORT EventObserverPool
{
 public:

  ~EventObserverPool();

 public:

  //! Adds the observer \a x
  void add(EventObserverBase* x);
  //! Clears all observers associated with this instance.
  void clear();

 private:

  UniqueArray<EventObserverBase*> m_observers;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup Utils
 * \brief Base class for an event handler.
 *
 * Instances of this class cannot be copied.
 *
 * This class allows registering a list of observers that can
 * be notified when notify() is called. \a Args contains
 * the list of notification parameters.
 *
 * It is possible to add an observer via the attach() method. If
 * the observer is a lambda function, it is necessary to
 * specify an instance of EventObserverPool to manage the lifetime
 * of the lambda, which will then be the same as that of
 * the associated EventObserverPool.
 * \code
 * //! Event calling a method void f(int,double):
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
   * \brief Attaches the observer \a o to this observable.
   *
   * An exception is thrown if the observer is already attached to an observable.
   */
  void attach(ObserverType* o) { _attachObserver(o, false); }

  /*!
   * \brief Detaches the observer \a o from this observable.
   *
   * An exception is thrown if the observer is not attached to this observable.
   */
  void detach(ObserverType* o) { _detachObserver(o); }

  /*!
   * \brief Adds an observer using the lambda \a lambda
   * and stores a reference in \a pool.
   */
  template <typename Lambda>
  void attach(EventObserverPool& pool, const Lambda& lambda)
  {
    auto x = new ObserverType(lambda);
    _attachObserver(x, false);
    pool.add(x);
  }

  //! Calls the observers associated with this observable.
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
 * \brief Class managing observers associated with an event.
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
   * \brief Attaches the observer \a o to this observable.
   *
   * An exception is thrown if the observer is already attached to an observable.
   */
  void attach(ObserverType* o) { m_observable_ref.attach(o); }

  /*!
   * \brief Detaches the observer \a o from this observable.
   *
   * An exception is thrown if the observer is not attached to this observable.
   */
  void detach(ObserverType* o) { m_observable_ref.detach(o); }

  /*!
   * \brief Adds an observer using the lambda \a lambda
   * and stores a reference in \a pool.
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
