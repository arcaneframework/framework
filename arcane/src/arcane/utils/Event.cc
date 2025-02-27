// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Event.cc                                                    (C) 2000-2025 */
/*                                                                           */
/* Gestionnaires d'évènements.                                               */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Event.h"
#include "arcane/utils/UniqueArray.h"
#include "arcane/utils/FatalErrorException.h"

#include <set>
#include <iostream>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \file utils/Event.h
 *
 * \brief Fichier contenant les mécanismes de gestion des évènements.
 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class EventObservableBase::Impl
{
 public:
  Impl(){}
 public:
  void rebuildOberversArray()
  {
    m_observers_array.clear();
    m_observers_array.reserve(arcaneCheckArraySize(m_observers.size()));
    for( auto o : m_observers )
      m_observers_array.add(o);
  }
 public:
  std::set<EventObserverBase*> m_auto_destroy_observers;
  std::set<EventObserverBase*> m_observers;
  UniqueArray<EventObserverBase*> m_observers_array;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

EventObservableBase::
EventObservableBase()
: m_p(new Impl())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

EventObservableBase::
~EventObservableBase()
{
  try{
    detachAllObservers();
  }
  catch(...){
    std::cerr << "ERROR: Exception launched during call to ~EventObservableBase().\n";
  }
  delete m_p;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void EventObservableBase::
_attachObserver(EventObserverBase* obs,bool is_auto_destroy)
{
  // Vérifie que l'observeur n'est pas dans la liste.
  if (m_p->m_observers.find(obs)!=m_p->m_observers.end())
    ARCANE_FATAL("Observer is already attached to this observable");
  obs->_notifyAttach(this);
  m_p->m_observers.insert(obs);
  m_p->rebuildOberversArray();
  if (is_auto_destroy)
    m_p->m_auto_destroy_observers.insert(obs);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void EventObservableBase::
_detachObserver(EventObserverBase* obs)
{
  // NOTE: il est impossible de détacher un observeur qui a été alloué
  // dynamiquement. Il n'y a donc pas besoin de mettre à jour
  // m_p->m_auto_destroy_observers.
  bool is_ok = false;
  for( auto o : m_p->m_observers )
    if (o==obs){
      m_p->m_observers.erase(o);
      is_ok = true;
      break;
    }

  // Lance une exception si pas trouvé
  if (!is_ok)
    ARCANE_FATAL("observer is not registered to this observable");
  obs->_notifyDetach();
  m_p->rebuildOberversArray();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ConstArrayView<EventObserverBase*> EventObservableBase::
_observers() const
{
  return m_p->m_observers_array;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void EventObservableBase::
detachAllObservers()
{
  for( auto o : m_p->m_observers )
    o->_notifyDetach();
  m_p->m_observers.clear();
  for( auto o : m_p->m_auto_destroy_observers )
    delete o;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool EventObservableBase::
hasObservers() const
{
  return (m_p->m_observers.size()!=0);
}


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

EventObserverBase::
~EventObserverBase() ARCANE_NOEXCEPT_FALSE
{
  if (m_observable)
    m_observable->_detachObserver(this);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void EventObserverBase::
_notifyDetach()
{
  if (!m_observable)
    ARCANE_FATAL("EventObserver is not attached to an EventObservable");
  m_observable = nullptr;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void EventObserverBase::
_notifyAttach(EventObservableBase* obs)
{
  if (m_observable)
    ARCANE_FATAL("EventObserver is already attached to an EventObservable");
  m_observable = obs;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

EventObserverPool::
~EventObserverPool()
{
  clear();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void EventObserverPool::
clear()
{
  for( auto o : m_observers )
    delete o;
  m_observers.clear();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void EventObserverPool::
add(EventObserverBase* obs)
{
  m_observers.add(obs);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

