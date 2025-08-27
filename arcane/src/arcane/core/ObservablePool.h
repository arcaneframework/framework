// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ObserverPool.h                                              (C) 2000-2022 */
/*                                                                           */
/* Liste d'observables.                                                      */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_OBSERVABLEPOOL_H
#define ARCANE_OBSERVABLEPOOL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/FatalErrorException.h"

#include "arcane/core/Observable.h"
#include "arcane/core/Observer.h"

#include <map>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IObservable;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Liste d'observables.
 *
 * Cette classe permet de gérer une liste d'observables. Chaque observable
 * est associée à une clé de type \a KeyType.
 */
template<typename KeyType>
class ObservablePool
{
 public:

  typedef std::map<KeyType,IObservable*> ObservableListType;

 public:
	
  //! Constructeur
  ObservablePool() {}
  //! Libère les ressources
  ~ObservablePool()
  {
    for( const auto& x : m_observables ){
      IObservable* o = x.second;
      o->detachAllObservers();
      delete o;
    }
  }

 public:
  
  void add(const KeyType& key)
  {
    IObservable* x = _getIfExists(key);
    if (x)
      ARCANE_FATAL("Observable with current key already exists");
    m_observables.insert(std::make_pair(key,new Observable()));
  }

  IObservable* operator[](const KeyType& key)
  {
    IObservable* x = _getIfExists(key);
    if (!x)
      ARCANE_FATAL("No observable with current key exists");
    return x;
  }

 protected:

 private:

  ObservableListType m_observables; //!< Liste des observables

  IObservable* _getIfExists(const KeyType& key) const
  {
    auto x = m_observables.find(key);
    if (x!=m_observables.end())
      return x->second;
    return nullptr;
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

