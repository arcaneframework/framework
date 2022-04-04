// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Dictionary.h                                                (C) 2000-2009 */
/*                                                                           */
/* Dictionnaire.                                                             */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_DICTIONARY_H
#define ARCANE_UTILS_DICTIONARY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcaneGlobal.h"
#include <map>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename T>
class PointerTraits
{
 public:
  typedef T* InstanceType;
};
template<> class PointerTraits<int>
{
 public:
  typedef int InstanceType;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class Key,class Value>
class Dictionary
{
 public:
  typedef typename PointerTraits<Key>::InstanceType KeyInstanceType;
  typedef typename PointerTraits<Value>::InstanceType ValueInstanceType;
 public:
  Dictionary(int size) {}
  ValueInstanceType item(KeyInstanceType key) { return m_map[key]; }
  void setItem(KeyInstanceType key,ValueInstanceType v)
  {
    throw Arcane::NotImplementedException(A_FUNCINFO);
  }
  void add(KeyInstanceType& key,ValueInstanceType& v)
  {
    m_map.insert(std::make_pair(key,v));
  }
 private:
  std::map<KeyInstanceType,ValueInstanceType> m_map;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
