// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SimpleProperty.h                                            (C) 2000-2025 */
/*                                                                           */
/* Implémentation basique d'une propriété.                                   */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_SIMPLEPROPERTY_H
#define ARCANE_CORE_SIMPLEPROPERTY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcaneGlobal.h"
#include "arcane/utils/String.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class T>
class SimplePropertyTraitsT
{
 public:
  typedef const T& ConstReferenceType;
  typedef T& ReferenceType;
  typedef T ValueType;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<>
class SimplePropertyTraitsT<String>
{
 public:
  typedef const String& ConstReferenceType;
  typedef String& ReferenceType;
  typedef String ValueType;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Implémentation basique d'une propriété en lecture seule.
 */
template<class T>
class SimpleReadOnlyPropertyT
{
 public:

  typedef typename SimplePropertyTraitsT<T>::ConstReferenceType ConstReferenceType;
  typedef typename SimplePropertyTraitsT<T>::ReferenceType ReferenceType;
  typedef typename SimplePropertyTraitsT<T>::ValueType ValueType;

 public:

  SimpleReadOnlyPropertyT() : m_value(T()) {}
  SimpleReadOnlyPropertyT(ConstReferenceType v) : m_value(v) {}

 public:

  ConstReferenceType get() const { return m_value; }

 protected:

  ValueType m_value;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Implémentation basique d'une propriété.
 */
template<class T>
class SimplePropertyT
: public SimpleReadOnlyPropertyT<T>
{
 public:

  typedef typename SimplePropertyTraitsT<T>::ConstReferenceType ConstReferenceType;
  typedef typename SimplePropertyTraitsT<T>::ReferenceType ReferenceType;
  typedef typename SimplePropertyTraitsT<T>::ValueType ValueType;

 public:

  SimplePropertyT() : SimpleReadOnlyPropertyT<T>() {}
  SimplePropertyT(ConstReferenceType v) : SimpleReadOnlyPropertyT<T>(v) {}

 public:
		  
  inline ConstReferenceType get() const { return this->m_value; }
  inline ReferenceType get() { return this->m_value; }
  inline void put(ConstReferenceType v) { this->m_value = v; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
