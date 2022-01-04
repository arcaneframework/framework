// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* BuiltInProxy.h                                              (C) 2000-2006 */
/*                                                                           */
/* Proxy d'un type du langage.                                               */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_BUILTINPROXY_H
#define ARCANE_UTILS_BUILTINPROXY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Numeric.h"
#include "arcane/utils/MemoryAccessInfo.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Proxy d'un type du langage.
 */
template<typename Type>
class BuiltInProxy
{
 public:
  
  typedef BuiltInProxy<Type> ThatClassType;

 public:

 
  BuiltInProxy(Type& ref,const MemoryAccessInfo& info)
  : m_value(ref), m_info(info) {}
  BuiltInProxy(const ThatClassType& f)
  : m_value(f.m_value), m_info(f.m_info) {}
  const Type& operator=(const ThatClassType& f)
    { setValue(f.m_value); return m_value; }
  const Type& operator=(Type v)
    { setValue(v); return m_value; }
  //operator Type&()
  //{ return getValueMutable(); }
  operator Type() const
    { return getValue(); }

 public:

  ThatClassType& operator+= (const Type& b)
    {
      return setValue(getValue()+b);
    }
  void operator++ ()
    {
      setValue(getValue()+1);
    }
  Type operator++ (int)
    {
      Type x = getValue();
      setValue(x+1);
      return x;
    }
  void operator-- ()
    {
      setValue(getValue()-1);
    }
  Type operator-- (int)
    {
      Type x = getValue();
      setValue(x-1);
      return x;
    }
  ThatClassType& operator-= (const Type& b)
    {
      return setValue(getValue()-b);
    }
  ThatClassType& operator*= (const Type& b)
    {
      return setValue(getValue()*b);
    }
  ThatClassType& operator/= (const Type& b)
    {
      return setValue(getValue()/b);
    }
		
 public:

  ThatClassType& setValue(const Type& v)
    {
      m_value = v;
      m_info.setWrite();
      return (*this);
    }
  Type getValue() const
    {
      m_info.setRead();
      return m_value;
    } 
  Type& getValueMutable()
    {
      m_info.setReadOrWrite();
      return m_value;
    } 
 private:
  
  Type& m_value;
  MemoryAccessInfo m_info;
};

template<typename Type> inline
bool operator==(const BuiltInProxy<Type>& a,const BuiltInProxy<Type>& b)
{
  return a.getValue()==b.getValue();
}
template<typename Type> inline
bool operator==(const BuiltInProxy<Type>& a,const Type& b)
{
  return a.getValue()==b;
}
template<typename Type> inline
bool operator==(const Type& a,const BuiltInProxy<Type>& b)
{
  return a==b.getValue();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Lit le triplet \a t à partir du flot \a o.
 * \relates Real3
 */
template<typename Type> inline std::istream&
operator>> (std::istream& i,BuiltInProxy<Type>& t)
{
  Type v;
  i >> v;
  t.setValue(v);
  return i;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace math
{
template<class _Type> inline bool
isNearlyZero(const BuiltInProxy<_Type>& a)
{
  return TypeEqualT<_Type>::isNearlyZero(a);
}
/*!
 * \brief Teste si une valeur est exactement égale à zéro.
 * \retval true si \a vaut zéro,
 * \retval false sinon.
 */
template<class _Type> inline bool
isZero(const BuiltInProxy<_Type>& a)
{
  return TypeEqualT<_Type>::isZero(a);
}
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
