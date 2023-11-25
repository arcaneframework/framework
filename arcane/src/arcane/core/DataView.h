// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DataView.h                                                  (C) 2000-2023 */
/*                                                                           */
/* Vues sur des données des variables.                                       */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_DATAVIEW_H
#define ARCANE_CORE_DATAVIEW_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/datatype/DataTypeTraits.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \file DataView.h
 *
 * Ce fichier contient les déclarations des types pour gérer
 * les vues pour les accélérateurs.
 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator::impl
{
class AtomicImpl;
}
namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe pour accéder à un élément d'une vue en lecture.
 */
template <typename DataType>
class DataViewGetter
{
 public:

  using ValueType = const DataType;
  using AccessorReturnType = const DataType;
  static ARCCORE_HOST_DEVICE AccessorReturnType build(const DataType* ptr)
  {
    return { *ptr };
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe pour accéder à un élément d'une vue en écriture.
 */
template <typename DataType>
class DataViewSetter
{
  // Pour accéder à m_ptr;
  friend class DataViewGetterSetter<DataType>;

 public:

  using ValueType = DataType;
  using AccessorReturnType = DataViewSetter<DataType>;

 public:

  explicit ARCCORE_HOST_DEVICE DataViewSetter(DataType* ptr)
  : m_ptr(ptr)
  {}
  ARCCORE_HOST_DEVICE DataViewSetter(const DataViewSetter<DataType>& v)
  : m_ptr(v.m_ptr)
  {}
  ARCCORE_HOST_DEVICE DataViewSetter<DataType>&
  operator=(const DataType& v)
  {
    *m_ptr = v;
    return (*this);
  }
  ARCCORE_HOST_DEVICE DataViewSetter<DataType>&
  operator=(const DataViewSetter<DataType>& v)
  {
    // Attention: il faut mettre à jour la valeur et pas le pointeur
    // sinon le code tel que a = b avec 'a' et 'b' deux instances de cette
    // classe ne fonctionnera pas.
    *m_ptr = *(v.m_ptr);
    return (*this);
  }
  static ARCCORE_HOST_DEVICE AccessorReturnType build(DataType* ptr)
  {
    return AccessorReturnType(ptr);
  }

 public:

  // Binary arithmetic operators
  // +=
  ARCCORE_HOST_DEVICE DataViewSetter<DataType>&
  operator+=(const DataType& v)
  {
    *m_ptr = (*m_ptr) + v;
    return (*this);
  }
  ARCCORE_HOST_DEVICE DataViewSetter<DataType>&
  operator+=(const DataViewSetter<DataType>& v)
  {
    *m_ptr = (*m_ptr) + *(v.m_ptr);
    return (*this);
  }

  // -=
  ARCCORE_HOST_DEVICE DataViewSetter<DataType>&
  operator-=(const DataType& v)
  {
    *m_ptr = (*m_ptr) - v;
    return (*this);
  }
  ARCCORE_HOST_DEVICE DataViewSetter<DataType>&
  operator-=(const DataViewSetter<DataType>& v)
  {
    *m_ptr = (*m_ptr) - *(v.m_ptr);
    return (*this);
  }

  // *=
  ARCCORE_HOST_DEVICE DataViewSetter<DataType>&
  operator*=(const DataType& v)
  {
    *m_ptr = (*m_ptr) * v;
    return (*this);
  }
  ARCCORE_HOST_DEVICE DataViewSetter<DataType>&
  operator*=(const DataViewSetter<DataType>& v)
  {
    *m_ptr = (*m_ptr) * *(v.m_ptr);
    return (*this);
  }

  // /=
  ARCCORE_HOST_DEVICE DataViewSetter<DataType>&
  operator/=(const DataType& v)
  {
    *m_ptr = (*m_ptr) / v;
    return (*this);
  }
  ARCCORE_HOST_DEVICE DataViewSetter<DataType>&
  operator/=(const DataViewSetter<DataType>& v)
  {
    *m_ptr = (*m_ptr) / *(v.m_ptr);
    return (*this);
  }

 public:

  template <typename X = DataType, typename = std::enable_if_t<DataTypeTraitsT<X>::HasComponentX()>>
  ARCCORE_HOST_DEVICE void setX(typename DataTypeTraitsT<X>::ComponentType value)
  {
    m_ptr->x = value;
  }
  template <typename X = DataType, typename = std::enable_if_t<DataTypeTraitsT<X>::HasComponentY()>>
  ARCCORE_HOST_DEVICE void setY(typename DataTypeTraitsT<X>::ComponentType value)
  {
    m_ptr->y = value;
  }
  template <typename X = DataType, typename = std::enable_if_t<DataTypeTraitsT<X>::HasComponentZ()>>
  ARCCORE_HOST_DEVICE void setZ(typename DataTypeTraitsT<X>::ComponentType value)
  {
    m_ptr->z = value;
  }

  template <typename X = DataType, typename = std::enable_if_t<DataTypeTraitsT<X>::HasComponentXX()>>
  ARCCORE_HOST_DEVICE void setXX(Real value)
  {
    m_ptr->x.x = value;
  }
  template <typename X = DataType, typename = std::enable_if_t<DataTypeTraitsT<X>::HasComponentYX()>>
  ARCCORE_HOST_DEVICE void setYX(Real value)
  {
    m_ptr->y.x = value;
  }
  template <typename X = DataType, typename = std::enable_if_t<DataTypeTraitsT<X>::HasComponentZX()>>
  ARCCORE_HOST_DEVICE void setZX(Real value)
  {
    m_ptr->z.x = value;
  }

  template <typename X = DataType, typename = std::enable_if_t<DataTypeTraitsT<X>::HasComponentXY()>>
  ARCCORE_HOST_DEVICE void setXY(Real value)
  {
    m_ptr->x.y = value;
  }
  template <typename X = DataType, typename = std::enable_if_t<DataTypeTraitsT<X>::HasComponentYY()>>
  ARCCORE_HOST_DEVICE void setYY(Real value)
  {
    m_ptr->y.y = value;
  }
  template <typename X = DataType, typename = std::enable_if_t<DataTypeTraitsT<X>::HasComponentZY()>>
  ARCCORE_HOST_DEVICE void setZY(Real value)
  {
    m_ptr->z.y = value;
  }

  template <typename X = DataType, typename = std::enable_if_t<DataTypeTraitsT<X>::HasComponentXZ()>>
  ARCCORE_HOST_DEVICE void setXZ(Real value)
  {
    m_ptr->x.z = value;
  }
  template <typename X = DataType, typename = std::enable_if_t<DataTypeTraitsT<X>::HasComponentYZ()>>
  ARCCORE_HOST_DEVICE void setYZ(Real value)
  {
    m_ptr->y.z = value;
  }
  template <typename X = DataType, typename = std::enable_if_t<DataTypeTraitsT<X>::HasComponentZZ()>>
  ARCCORE_HOST_DEVICE void setZZ(Real value)
  {
    m_ptr->z.z = value;
  }
  template <typename X = DataType, typename = std::enable_if_t<DataTypeTraitsT<X>::HasSubscriptOperator()>>
  ARCCORE_HOST_DEVICE DataViewSetter<typename DataTypeTraitsT<X>::SubscriptType> operator[](Int32 index)
  {
    return DataViewSetter<typename DataTypeTraitsT<X>::SubscriptType>(&m_ptr->operator[](index));
  }

 private:

  DataType* m_ptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe pour accéder à un élément d'une vue en lecture/écriture.
 *
 * Cette classe étend les fonctionnalités de DataViewSetter en ajoutant
 * la possibilité d'accéder à la valeur de la donnée.
 */
template <typename DataType>
class DataViewGetterSetter
: public DataViewSetter<DataType>
{
  using BaseType = DataViewSetter<DataType>;
  using BaseType::m_ptr;
  friend class Arcane::Accelerator::impl::AtomicImpl;

 public:

  using ValueType = DataType;
  using AccessorReturnType = DataViewGetterSetter<DataType>;

 public:

  explicit ARCCORE_HOST_DEVICE DataViewGetterSetter(DataType* ptr)
  : BaseType(ptr)
  {}
  ARCCORE_HOST_DEVICE DataViewGetterSetter(const DataViewGetterSetter& v)
  : BaseType(v)
  {}
  ARCCORE_HOST_DEVICE operator DataType() const
  {
    return *m_ptr;
  }
  ARCCORE_HOST_DEVICE DataViewSetter<DataType>&
  operator=(const DataViewGetterSetter<DataType>& v)
  {
    BaseType::operator=(v);
    return (*this);
  }
  ARCCORE_HOST_DEVICE DataViewSetter<DataType>&
  operator=(const DataType& v)
  {
    BaseType::operator=(v);
    return (*this);
  }
  static ARCCORE_HOST_DEVICE AccessorReturnType build(DataType* ptr)
  {
    return AccessorReturnType(ptr);
  }

  template <typename X = DataType, typename = std::enable_if_t<DataTypeTraitsT<X>::HasSubscriptOperator()>>
  ARCCORE_HOST_DEVICE DataViewGetterSetter<typename DataTypeTraitsT<X>::SubscriptType> operator[](Int32 index)
  {
    return DataViewGetterSetter<typename DataTypeTraitsT<X>::SubscriptType>(&m_ptr->operator[](index));
  }

  template <typename X = DataType> constexpr ARCCORE_HOST_DEVICE auto
  operator()(Int32 i0) -> DataViewGetterSetter<typename DataTypeTraitsT<X>::FunctionCall1ReturnType>
  {
    return DataViewGetterSetter<typename DataTypeTraitsT<X>::FunctionCall1ReturnType>(&m_ptr->operator()(i0));
  }

  template <typename X = DataType> constexpr ARCCORE_HOST_DEVICE auto
  operator()(Int32 i0, Int32 i1) -> DataViewGetterSetter<typename DataTypeTraitsT<X>::FunctionCall2ReturnType>
  {
    return DataViewGetterSetter<typename DataTypeTraitsT<X>::FunctionCall2ReturnType>(&m_ptr->operator()(i0, i1));
  }

 private:

  //! Adresse de la donnée. Valide uniquement pour les types simples (i.e pas les Real3)
  constexpr ARCCORE_HOST_DEVICE DataType* _address() const { return m_ptr; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
