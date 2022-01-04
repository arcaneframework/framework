// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ViewsCommon.h                                               (C) 2000-2021 */
/*                                                                           */
/* Types de base pour la gestion des vues pour les accélérateurs.            */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ACCELERATOR_VIEWSCOMMON_H
#define ARCANE_ACCELERATOR_VIEWSCOMMON_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/accelerator/AcceleratorGlobal.h"

#include "arcane/datatype/DataTypeTraits.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \file ViewsCommon.h
 *
 * Ce fichier contient les déclarations des types pour gérer
 * les vues pour les accélérateurs.
 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType>
class DataViewSetter;
template<typename DataType>
class DataViewGetterSetter;


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe pour accéder à un élément d'une vue en lecture.
 */
template<typename DataType>
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
template<typename DataType>
class DataViewSetter
{
  // Pour accéder à m_ptr;
  friend class DataViewGetterSetter<DataType>;
 public:
  using ValueType = DataType;
  using AccessorReturnType = DataViewSetter<DataType>;
 public:
  explicit ARCCORE_HOST_DEVICE DataViewSetter(DataType* ptr)
  : m_ptr(ptr){}
  ARCCORE_HOST_DEVICE DataViewSetter(const DataViewSetter<DataType>& v)
  : m_ptr(v.m_ptr){}
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
  ARCCORE_HOST_DEVICE void operator+=(const DataType& v)
  {
    *m_ptr = (*m_ptr) + v;
  }
 public:

  template<typename X = DataType,typename = std::enable_if_t<DataTypeTraitsT<X>::HasComponentX()> >
  ARCCORE_HOST_DEVICE void setX(typename DataTypeTraitsT<X>::ComponentType value)
  {
    m_ptr->x = value;
  }
  template<typename X = DataType,typename = std::enable_if_t<DataTypeTraitsT<X>::HasComponentY()> >
  ARCCORE_HOST_DEVICE void setY(typename DataTypeTraitsT<X>::ComponentType value)
  {
    m_ptr->y = value;
  }
  template<typename X = DataType,typename = std::enable_if_t<DataTypeTraitsT<X>::HasComponentZ()> >
  ARCCORE_HOST_DEVICE void setZ(typename DataTypeTraitsT<X>::ComponentType value)
  {
    m_ptr->z = value;
  }

  template<typename X = DataType,typename = std::enable_if_t<DataTypeTraitsT<X>::HasComponentXX()> >
  ARCCORE_HOST_DEVICE void setXX(Real value)
  {
    m_ptr->x.x = value;
  }
  template<typename X = DataType,typename = std::enable_if_t<DataTypeTraitsT<X>::HasComponentYX()> >
  ARCCORE_HOST_DEVICE void setYX(Real value)
  {
    m_ptr->y.x = value;
  }
  template<typename X = DataType,typename = std::enable_if_t<DataTypeTraitsT<X>::HasComponentZX()> >
  ARCCORE_HOST_DEVICE void setZX(Real value)
  {
    m_ptr->z.x = value;
  }

  template<typename X = DataType,typename = std::enable_if_t<DataTypeTraitsT<X>::HasComponentXY()> >
  ARCCORE_HOST_DEVICE void setXY(Real value)
  {
    m_ptr->x.y = value;
  }
  template<typename X = DataType,typename = std::enable_if_t<DataTypeTraitsT<X>::HasComponentYY()> >
  ARCCORE_HOST_DEVICE void setYY(Real value)
  {
    m_ptr->y.y = value;
  }
  template<typename X = DataType,typename = std::enable_if_t<DataTypeTraitsT<X>::HasComponentZY()> >
  ARCCORE_HOST_DEVICE void setZY(Real value)
  {
    m_ptr->z.y = value;
  }

  template<typename X = DataType,typename = std::enable_if_t<DataTypeTraitsT<X>::HasComponentXZ()> >
  ARCCORE_HOST_DEVICE void setXZ(Real value)
  {
    m_ptr->x.z = value;
  }
  template<typename X = DataType,typename = std::enable_if_t<DataTypeTraitsT<X>::HasComponentYZ()> >
  ARCCORE_HOST_DEVICE void setYZ(Real value)
  {
    m_ptr->y.z = value;
  }
  template<typename X = DataType,typename = std::enable_if_t<DataTypeTraitsT<X>::HasComponentZZ()> >
  ARCCORE_HOST_DEVICE void setZZ(Real value)
  {
    m_ptr->z.z = value;
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
template<typename DataType>
class DataViewGetterSetter
: public DataViewSetter<DataType>
{
  using BaseType = DataViewSetter<DataType>;
  using BaseType::m_ptr;
 public:
  using ValueType = DataType;
  using AccessorReturnType = DataViewGetterSetter<DataType>;
 public:
  explicit ARCCORE_HOST_DEVICE DataViewGetterSetter(DataType* ptr) : BaseType(ptr){}
  ARCCORE_HOST_DEVICE DataViewGetterSetter(const DataViewGetterSetter& v) : BaseType(v){}
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
};


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
