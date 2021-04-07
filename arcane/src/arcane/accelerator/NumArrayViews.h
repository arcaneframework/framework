// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* NumArrayViews.h                                             (C) 2000-2021 */
/*                                                                           */
/* Gestion des vues pour les 'NumArray' pour les accélérateurs.              */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ACCELERATOR_NUMARRAYVIEW_H
#define ARCANE_ACCELERATOR_NUMARRAYVIEW_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/accelerator/NumArray.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \file NumArrayViews.h
 *
 * Ce fichier contient les déclarations des types pour gérer
 * les vues pour les accélérateurs de la classe 'NumArray'.
 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType,int N>
class NumArrayViewSetter;
template<typename Accessor,int N>
class NumArrayView;
template<typename DataType>
class DataViewSetter;
template<typename DataType>
class DataViewGetterSetter;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe de base des vues sur les 'NumArray'.
 */
class NumArrayViewBase
{
 public:
  explicit NumArrayViewBase(RunCommand& command)
  : m_run_command(&command)
  {
  }
 private:
  RunCommand* m_run_command = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe pour accéder à un élément d'une vue en lecture.
 * TODO: fusionner avec les vues sur les variables
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
 * TODO: fusionner avec les vues sur les variables
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
/*!
 * \brief Vue en lecture, écriture ou lecture/écriture sur un 'NumArray' 1D.
 */
template<typename Accessor>
class NumArrayView<Accessor,1>
: public NumArrayViewBase
{
 public:

  using DataType = typename Accessor::ValueType;
  using SpanType = MDSpan<DataType,1>;
  using AccessorReturnType = typename Accessor::AccessorReturnType;

 public:

  NumArrayView(RunCommand& command,SpanType v)
  : NumArrayViewBase(command), m_values(v){}

  ARCCORE_HOST_DEVICE AccessorReturnType operator()(Int64 i) const
  {
    return Accessor::build(m_values.ptrAt(i));
  }

 private:

  SpanType m_values;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vue en lecture, écriture ou lecture/écriture sur un 'NumArray' 2D.
 */
template<typename Accessor>
class NumArrayView<Accessor,2>
: public NumArrayViewBase
{
 public:

  using DataType = typename Accessor::ValueType;
  using SpanType = MDSpan<DataType,2>;
  using AccessorReturnType = typename Accessor::AccessorReturnType;

 public:

  NumArrayView(RunCommand& command,SpanType v)
  : NumArrayViewBase(command), m_values(v){}

  ARCCORE_HOST_DEVICE AccessorReturnType operator()(Int64 i,Int64 j) const
  {
    return Accessor::build(m_values.ptrAt(i,j));
  }

 private:

  SpanType m_values;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vue en lecture, écriture ou lecture/écriture sur un 'NumArray' 2D.
 */
template<typename Accessor>
class NumArrayView<Accessor,3>
: public NumArrayViewBase
{
 public:

  using DataType = typename Accessor::ValueType;
  using SpanType = MDSpan<DataType,3>;
  using AccessorReturnType = typename Accessor::AccessorReturnType;

 public:

  NumArrayView(RunCommand& command,SpanType v)
  : NumArrayViewBase(command), m_values(v){}

  ARCCORE_HOST_DEVICE AccessorReturnType operator()(Int64 i,Int64 j,Int64 k) const
  {
    return Accessor::build(m_values.ptrAt(i,j,k));
  }

 private:

  SpanType m_values;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vue en écriture.
 */
template<typename DataType,int N> auto
viewOut(RunCommand& command,NumArray<DataType,N>& var)
{
  using Accessor = DataViewSetter<DataType>;
  return NumArrayView<Accessor,N>(command,var.span());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Vue en lecture/écriture.
 */
template<typename DataType,int N> auto
viewInOut(RunCommand& command,NumArray<DataType,N>& v)
{
  using Accessor = DataViewGetterSetter<DataType>;
  return NumArrayView<Accessor,N>(command,v.span());
}

/*----------------------------------------------1-----------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vue en lecture.
 */
template<typename DataType,int N> auto
viewIn(RunCommand& command,const NumArray<DataType,N>& v)
{
  using Accessor = DataViewGetter<DataType>;
  return NumArrayView<Accessor,N>(command,v.constSpan());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
