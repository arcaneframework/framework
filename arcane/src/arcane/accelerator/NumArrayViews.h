// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* NumArrayViews.h                                             (C) 2000-2021 */
/*                                                                           */
/* Gestion des vues pour les 'NumArray' pour les accélérateurs.              */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ACCELERATOR_NUMARRAYVIEWS_H
#define ARCANE_ACCELERATOR_NUMARRAYVIEWS_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/accelerator/NumArray.h"
#include "arcane/accelerator/ViewsCommon.h"

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

template<typename DataType,int N,typename LayoutType>
class NumArrayViewSetter;
template<typename Accessor,int N,typename LayoutType>
class NumArrayView;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe de base des vues sur les 'NumArray'.
 */
class NumArrayViewBase
{
 public:
  // Pour l'instant n'utilise pas encore \a command
  // mais il ne faut pas le supprimer
  explicit NumArrayViewBase(RunCommand&)
  {
  }
 private:
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vue en lecture, écriture ou lecture/écriture sur un 'NumArray' 1D.
 */
template<typename Accessor,typename LayoutType>
class NumArrayView<Accessor,1,LayoutType>
: public NumArrayViewBase
{
 public:

  using DataType = typename Accessor::ValueType;
  using SpanType = MDSpan<DataType,1,LayoutType>;
  using AccessorReturnType = typename Accessor::AccessorReturnType;

 public:

  NumArrayView(RunCommand& command,SpanType v)
  : NumArrayViewBase(command), m_values(v){}

  ARCCORE_HOST_DEVICE AccessorReturnType operator()(Int32 i) const
  {
    return Accessor::build(m_values.ptrAt(i));
  }
  ARCCORE_HOST_DEVICE AccessorReturnType operator()(ArrayBoundsIndex<1> idx) const
  {
    return Accessor::build(m_values.ptrAt(idx));
  }
  ARCCORE_HOST_DEVICE AccessorReturnType operator[](Int32 i) const
  {
    return Accessor::build(m_values.ptrAt(i));
  }
  ARCCORE_HOST_DEVICE AccessorReturnType operator[](ArrayBoundsIndex<1> idx) const
  {
    return Accessor::build(m_values.ptrAt(idx));
  }

 private:

  SpanType m_values;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vue en lecture, écriture ou lecture/écriture sur un 'NumArray' 2D.
 */
template<typename Accessor,typename LayoutType>
class NumArrayView<Accessor,2,LayoutType>
: public NumArrayViewBase
{
 public:

  using DataType = typename Accessor::ValueType;
  using SpanType = MDSpan<DataType,2,LayoutType>;
  using AccessorReturnType = typename Accessor::AccessorReturnType;

 public:

  NumArrayView(RunCommand& command,SpanType v)
  : NumArrayViewBase(command), m_values(v){}

  ARCCORE_HOST_DEVICE AccessorReturnType operator()(Int32 i,Int32 j) const
  {
    return Accessor::build(m_values.ptrAt(i,j));
  }
  ARCCORE_HOST_DEVICE AccessorReturnType operator()(ArrayBoundsIndex<2> idx) const
  {
    return Accessor::build(m_values.ptrAt(idx));
  }

 private:

  SpanType m_values;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vue en lecture, écriture ou lecture/écriture sur un 'NumArray' 3D.
 */
template<typename Accessor,typename LayoutType>
class NumArrayView<Accessor,3,LayoutType>
: public NumArrayViewBase
{
 public:

  using DataType = typename Accessor::ValueType;
  using SpanType = MDSpan<DataType,3,LayoutType>;
  using AccessorReturnType = typename Accessor::AccessorReturnType;

 public:

  NumArrayView(RunCommand& command,SpanType v)
  : NumArrayViewBase(command), m_values(v){}

  ARCCORE_HOST_DEVICE AccessorReturnType operator()(Int32 i,Int32 j,Int32 k) const
  {
    return Accessor::build(m_values.ptrAt(i,j,k));
  }
  ARCCORE_HOST_DEVICE AccessorReturnType operator()(ArrayBoundsIndex<3> idx) const
  {
    return Accessor::build(m_values.ptrAt(idx));
  }

 private:

  SpanType m_values;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vue en lecture, écriture ou lecture/écriture sur un 'NumArray' 4D.
 */
template<typename Accessor,typename LayoutType>
class NumArrayView<Accessor,4,LayoutType>
: public NumArrayViewBase
{
 public:

  using DataType = typename Accessor::ValueType;
  using SpanType = MDSpan<DataType,4,LayoutType>;
  using AccessorReturnType = typename Accessor::AccessorReturnType;

 public:

  NumArrayView(RunCommand& command,SpanType v)
  : NumArrayViewBase(command), m_values(v){}

  ARCCORE_HOST_DEVICE AccessorReturnType operator()(Int32 i,Int32 j,Int32 k,Int32 l) const
  {
    return Accessor::build(m_values.ptrAt(i,j,k,l));
  }
  ARCCORE_HOST_DEVICE AccessorReturnType operator()(ArrayBoundsIndex<4> idx) const
  {
    return Accessor::build(m_values.ptrAt(idx));
  }

 private:

  SpanType m_values;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vue en écriture.
 */
template<typename DataType,int N,typename LayoutType> auto
viewOut(RunCommand& command,NumArray<DataType,N,LayoutType>& var)
{
  using Accessor = DataViewSetter<DataType>;
  return NumArrayView<Accessor,N,LayoutType>(command,var.span());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Vue en lecture/écriture.
 */
template<typename DataType,int N,typename LayoutType> auto
viewInOut(RunCommand& command,NumArray<DataType,N,LayoutType>& v)
{
  using Accessor = DataViewGetterSetter<DataType>;
  return NumArrayView<Accessor,N,LayoutType>(command,v.span());
}

/*----------------------------------------------1-----------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vue en lecture.
 */
template<typename DataType,int N,typename LayoutType> auto
viewIn(RunCommand& command,const NumArray<DataType,N,LayoutType>& v)
{
  using Accessor = DataViewGetter<DataType>;
  return NumArrayView<Accessor,N,LayoutType>(command,v.constSpan());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
