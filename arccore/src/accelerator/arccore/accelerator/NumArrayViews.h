// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* NumArrayViews.h                                             (C) 2000-2025 */
/*                                                                           */
/* Gestion des vues pour les 'NumArray' pour les accélérateurs.              */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_ACCELERATOR_NUMARRAYVIEWS_H
#define ARCCORE_ACCELERATOR_NUMARRAYVIEWS_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/common/NumArray.h"
#include "arccore/common/DataView.h"
#include "arccore/common/accelerator/ViewBuildInfo.h"

#include "arccore/accelerator/AcceleratorGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename DataType, typename Extents, typename LayoutPolicy>
class NumArrayViewSetter;
template <typename Accessor, typename Extents, typename LayoutPolicy>
class NumArrayView;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe de base des vues sur les 'NumArray'.
 */
class ARCCORE_ACCELERATOR_EXPORT NumArrayViewBase
{
 protected:

  // Pour l'instant n'utilise pas encore \a command
  // mais il ne faut pas le supprimer
  explicit NumArrayViewBase(const ViewBuildInfo&,Span<const std::byte> bytes);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vue en lecture, écriture ou lecture/écriture sur un 'NumArray'.
 *
 * Les vues fonctionnent jusqu'à des tableaux de rang 4.
 */
template <typename Accessor, typename Extents, typename LayoutType>
class NumArrayView
: public NumArrayViewBase
{
 public:

  using DataType = typename Accessor::ValueType;
  using SpanType = MDSpan<DataType, Extents, LayoutType>;
  using AccessorReturnType = typename Accessor::AccessorReturnType;

 public:

  NumArrayView(const ViewBuildInfo& command, SpanType v)
  : NumArrayViewBase(command, Arccore::asBytes(v.to1DSpan()))
  , m_values(v)
  {}

  //! Accesseur pour un tableau de rang 1
  template <typename X = Extents, typename = std::enable_if_t<X::rank() == 1, void>>
  constexpr ARCCORE_HOST_DEVICE AccessorReturnType operator()(Int32 i) const
  {
    return Accessor::build(m_values.ptrAt(i));
  }
  //! Accesseur pour un tableau de rang 1
  template <typename X = Extents, typename = std::enable_if_t<X::rank() == 1, void>>
  constexpr ARCCORE_HOST_DEVICE AccessorReturnType operator()(ArrayIndex<1> idx) const
  {
    return Accessor::build(m_values.ptrAt(idx));
  }
  //! Accesseur pour un tableau de rang 1
  template <typename X = Extents, typename = std::enable_if_t<X::rank() == 1, void>>
  constexpr ARCCORE_HOST_DEVICE AccessorReturnType operator[](Int32 i) const
  {
    return Accessor::build(m_values.ptrAt(i));
  }
  //! Accesseur pour un tableau de rang 1
  template <typename X = Extents, typename = std::enable_if_t<X::rank() == 1, void>>
  constexpr ARCCORE_HOST_DEVICE AccessorReturnType operator[](ArrayIndex<1> idx) const
  {
    return Accessor::build(m_values.ptrAt(idx));
  }

  //! Accesseur pour un tableau de rang 2
  template <typename X = Extents, typename = std::enable_if_t<X::rank() == 2, void>>
  constexpr ARCCORE_HOST_DEVICE AccessorReturnType operator()(Int32 i, Int32 j) const
  {
    return Accessor::build(m_values.ptrAt(i, j));
  }
  //! Accesseur pour un tableau de rang 2
  template <typename X = Extents, typename = std::enable_if_t<X::rank() == 2, void>>
  constexpr ARCCORE_HOST_DEVICE AccessorReturnType operator()(ArrayIndex<2> idx) const
  {
    return Accessor::build(m_values.ptrAt(idx));
  }

  //! Accesseur pour un tableau de rang 3
  template <typename X = Extents, typename = std::enable_if_t<X::rank() == 3, void>>
  constexpr ARCCORE_HOST_DEVICE AccessorReturnType operator()(Int32 i, Int32 j, Int32 k) const
  {
    return Accessor::build(m_values.ptrAt(i, j, k));
  }
  //! Accesseur pour un tableau de rang 3
  template <typename X = Extents, typename = std::enable_if_t<X::rank() == 3, void>>
  constexpr ARCCORE_HOST_DEVICE AccessorReturnType operator()(ArrayIndex<3> idx) const
  {
    return Accessor::build(m_values.ptrAt(idx));
  }

  //! Accesseur pour un tableau de rang 4
  template <typename X = Extents, typename = std::enable_if_t<X::rank() == 4, void>>
  constexpr ARCCORE_HOST_DEVICE AccessorReturnType operator()(Int32 i, Int32 j, Int32 k, Int32 l) const
  {
    return Accessor::build(m_values.ptrAt(i, j, k, l));
  }
  //! Accesseur pour un tableau de rang 4
  template <typename X = Extents, typename = std::enable_if_t<X::rank() == 4, void>>
  constexpr ARCCORE_HOST_DEVICE AccessorReturnType operator()(ArrayIndex<4> idx) const
  {
    return Accessor::build(m_values.ptrAt(idx));
  }

  //! Converti en une vue 1D.
  constexpr ARCCORE_HOST_DEVICE Span<DataType> to1DSpan() const
  {
    return m_values.to1DSpan();
  }

 private:

  SpanType m_values;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vue en écriture.
 */
template <typename DataType, typename Extents, typename LayoutPolicy> auto
viewOut(const ViewBuildInfo& command, NumArray<DataType, Extents, LayoutPolicy>& var)
{
  using Accessor = DataViewSetter<DataType>;
  return NumArrayView<Accessor, Extents, LayoutPolicy>(command, var.mdspan());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Vue en lecture/écriture.
 */
template <typename DataType, typename Extents, typename LayoutPolicy> auto
viewInOut(const ViewBuildInfo& command, NumArray<DataType, Extents, LayoutPolicy>& v)
{
  using Accessor = DataViewGetterSetter<DataType>;
  return NumArrayView<Accessor, Extents, LayoutPolicy>(command, v.mdspan());
}

/*----------------------------------------------1-----------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vue en lecture.
 */
template <typename DataType, typename Extents, typename LayoutType> auto
viewIn(const ViewBuildInfo& command, const NumArray<DataType, Extents, LayoutType>& v)
{
  using Accessor = DataViewGetter<DataType>;
  return NumArrayView<Accessor, Extents, LayoutType>(command, v.constMDSpan());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Vue en entrée sur un NumArray
template <typename DataType, typename Extents, typename LayoutType = DefaultLayout>
using NumArrayInView = NumArrayView<DataViewGetter<DataType>, Extents, LayoutType>;

//! Vue en sortie sur un NumArray
template <typename DataType, typename Extents, typename LayoutType = DefaultLayout>
using NumArrayOutView = NumArrayView<DataViewSetter<DataType>, Extents, LayoutType>;

//! Vue en entrée/sortie sur un NumArray
template <typename DataType, typename Extents, typename LayoutType = DefaultLayout>
using NumArrayInOutView = NumArrayView<DataViewGetterSetter<DataType>, Extents, LayoutType>;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
