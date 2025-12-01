// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SpanViews.h                                                 (C) 2000-2025 */
/*                                                                           */
/* Gestion des vues pour les 'Span' pour les accélérateurs.                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_ACCELERATOR_SPANVIEW_H
#define ARCCORE_ACCELERATOR_SPANVIEW_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/common/NumArray.h"
#include "arccore/common/DataView.h"
#include "arccore/common/accelerator/ViewBuildInfo.h"

#include "arccore/accelerator/AcceleratorGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \file SpanViews.h
 *
 * Ce fichier contient les déclarations des types pour gérer
 * les vues pour les accélérateurs des classes Array, Span, SmallSpan,
 * ArrayView et ConstArrayView.
 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename DataType, typename Extents, typename LayoutPolicy>
class SpanViewSetter;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe de base des vues sur les 'NumArray'.
 */
class SpanViewBase
{
 protected:

  // Pour l'instant n'utilise pas encore \a command
  // mais il ne faut pas le supprimer
  explicit SpanViewBase(const ViewBuildInfo&)
  {
  }

 private:
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vue en lecture, écriture ou lecture/écriture sur un 'Span'.
 */
template <typename Accessor>
class SpanView
: public SpanViewBase
{
 public:

  using DataType = typename Accessor::ValueType;
  using AccessorReturnType = typename Accessor::AccessorReturnType;
  using SpanType = Span<DataType>;
  using size_type = typename SpanType::size_type;

 public:

  SpanView(const ViewBuildInfo& command, SpanType v)
  : SpanViewBase(command)
  , m_values(v)
  {}

  constexpr ARCCORE_HOST_DEVICE AccessorReturnType operator()(size_type i) const
  {
    return Accessor::build(m_values.ptrAt(i));
  }

  constexpr ARCCORE_HOST_DEVICE AccessorReturnType operator[](size_type i) const
  {
    return Accessor::build(m_values.ptrAt(i));
  }
  constexpr ARCCORE_HOST_DEVICE size_type size() const { return m_values.size(); }

 private:

  SpanType m_values;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vue en lecture, écriture ou lecture/écriture sur un 'SmallSpan'.
 */
template <typename Accessor>
class SmallSpanView
: public SpanViewBase
{
 public:

  using DataType = typename Accessor::ValueType;
  using AccessorReturnType = typename Accessor::AccessorReturnType;
  using SpanType = SmallSpan<DataType>;
  using size_type = typename SpanType::size_type;

 public:

  SmallSpanView(const ViewBuildInfo& command, SpanType v)
  : SpanViewBase(command)
  , m_values(v)
  {}

  constexpr ARCCORE_HOST_DEVICE AccessorReturnType operator()(size_type i) const
  {
    return Accessor::build(m_values.ptrAt(i));
  }

  constexpr ARCCORE_HOST_DEVICE AccessorReturnType operator[](size_type i) const
  {
    return Accessor::build(m_values.ptrAt(i));
  }
  constexpr ARCCORE_HOST_DEVICE size_type size() const { return m_values.size(); }

 private:

  SpanType m_values;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vue en écriture.
 */
template <typename DataType> auto
viewOut(const ViewBuildInfo& command, Span<DataType> var)
{
  using Accessor = DataViewSetter<DataType>;
  return SpanView<Accessor>(command, var);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vue en écriture.
 */
template <typename DataType> auto
viewOut(const ViewBuildInfo& command, Array<DataType>& var)
{
  return viewOut(command, var.span());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vue en écriture.
 */
template <typename DataType> auto
viewOut(const ViewBuildInfo& command, SmallSpan<DataType> var)
{
  using Accessor = DataViewSetter<DataType>;
  return SmallSpanView<Accessor>(command, var);
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vue en écriture.
 */
template <typename DataType> auto
viewOut(const ViewBuildInfo& command, ArrayView<DataType> var)
{
  return viewOut(command, SmallSpan<DataType>(var));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vue en lecture/écriture.
 */
template <typename DataType> auto
viewInOut(const ViewBuildInfo& command, Span<DataType> var)
{
  using Accessor = DataViewGetterSetter<DataType>;
  return SpanView<Accessor>(command, var);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vue en lecture/écriture.
 */
template <typename DataType> auto
viewInOut(const ViewBuildInfo& command, Array<DataType>& var)
{
  return viewInOut(command, var.span());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vue en lecture/écriture.
 */
template <typename DataType> auto
viewInOut(const ViewBuildInfo& command, SmallSpan<DataType> var)
{
  using Accessor = DataViewGetterSetter<DataType>;
  return SmallSpanView<Accessor>(command, var);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vue en lecture/écriture.
 */
template <typename DataType> auto
viewInOut(const ViewBuildInfo& command, ArrayView<DataType> var)
{
  return viewInOut(command, SmallSpan<DataType>(var));
}

/*----------------------------------------------1-----------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vue en lecture.
 */
template <typename DataType> auto
viewIn(const ViewBuildInfo& command, Span<DataType> var)
{
  using Accessor = DataViewGetter<DataType>;
  return SpanView<Accessor>(command, var);
}

/*----------------------------------------------1-----------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vue en lecture.
 */
template <typename DataType> auto
viewIn(const ViewBuildInfo& command, const Array<DataType>& var)
{
  return viewIn(command, var.span());
}

/*----------------------------------------------1-----------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vue en lecture.
 */
template <typename DataType> auto
viewIn(const ViewBuildInfo& command, SmallSpan<DataType> var)
{
  using Accessor = DataViewGetter<DataType>;
  return SmallSpanView<Accessor>(command, var);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! \brief Vue en lecture
template <typename DataType> auto
viewIn(const ViewBuildInfo& command, ConstArrayView<DataType> var)
{
  return viewIn(command, SmallSpan<const DataType>(var));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
