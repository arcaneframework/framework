// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SpanViews.h                                                 (C) 2000-2023 */
/*                                                                           */
/* Gestion des vues pour les 'Span' pour les accélérateurs.                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ACCELERATOR_SPANVIEW_H
#define ARCANE_ACCELERATOR_SPANVIEW_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/NumArray.h"
#include "arcane/accelerator/ViewsCommon.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \file SpanViews.h
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
  explicit SpanViewBase(RunCommand&)
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

 public:

  SpanView(RunCommand& command, SpanType v)
  : SpanViewBase(command)
  , m_values(v)
  {}

  constexpr ARCCORE_HOST_DEVICE AccessorReturnType operator()(Int32 i) const
  {
    return Accessor::build(m_values.ptrAt(i));
  }

  constexpr ARCCORE_HOST_DEVICE AccessorReturnType operator[](Int32 i) const
  {
    return Accessor::build(m_values.ptrAt(i));
  }

 private:

  SpanType m_values;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vue en écriture.
 */
template <typename DataType> auto
viewOut(RunCommand& command, Span<DataType>& var)
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
viewOut(RunCommand& command, Array<DataType>& var)
{
  using Accessor = DataViewSetter<DataType>;
  return SpanView<Accessor>(command, var.span());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vue en lecture/écriture.
 */
template <typename DataType> auto
viewInOut(RunCommand& command, Span<DataType>& var)
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
viewInOut(RunCommand& command, Array<DataType>& var)
{
  using Accessor = DataViewGetterSetter<DataType>;
  return SpanView<Accessor>(command, var.span());
}

/*----------------------------------------------1-----------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vue en lecture.
 */
template <typename DataType> auto
viewIn(RunCommand& command, const Span<DataType>& var)
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
viewIn(RunCommand& command, const Array<DataType>& var)
{
  using Accessor = DataViewGetter<DataType>;
  return SpanView<Accessor>(command, var.span());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
