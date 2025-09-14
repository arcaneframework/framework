// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ArrayBounds.h                                               (C) 2000-2025 */
/*                                                                           */
/* Gestion des itérations sur les tableaux N-dimensions                      */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_ARRAYBOUNDS_H
#define ARCANE_UTILS_ARRAYBOUNDS_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArrayExtents.h"

/*
 * ATTENTION:
 *
 * Toutes les classes de ce fichier sont expérimentales et l'API n'est pas
 * figée. A NE PAS UTILISER EN DEHORS DE ARCANE.
 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename Extents>
class ArrayBoundsBase
: private ArrayExtents<Extents>
{
 public:

  using BaseClass = ArrayExtents<Extents>;
  using BaseClass::asStdArray;
  using BaseClass::constExtent;
  using BaseClass::getIndices;
  using IndexType ARCANE_DEPRECATED_REASON("Use 'MDIndexType' instead") = typename BaseClass::MDIndexType;
  using MDIndexType = typename BaseClass::MDIndexType;
  using ArrayExtentType = Arcane::ArrayExtents<Extents>;

 public:

  ArrayBoundsBase() = default;
  constexpr explicit ArrayBoundsBase(const BaseClass& rhs)
  : ArrayExtents<Extents>(rhs)
  {
  }

  constexpr explicit ArrayBoundsBase(const std::array<Int32, Extents::nb_dynamic>& v)
  : BaseClass(v)
  {
  }

 public:

  constexpr ARCCORE_HOST_DEVICE Int64 nbElement() const { return this->totalNbElement(); }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename Extents>
class ArrayBounds
: public ArrayBoundsBase<Extents>
{
 public:

  using ExtentsType = Extents;
  using BaseClass = ArrayBoundsBase<ExtentsType>;
  using ArrayExtentsType = ArrayExtents<ExtentsType>;

 public:

  template <typename X = Extents, typename = std::enable_if_t<X::nb_dynamic == 4, void>>
  constexpr ArrayBounds(Int32 dim1, Int32 dim2, Int32 dim3, Int32 dim4)
  : BaseClass(ArrayExtentsType(dim1, dim2, dim3, dim4))
  {
  }

  template <typename X = Extents, typename = std::enable_if_t<X::nb_dynamic == 3, void>>
  constexpr ArrayBounds(Int32 dim1, Int32 dim2, Int32 dim3)
  : BaseClass(ArrayExtentsType(dim1, dim2, dim3))
  {
  }

  template <typename X = Extents, typename = std::enable_if_t<X::nb_dynamic == 2, void>>
  constexpr ArrayBounds(Int32 dim1, Int32 dim2)
  : BaseClass(ArrayExtentsType(dim1, dim2))
  {
  }

  template <typename X = Extents, typename = std::enable_if_t<X::nb_dynamic == 1, void>>
  constexpr ArrayBounds(Int32 dim1)
  : BaseClass(ArrayExtentsType(dim1))
  {
  }

  constexpr explicit ArrayBounds(const ArrayExtentsType& v)
  : BaseClass(v)
  {
  }

  constexpr explicit ArrayBounds(std::array<Int32, Extents::nb_dynamic>& v)
  : BaseClass(v)
  {
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
