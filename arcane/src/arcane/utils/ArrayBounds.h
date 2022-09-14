// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ArrayBounds.h                                               (C) 2000-2022 */
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

namespace impl
{
template <class T> ARCCORE_HOST_DEVICE
constexpr T fastmod(T a , T b)
{
  return a < b ? a : a-b*(a/b);
}
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<A_MDRANK_TYPE(RankValue)>
class ArrayBoundsBase
: private ArrayExtents<RankValue>
{
 public:
  using ArrayExtents<RankValue>::extent;
 public:
  constexpr ArrayBoundsBase() : m_nb_element(0) {}
 public:
  ARCCORE_HOST_DEVICE constexpr Int64 nbElement() const { return m_nb_element; }
  ARCCORE_HOST_DEVICE constexpr std::array<Int32,A_MDRANK_RANK_VALUE(RankValue)> asStdArray() const { return ArrayExtents<RankValue>::asStdArray(); }
 protected:
  constexpr void _computeNbElement()
  {
    m_nb_element = this->totalNbElement();
  }
 protected:
  using ArrayExtents<RankValue>::m_extents;
  Int64 m_nb_element;
};

template<>
class ArrayBounds<MDDim1>
: public ArrayBoundsBase<MDDim1>
{
 public:
  using IndexType = ArrayBoundsIndex<1>;
  using ArrayBoundsBase<MDDim1>::m_extents;
  // Note: le constructeur ne doit pas être explicite pour permettre la conversion
  // à partir d'un entier.
  constexpr ArrayBounds(Int32 dim1)
  : ArrayBoundsBase<MDDim1>()
  {
    m_extents[0] = dim1;
    _computeNbElement();
  }
  ARCCORE_HOST_DEVICE constexpr IndexType getIndices(Int32 i) const
  {
    return { i };
  }
};

template<>
class ArrayBounds<MDDim2>
: public ArrayBoundsBase<MDDim2>
{
 public:
  using IndexType = ArrayBoundsIndex<2>;
  using ArrayBoundsBase<MDDim2>::m_extents;
  constexpr ArrayBounds(Int32 dim1,Int32 dim2)
  : ArrayBoundsBase<MDDim2>()
  {
    m_extents[0] = dim1;
    m_extents[1] = dim2;
    _computeNbElement();
  }
  ARCCORE_HOST_DEVICE constexpr IndexType getIndices(Int32 i) const
  {
    Int32 i1 = impl::fastmod(i,m_extents[1]);
    Int32 i0 = i / m_extents[1];
    return { i0, i1 };
  }
};

template<>
class ArrayBounds<MDDim3>
: public ArrayBoundsBase<MDDim3>
{
 public:
  using IndexType = ArrayBoundsIndex<3>;
  using ArrayBoundsBase<MDDim3>::m_extents;
  constexpr ArrayBounds(Int32 dim1,Int32 dim2,Int32 dim3)
  : ArrayBoundsBase<MDDim3>()
  {
    m_extents[0] = dim1;
    m_extents[1] = dim2;
    m_extents[2] = dim3;
    _computeNbElement();
  }
  ARCCORE_HOST_DEVICE constexpr IndexType getIndices(Int32 i) const
  {
    Int32 i2 = impl::fastmod(i,m_extents[2]);
    Int32 fac = m_extents[2];
    Int32 i1 = impl::fastmod(i / fac,m_extents[1]);
    fac *= m_extents[1];
    Int32 i0 = i / fac;
    return { i0, i1, i2 };
  }
};

template<>
class ArrayBounds<MDDim4>
: public ArrayBoundsBase<MDDim4>
{
 public:
  using IndexType = ArrayBoundsIndex<4>;
  using ArrayBoundsBase<MDDim4>::m_extents;
  constexpr ArrayBounds(Int32 dim1,Int32 dim2,Int32 dim3,Int32 dim4)
  : ArrayBoundsBase<MDDim4>()
  {
    m_extents[0] = dim1;
    m_extents[1] = dim2;
    m_extents[2] = dim3;
    m_extents[3] = dim4;
    _computeNbElement();
  }
  ARCCORE_HOST_DEVICE constexpr IndexType getIndices(Int32 i) const
  {
    // Compute base indices
    Int32 i3 = impl::fastmod(i,m_extents[3]);
    Int32 fac = m_extents[3];
    Int32 i2 = impl::fastmod(i/fac,m_extents[2]);
    fac *= m_extents[2];
    Int32 i1 = impl::fastmod(i/fac,m_extents[1]);
    fac *= m_extents[1];
    Int32 i0 = i /fac;
    return { i0, i1, i2, i3 };
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
