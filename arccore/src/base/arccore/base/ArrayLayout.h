// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ArrayLayout.h                                               (C) 2000-2025 */
/*                                                                           */
/* Gestion de la disposition mémoire pour les tableaux N-dimensions.         */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_ARRAYLAYOUT_H
#define ARCCORE_BASE_ARRAYLAYOUT_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/MDIndex.h"
#include "arccore/base/ArrayExtentsValue.h"
#include "arccore/base/MDDim.h"

#include <array>

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

template<Int32 I,Int32 J>
class ArrayLayout2
{
 public:

  static constexpr Int64 LastExtent = J;

  static ARCCORE_HOST_DEVICE constexpr Int64
  offset(ArrayIndex<2> idx,Int64 extent1)
  {
    return (extent1 * idx[I]) + Int64(idx[J]);
  }

  static constexpr std::array<Int32,2> layoutInfo() { return { I, J }; }
  static constexpr ARCCORE_HOST_DEVICE Int32 layout0() { return I; }
  static constexpr ARCCORE_HOST_DEVICE Int32 layout1() { return J; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<Int32 I,Int32 J,Int32 K>
class ArrayLayout3
{
 public:

  static constexpr Int64 LastExtent = K;

  static ARCCORE_HOST_DEVICE constexpr Int64
  offset(ArrayIndex<3> idx,Int64 extent1,Int64 extent2)
  {
    return (extent2 * idx[I]) + (extent1*idx[J]) + idx.asInt64(K);
  }

  template<typename ExtentType> static ARCCORE_HOST_DEVICE constexpr Int64
  computeOffsetIndexes(const ExtentType& extents)
  {
    return extents.template constLargeExtent<J>() * extents.template constLargeExtent<K>();
  }

  static constexpr std::array<Int32,3> layoutInfo() { return { I, J, K }; }

  static constexpr ARCCORE_HOST_DEVICE Int32 layout0() { return I; }
  static constexpr ARCCORE_HOST_DEVICE Int32 layout1() { return J; }
  static constexpr ARCCORE_HOST_DEVICE Int32 layout2() { return K; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
// Layout par défaut pour chaque dimension

template<int N> class RightLayoutN;
template<int N> class LeftLayoutN;

class RightLayout
{
 public:
  //! Implémentation pour le rang N
  template <int Rank> using LayoutType = RightLayoutN<Rank>;
  using Layout1Type = LayoutType<1>;
  using Layout2Type = LayoutType<2>;
  using Layout3Type = LayoutType<3>;
  using Layout4Type = LayoutType<4>;
};

class LeftLayout
{
 public:
  template <int Rank> using LayoutType = LeftLayoutN<Rank>;
  using Layout1Type = LayoutType<1>;
  using Layout2Type = LayoutType<2>;
  using Layout3Type = LayoutType<3>;
  using Layout4Type = LayoutType<4>;
};

template<> class RightLayoutN<2> : public ArrayLayout2<0,1> {};
template<> class RightLayoutN<3> : public ArrayLayout3<0,1,2> {};

template<> class LeftLayoutN<2> : public ArrayLayout2<1,0> {};
template<> class LeftLayoutN<3> : public ArrayLayout3<2,1,0> {};

// Les 4 using suivants sont pour compatibilité. A supprimer dans la 3.9
using LeftLayout2 = LeftLayout;
using LeftLayout3 = LeftLayout;
using RightLayout2 = RightLayout;
using RightLayout3 = RightLayout;

//! Le layout par défaut est toujours RightLayout
class DefaultLayout : public RightLayout {};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
