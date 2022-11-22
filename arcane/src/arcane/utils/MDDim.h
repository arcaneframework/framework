// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MDDim.h                                                     (C) 2000-2022 */
/*                                                                           */
/* Tag pour les tableaux N-dimensions.                                       */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_MDDIM_H
#define ARCANE_UTILS_MDDIM_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/UtilsTypes.h"

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

template<Int32... RankSize> class ExtentsV;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<>
class ExtentsV<>
{
 public:

  using ArrayExtentsValueType = ArrayExtentsValue<>;

  static constexpr int rank() { return 0; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<Int32 X0>
class ExtentsV<X0>
{
 public:

  using IndexType = ArrayBoundsIndex<1>;
  using ArrayExtentsValueType = ArrayExtentsValue<X0>;
  using RemovedFirstExtentType = ExtentsV<>;

  static constexpr int rank() { return 1; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<Int32 X0,Int32 X1>
class ExtentsV<X0,X1>
{
 public:

  using IndexType = ArrayBoundsIndex<2>;
  using ArrayExtentsValueType = ArrayExtentsValue<X0,X1>;
  using RemovedFirstExtentType = ExtentsV<X1>;

  static constexpr int rank() { return 2; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<Int32 X0,Int32 X1,Int32 X2>
class ExtentsV<X0,X1,X2>
{
 public:

  using IndexType = ArrayBoundsIndex<3>;
  using ArrayExtentsValueType = ArrayExtentsValue<X0,X1,X2>;
  using RemovedFirstExtentType = ExtentsV<X1,X2>;

  static constexpr int rank() { return 3; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<Int32 X0,Int32 X1,Int32 X2,Int32 X3>
class ExtentsV<X0,X1,X2,X3>
{
 public:

  using IndexType = ArrayBoundsIndex<4>;
  using ArrayExtentsValueType = ArrayExtentsValue<X0,X1,X2,X3>;
  using RemovedFirstExtentType = ExtentsV<X1,X2,X3>;

  static constexpr int rank() { return 4; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Constante pour un tableau dynamique de rang 0
using MDDim0 = ExtentsV<>;

//! Constante pour un tableau dynamique de rang 1
using MDDim1 = ExtentsV<-1>;

//! Constante pour un tableau dynamique de rang 2
using MDDim2 = ExtentsV<-1,-1>;

//! Constante pour un tableau dynamique de rang 3
using MDDim3 = ExtentsV<-1,-1,-1>;

//! Constante pour un tableau dynamique de rang 4
using MDDim4 = ExtentsV<-1,-1,-1,-1>;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<int RankValue>
class MDDimType;

template<>
class MDDimType<0>
{
 public:
  using DimType = MDDim0;
};
template<>
class MDDimType<1>
{
 public:
  using DimType = MDDim1;
};
template<>
class MDDimType<2>
{
 public:
  using DimType = MDDim2;
};
template<>
class MDDimType<3>
{
 public:
  using DimType = MDDim3;
};
template<>
class MDDimType<4>
{
 public:
  using DimType = MDDim4;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Ces quatres macros pourront être supprimées après la 3.8

// A définir lorsqu'on voudra que le rang des classes NumArray et associées
// soit spécifier par une classe au lieu d'un entier
#define ARCANE_USE_TYPE_FOR_EXTENT
#define A_MDRANK_TYPE(rank_name) typename rank_name
#define A_MDRANK_RANK_VALUE(rank_name) (rank_name :: rank())

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
