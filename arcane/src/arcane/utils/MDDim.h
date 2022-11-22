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

//! Constante pour un tableau dynamique de rang 0
class MDDim0
{
 public:
  static constexpr int rank() { return 0; }
};

//! Constante pour un tableau dynamique de rang 1
class MDDim1
{
 public:
  using PreviousRankType = MDDim0;
  static constexpr int rank() { return 1; }
};

//! Constante pour un tableau dynamique de rang 2
class MDDim2
{
 public:
  using PreviousRankType = MDDim1;
  static constexpr int rank() { return 2; }
};

//! Constante pour un tableau dynamique de rang 3
class MDDim3
{
 public:
  using PreviousRankType = MDDim2;
  static constexpr int rank() { return 3; }
};

//! Constante pour un tableau dynamique de rang 4
class MDDim4
{
 public:
  using PreviousRankType = MDDim3;
  static constexpr int rank() { return 4; }
};

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
#define A_MDDIM(rank_value) MDDim< rank_value >

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
