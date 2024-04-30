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
namespace impl::extent
{
  template <class... Int32> constexpr int doSum(Int32... x)
  {
    return (x + ...);
  }
  constexpr int oneIfDynamic(Int32 x)
  {
    return ((x == DynExtent) ? 1 : 0);
  }
  // Nombre de valeurs dynamiques dans la liste des arguments
  // Un argument est dynamique s'il vaut Arcane::DynExtent
  template <class... Int32> constexpr int nbDynamic(Int32... args)
  {
    return doSum(oneIfDynamic(args)...);
  }
} // namespace impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Spécialisation pour les dimensions des tableaux à 0 dimensions.
 */
template <typename IndexType_>
class ExtentsV<IndexType_>
{
 public:

  using ArrayExtentsValueType = impl::ArrayExtentsValue<IndexType_>;

  static constexpr int rank() { return 0; }
  static constexpr int nb_dynamic = 0;

  template <int X> using AddedFirstExtentsType = ExtentsV<IndexType_, X>;
  template <int X, int Last> using AddedFirstLastExtentsType = ExtentsV<IndexType_, X, Last>;
  template <int X, int Last1, int Last2> using AddedFirstLastLastExtentsType = ExtentsV<IndexType_, X, Last1, Last2>;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Spécialisation pour les dimensions des tableaux à 1 dimension.
 */
template <typename IndexType_, Int32 X0>
class ExtentsV<IndexType_, X0>
{
 public:

  static constexpr int rank() { return 1; }
  static constexpr int nb_dynamic = impl::extent::nbDynamic(X0);
  static constexpr bool is_full_dynamic() { return (nb_dynamic == 1); }

  using IndexType = ArrayIndex<1>;
  using ArrayExtentsValueType = impl::ArrayExtentsValue<IndexType_, X0>;
  using RemovedFirstExtentsType = ExtentsV<IndexType_>;
  using DynamicDimsType = ArrayIndex<nb_dynamic>;
  template <int X> using AddedFirstExtentsType = ExtentsV<IndexType_, X, X0>;
  template <int X, int Last> using AddedFirstLastExtentsType = ExtentsV<IndexType_, X, X0, Last>;
  template <int X, int Last1, int Last2> using AddedFirstLastLastExtentsType = ExtentsV<IndexType_, X, X0, Last1, Last2>;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Spécialisation pour les dimensions des tableaux à 2 dimensions.
 */
template <typename IndexType_, Int32 X0, Int32 X1>
class ExtentsV<IndexType_, X0, X1>
{
 public:

  static constexpr int rank() { return 2; }
  static constexpr int nb_dynamic = impl::extent::nbDynamic(X0, X1);
  static constexpr bool is_full_dynamic() { return (nb_dynamic == 2); }

  using IndexType = ArrayIndex<2>;
  using ArrayExtentsValueType = impl::ArrayExtentsValue<IndexType_, X0, X1>;
  using RemovedFirstExtentsType = ExtentsV<IndexType_, X1>;
  using DynamicDimsType = ArrayIndex<nb_dynamic>;
  template <int X> using AddedFirstExtentsType = ExtentsV<IndexType_, X, X0, X1>;
  template <int X, int Last> using AddedFirstLastExtentsType = ExtentsV<IndexType_, X, X0, X1, Last>;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Spécialisation pour les dimensions des tableaux à 3 dimensions.
 */
template <typename IndexType_, Int32 X0, Int32 X1, Int32 X2>
class ExtentsV<IndexType_, X0, X1, X2>
{
 public:

  static constexpr int rank() { return 3; }
  static constexpr int nb_dynamic = impl::extent::nbDynamic(X0, X1, X2);
  static constexpr bool is_full_dynamic() { return (nb_dynamic == 3); }

  using IndexType = ArrayIndex<3>;
  using ArrayExtentsValueType = impl::ArrayExtentsValue<IndexType_, X0, X1, X2>;
  using RemovedFirstExtentsType = ExtentsV<IndexType_, X1, X2>;
  using DynamicDimsType = ArrayIndex<nb_dynamic>;
  template <int X> using AddedFirstExtentsType = ExtentsV<IndexType_, X, X0, X1, X2>;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Spécialisation pour les dimensions des tableaux à 4 dimensions.
 */
template <typename IndexType_, Int32 X0, Int32 X1, Int32 X2, Int32 X3>
class ExtentsV<IndexType_, X0, X1, X2, X3>
{
 public:

  static constexpr int rank() { return 4; }
  static constexpr int nb_dynamic = impl::extent::nbDynamic(X0, X1, X2, X3);
  static constexpr bool is_full_dynamic() { return (nb_dynamic == 4); }

  using IndexType = ArrayIndex<4>;
  using ArrayExtentsValueType = impl::ArrayExtentsValue<IndexType_, X0, X1, X2, X3>;
  using RemovedFirstExtentsType = ExtentsV<IndexType_, X1, X2, X3>;
  using DynamicDimsType = ArrayIndex<nb_dynamic>;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Constante pour un tableau dynamique de rang 0
using MDDim0 = ExtentsV<Int32>;

//! Constante pour un tableau dynamique de rang 1
using MDDim1 = ExtentsV<Int32, DynExtent>;

//! Constante pour un tableau dynamique de rang 2
using MDDim2 = ExtentsV<Int32, DynExtent, DynExtent>;

//! Constante pour un tableau dynamique de rang 3
using MDDim3 = ExtentsV<Int32, DynExtent, DynExtent, DynExtent>;

//! Constante pour un tableau dynamique de rang 4
using MDDim4 = ExtentsV<Int32, DynExtent, DynExtent, DynExtent, DynExtent>;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

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

// Ces trois macros pourront être supprimées après la 3.8

// A définir lorsqu'on voudra que le rang des classes NumArray et associées
// soit spécifié par une classe au lieu d'un entier
#define ARCANE_USE_TYPE_FOR_EXTENT
#define A_MDRANK_TYPE(rank_name) typename rank_name
#define A_MDRANK_RANK_VALUE(rank_name) (rank_name :: rank())

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
