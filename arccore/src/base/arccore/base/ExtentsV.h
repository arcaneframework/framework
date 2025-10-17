// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ExtentsV.h                                                  (C) 2000-2025 */
/*                                                                           */
/* Tag pour les tableaux N-dimensions.                                       */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_EXTENTSV_H
#define ARCCORE_BASE_EXTENTSV_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/BaseTypes.h"

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
} // namespace impl::extent

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
  static constexpr bool isDynamic1D() { return false; }

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
  static constexpr bool isDynamic1D() { return (nb_dynamic == 1); }

  using MDIndexType = MDIndex<1>;
  using ArrayExtentsValueType = impl::ArrayExtentsValue<IndexType_, X0>;
  using RemovedFirstExtentsType = ExtentsV<IndexType_>;
  using DynamicDimsType = MDIndex<nb_dynamic>;
  template <int X> using AddedFirstExtentsType = ExtentsV<IndexType_, X, X0>;
  template <int X, int Last> using AddedFirstLastExtentsType = ExtentsV<IndexType_, X, X0, Last>;
  template <int X, int Last1, int Last2> using AddedFirstLastLastExtentsType = ExtentsV<IndexType_, X, X0, Last1, Last2>;

  using IndexType ARCCORE_DEPRECATED_REASON("Use 'MDIndexType' instead") = ArrayIndex<1>;
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
  static constexpr bool isDynamic1D() { return false; }

  using MDIndexType = MDIndex<2>;
  using ArrayExtentsValueType = impl::ArrayExtentsValue<IndexType_, X0, X1>;
  using RemovedFirstExtentsType = ExtentsV<IndexType_, X1>;
  using DynamicDimsType = MDIndex<nb_dynamic>;
  template <int X> using AddedFirstExtentsType = ExtentsV<IndexType_, X, X0, X1>;
  template <int X, int Last> using AddedFirstLastExtentsType = ExtentsV<IndexType_, X, X0, X1, Last>;

  using IndexType ARCCORE_DEPRECATED_REASON("Use 'MDIndexType' instead") = ArrayIndex<2>;
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
  static constexpr bool isDynamic1D() { return false; }

  using MDIndexType = MDIndex<3>;
  using ArrayExtentsValueType = impl::ArrayExtentsValue<IndexType_, X0, X1, X2>;
  using RemovedFirstExtentsType = ExtentsV<IndexType_, X1, X2>;
  using DynamicDimsType = MDIndex<nb_dynamic>;
  template <int X> using AddedFirstExtentsType = ExtentsV<IndexType_, X, X0, X1, X2>;

  using IndexType ARCCORE_DEPRECATED_REASON("Use 'MDIndexType' instead") = MDIndex<3>;
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
  static constexpr bool isDynamic1D() { return false; }

  using MDIndexType = MDIndex<4>;
  using ArrayExtentsValueType = impl::ArrayExtentsValue<IndexType_, X0, X1, X2, X3>;
  using RemovedFirstExtentsType = ExtentsV<IndexType_, X1, X2, X3>;
  using DynamicDimsType = MDIndex<nb_dynamic>;

  using IndexType ARCCORE_DEPRECATED_REASON("Use 'MDIndexType' instead") = MDIndex<4>;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
