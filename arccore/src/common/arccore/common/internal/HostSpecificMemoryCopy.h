// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* HostSpecificMemoryCopy.h                                    (C) 2000-2026 */
/*                                                                           */
/* Classe template pour gérer des fonctions spécialisées de copie mémoire.   */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_COMMON_INTERNAL_HOSTSPECIFICMEMORYCOPYLIST_H
#define ARCCORE_COMMON_INTERNAL_HOSTSPECIFICMEMORYCOPYLIST_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/common/internal/SpecificMemoryCopyList.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Implémentation des copies et du remplissage sur hôte.
 */
template <typename DataType, typename Extent>
class HostSpecificMemoryCopy
: public SpecificMemoryCopyBase<DataType, Extent>
{
  using BaseClass = SpecificMemoryCopyBase<DataType, Extent>;
  using BaseClass::_toTrueType;

 public:

  using BaseClass::m_extent;

 public:

  void copyFrom(const IndexedMemoryCopyArgs& args) override
  {
    _copyFrom(args.m_indexes, _toTrueType(args.m_source), _toTrueType(args.m_destination));
  }
  void copyTo(const IndexedMemoryCopyArgs& args) override
  {
    _copyTo(args.m_indexes, _toTrueType(args.m_source), _toTrueType(args.m_destination));
  }
  void fill(const IndexedMemoryCopyArgs& args) override
  {
    _fill(args.m_indexes, _toTrueType(args.m_source), _toTrueType(args.m_destination));
  }
  void copyFrom(const IndexedMultiMemoryCopyArgs& args) override
  {
    _copyFrom(args.m_indexes, args.m_multi_memory, _toTrueType(args.m_source_buffer));
  }
  void copyTo(const IndexedMultiMemoryCopyArgs& args) override
  {
    _copyTo(args.m_indexes, args.m_const_multi_memory, _toTrueType(args.m_destination_buffer));
  }
  void fill(const IndexedMultiMemoryCopyArgs& args) override
  {
    _fill(args.m_indexes, args.m_multi_memory, _toTrueType(args.m_source_buffer));
  }

 public:

  void _copyFrom(SmallSpan<const Int32> indexes, Span<const DataType> source,
                 Span<DataType> destination)
  {
    ARCCORE_CHECK_POINTER(indexes.data());
    ARCCORE_CHECK_POINTER(source.data());
    ARCCORE_CHECK_POINTER(destination.data());

    Int32 nb_index = indexes.size();
    for (Int32 i = 0; i < nb_index; ++i) {
      Int64 z_index = (Int64)i * m_extent.v;
      Int64 zci = (Int64)(indexes[i]) * m_extent.v;
      for (Int32 z = 0, n = m_extent.v; z < n; ++z)
        destination[z_index + z] = source[zci + z];
    }
  }
  void _copyFrom(SmallSpan<const Int32> indexes, SmallSpan<Span<std::byte>> multi_views,
                 Span<const DataType> source)
  {
    ARCCORE_CHECK_POINTER(indexes.data());
    ARCCORE_CHECK_POINTER(source.data());
    ARCCORE_CHECK_POINTER(multi_views.data());

    const Int32 value_size = indexes.size() / 2;
    for (Int32 i = 0; i < value_size; ++i) {
      Int32 index0 = indexes[i * 2];
      Int32 index1 = indexes[(i * 2) + 1];
      Span<std::byte> orig_view_bytes = multi_views[index0];
      auto* orig_view_data = reinterpret_cast<DataType*>(orig_view_bytes.data());
      // Utilise un span pour tester les débordements de tableau mais on
      // pourrait directement utiliser 'orig_view_data' pour plus de performances
      Span<DataType> orig_view = { orig_view_data, orig_view_bytes.size() / (Int64)sizeof(DataType) };
      Int64 zci = ((Int64)(index1)) * m_extent.v;
      Int64 z_index = (Int64)i * m_extent.v;
      for (Int32 z = 0, n = m_extent.v; z < n; ++z)
        orig_view[zci + z] = source[z_index + z];
    }
  }

  /*!
   * \brief Remplit les valeurs d'indices spécifiés par \a indexes.
   *
   * Si \a indexes est vide, remplit toutes les valeurs.
   */
  void _fill(SmallSpan<const Int32> indexes, Span<const DataType> source,
             Span<DataType> destination)
  {
    ARCCORE_CHECK_POINTER(source.data());
    ARCCORE_CHECK_POINTER(destination.data());

    // Si \a indexes est vide, cela signifie qu'on copie toutes les valeurs
    Int32 nb_index = indexes.size();
    if (nb_index == 0) {
      Int64 nb_value = destination.size() / m_extent.v;
      for (Int64 i = 0; i < nb_value; ++i) {
        Int64 zci = i * m_extent.v;
        for (Int32 z = 0, n = m_extent.v; z < n; ++z)
          destination[zci + z] = source[z];
      }
    }
    else {
      ARCCORE_CHECK_POINTER(indexes.data());
      for (Int32 i = 0; i < nb_index; ++i) {
        Int64 zci = (Int64)(indexes[i]) * m_extent.v;
        for (Int32 z = 0, n = m_extent.v; z < n; ++z)
          destination[zci + z] = source[z];
      }
    }
  }

  void _fill(SmallSpan<const Int32> indexes, SmallSpan<Span<std::byte>> multi_views,
             Span<const DataType> source)
  {
    ARCCORE_CHECK_POINTER(source.data());
    ARCCORE_CHECK_POINTER(multi_views.data());

    const Int32 nb_index = indexes.size() / 2;
    if (nb_index == 0) {
      // Remplit toutes les valeurs du tableau avec la source.
      const Int32 nb_dim1 = multi_views.size();
      for (Int32 zz = 0; zz < nb_dim1; ++zz) {
        Span<std::byte> orig_view_bytes = multi_views[zz];
        Int64 nb_value = orig_view_bytes.size() / ((Int64)sizeof(DataType));
        auto* orig_view_data = reinterpret_cast<DataType*>(orig_view_bytes.data());
        Span<DataType> orig_view = { orig_view_data, nb_value };
        for (Int64 i = 0; i < nb_value; i += m_extent.v) {
          // Utilise un span pour tester les débordements de tableau mais on
          // pourrait directement utiliser 'orig_view_data' pour plus de performances
          for (Int32 z = 0, n = m_extent.v; z < n; ++z) {
            orig_view[i + z] = source[z];
          }
        }
      }
    }
    else {
      ARCCORE_CHECK_POINTER(indexes.data());
      for (Int32 i = 0; i < nb_index; ++i) {
        Int32 index0 = indexes[i * 2];
        Int32 index1 = indexes[(i * 2) + 1];
        Span<std::byte> orig_view_bytes = multi_views[index0];
        auto* orig_view_data = reinterpret_cast<DataType*>(orig_view_bytes.data());
        // Utilise un span pour tester les débordements de tableau mais on
        // pourrait directement utiliser 'orig_view_data' pour plus de performances
        Span<DataType> orig_view = { orig_view_data, orig_view_bytes.size() / (Int64)sizeof(DataType) };
        Int64 zci = ((Int64)(index1)) * m_extent.v;
        for (Int32 z = 0, n = m_extent.v; z < n; ++z)
          orig_view[zci + z] = source[z];
      }
    }
  }

  void _copyTo(SmallSpan<const Int32> indexes, Span<const DataType> source,
               Span<DataType> destination)
  {
    ARCCORE_CHECK_POINTER(indexes.data());
    ARCCORE_CHECK_POINTER(source.data());
    ARCCORE_CHECK_POINTER(destination.data());

    Int32 nb_index = indexes.size();

    for (Int32 i = 0; i < nb_index; ++i) {
      Int64 z_index = (Int64)i * m_extent.v;
      Int64 zci = (Int64)(indexes[i]) * m_extent.v;
      for (Int32 z = 0, n = m_extent.v; z < n; ++z)
        destination[zci + z] = source[z_index + z];
    }
  }

  void _copyTo(SmallSpan<const Int32> indexes, SmallSpan<const Span<const std::byte>> multi_views,
               Span<DataType> destination)
  {
    ARCCORE_CHECK_POINTER(indexes.data());
    ARCCORE_CHECK_POINTER(destination.data());
    ARCCORE_CHECK_POINTER(multi_views.data());

    const Int32 value_size = indexes.size() / 2;
    for (Int32 i = 0; i < value_size; ++i) {
      Int32 index0 = indexes[i * 2];
      Int32 index1 = indexes[(i * 2) + 1];
      Span<const std::byte> orig_view_bytes = multi_views[index0];
      auto* orig_view_data = reinterpret_cast<const DataType*>(orig_view_bytes.data());
      // Utilise un span pour tester les débordements de tableau mais on
      // pourrait directement utiliser 'orig_view_data' pour plus de performances
      Span<const DataType> orig_view = { orig_view_data, orig_view_bytes.size() / (Int64)sizeof(DataType) };
      Int64 zci = ((Int64)(index1)) * m_extent.v;
      Int64 z_index = (Int64)i * m_extent.v;
      for (Int32 z = 0, n = m_extent.v; z < n; ++z)
        destination[z_index + z] = orig_view[zci + z];
    }
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
