﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MemoryCopier.cc                                             (C) 2000-2024 */
/*                                                                           */
/* Fonctions diverses de copie mémoire.                                      */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/accelerator/AcceleratorGlobal.h"

#include "arcane/utils/Ref.h"
#include "arcane/utils/FixedArray.h"
#include "arcane/utils/NotSupportedException.h"
#include "arcane/utils/internal/SpecificMemoryCopyList.h"

#include "arcane/accelerator/core/RunQueue.h"
#include "arcane/accelerator/RunCommandLoop.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator::impl
{
using IndexedMemoryCopyArgs = Arcane::impl::IndexedMemoryCopyArgs;
using IndexedMultiMemoryCopyArgs = Arcane::impl::IndexedMultiMemoryCopyArgs;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename DataType, typename Extent>
class AcceleratorSpecificMemoryCopy
: public Arcane::impl::SpecificMemoryCopyBase<DataType, Extent>
{
  using BaseClass = Arcane::impl::SpecificMemoryCopyBase<DataType, Extent>;
  using BaseClass::_toTrueType;

 public:

  using BaseClass::m_extent;

 public:

  void copyFrom(const IndexedMemoryCopyArgs& args) override
  {
    _copyFrom(args.m_queue, args.m_indexes, _toTrueType(args.m_source), _toTrueType(args.m_destination));
  }

  void copyTo(const IndexedMemoryCopyArgs& args) override
  {
    _copyTo(args.m_queue, args.m_indexes, _toTrueType(args.m_source), _toTrueType(args.m_destination));
  }

  void fill(const IndexedMemoryCopyArgs& args) override
  {
    _fill(args.m_queue, args.m_indexes, _toTrueType(args.m_source), _toTrueType(args.m_destination));
  }

  void copyFrom(const IndexedMultiMemoryCopyArgs& args) override
  {
    _copyFrom(args.m_queue, args.m_indexes, args.m_multi_memory, _toTrueType(args.m_source_buffer));
  }

  void copyTo(const IndexedMultiMemoryCopyArgs& args) override
  {
    _copyTo(args.m_queue, args.m_indexes, args.m_const_multi_memory, _toTrueType(args.m_destination_buffer));
  }

  void fill(const IndexedMultiMemoryCopyArgs& args) override
  {
    _fill(args.m_queue, args.m_indexes, args.m_multi_memory, _toTrueType(args.m_source_buffer));
  }

 public:

  void _copyFrom(RunQueue* queue, SmallSpan<const Int32> indexes,
                 Span<const DataType> source, Span<DataType> destination)
  {
    ARCANE_CHECK_POINTER(queue);

    ARCANE_CHECK_ACCESSIBLE_POINTER(queue, indexes.data());
    ARCANE_CHECK_ACCESSIBLE_POINTER(queue, source.data());
    ARCANE_CHECK_ACCESSIBLE_POINTER(queue, destination.data());

    Int32 nb_index = indexes.size();
    const Int64 sub_size = m_extent.v;

    auto command = makeCommand(queue);
    command << RUNCOMMAND_LOOP1(iter, nb_index)
    {
      auto [i] = iter();
      Int64 zindex = i * sub_size;
      Int64 zci = indexes[i] * sub_size;
      for (Int32 z = 0; z < sub_size; ++z)
        destination[zindex + z] = source[zci + z];
    };
  }

  void _copyFrom(RunQueue* queue, SmallSpan<const Int32> indexes, SmallSpan<Span<std::byte>> multi_views,
                 Span<const DataType> source)
  {
    ARCANE_CHECK_POINTER(queue);

    if (arcaneIsCheck()) {
      ARCANE_CHECK_ACCESSIBLE_POINTER_ALWAYS(queue, indexes.data());
      ARCANE_CHECK_ACCESSIBLE_POINTER_ALWAYS(queue, source.data());
      ARCANE_CHECK_ACCESSIBLE_POINTER_ALWAYS(queue, multi_views.data());
      // Idéalement il faudrait tester les valeurs des éléments de multi_views
      // mais si on fait cela on peut potentiellement faire des transferts
      // entre l'accélérateur et le CPU.
    }
    const Int32 nb_index = indexes.size() / 2;
    // On devrait pouvoir utiliser 'm_extent.v' mais avec CUDA 12.1 cela génère
    // une erreur lors de l'exécution: error 98 : invalid device function
    const Int32 sub_size = m_extent.v;

    auto command = makeCommand(queue);
    command << RUNCOMMAND_LOOP1(iter, nb_index)
    {
      auto [i] = iter();
      Int32 index0 = indexes[i * 2];
      Int32 index1 = indexes[(i * 2) + 1];
      Span<std::byte> orig_view_bytes = multi_views[index0];
      auto* orig_view_data = reinterpret_cast<DataType*>(orig_view_bytes.data());
      // Utilise un span pour tester les débordements de tableau mais on
      // pourrait directement utiliser 'orig_view_data' pour plus de performances
      Span<DataType> orig_view = { orig_view_data, orig_view_bytes.size() / (Int64)sizeof(DataType) };
      Int64 zci = ((Int64)(index1)) * sub_size;
      Int64 z_index = (Int64)i * sub_size;
      for (Int32 z = 0, n = sub_size; z < n; ++z)
        orig_view[zci + z] = source[z_index + z];
    };
  }

  /*!
   * \brief Remplit les valeurs d'indices spécifiés par \a indexes.
   *
   * Si \a indexes est vide, remplit toutes les valeurs.
   */
  void _fill(RunQueue* queue, SmallSpan<const Int32> indexes, Span<const DataType> source,
             Span<DataType> destination)
  {
    ARCANE_CHECK_POINTER(queue);

    ARCANE_CHECK_ACCESSIBLE_POINTER(queue, indexes.data());
    ARCANE_CHECK_ACCESSIBLE_POINTER(queue, destination.data());
    ARCANE_CHECK_ACCESSIBLE_POINTER(eExecutionPolicy::Sequential, source.data());

    Int32 nb_index = indexes.size();
    const Int32 sub_size = m_extent.v;
    constexpr Int32 max_size = 24;

    // Pour l'instant on limite la taille de DataType en dur.
    // A terme, il faudrait allouer sur le device et désallouer en fin
    // d'exécution (via cudaMallocAsync/cudaFreeAsync pour gérer l'asynchronisme)
    if (sub_size > max_size)
      ARCANE_THROW(NotSupportedException, "sizeof(type) is too big (v={0} max={1})",
                   sizeof(DataType) * sub_size, sizeof(DataType) * max_size);
    FixedArray<DataType, max_size> local_source;
    for (Int32 z = 0; z < sub_size; ++z)
      local_source[z] = source[z];
    for (Int32 z = sub_size; z < max_size; ++z)
      local_source[z] = {};

    auto command = makeCommand(queue);
    // Si \a nb_index vaut 0, on remplit tous les éléments
    if (nb_index == 0) {
      Int32 nb_value = CheckedConvert::toInt32(destination.size() / sub_size);
      command << RUNCOMMAND_LOOP1(iter, nb_value)
      {
        auto [i] = iter();
        Int64 zci = i * sub_size;
        for (Int32 z = 0; z < sub_size; ++z)
          destination[zci + z] = local_source[z];
      };
    }
    else {
      command << RUNCOMMAND_LOOP1(iter, nb_index)
      {
        auto [i] = iter();
        Int64 zci = indexes[i] * sub_size;
        for (Int32 z = 0; z < sub_size; ++z)
          destination[zci + z] = local_source[z];
      };
    }
  }

  void _fill(RunQueue* queue, SmallSpan<const Int32> indexes, SmallSpan<Span<std::byte>> multi_views,
             Span<const DataType> source)
  {
    ARCANE_CHECK_POINTER(queue);

    if (arcaneIsCheck()) {
      ARCANE_CHECK_ACCESSIBLE_POINTER_ALWAYS(queue, indexes.data());
      ARCANE_CHECK_ACCESSIBLE_POINTER_ALWAYS(eExecutionPolicy::Sequential, source.data());
      ARCANE_CHECK_ACCESSIBLE_POINTER_ALWAYS(queue, multi_views.data());
      // Idéalement il faudrait tester les valeurs des éléments de multi_views
      // mais si on fait cela on peut potentiellement faire des transferts
      // entre l'accélérateur et le CPU.
    }
    const Int32 nb_index = indexes.size() / 2;
    // On devrait pouvoir utiliser 'm_extent.v' mais avec CUDA 12.1 cela génère
    // une erreur lors de l'exécution: error 98 : invalid device function
    const Int32 sub_size = m_extent.v;
    constexpr Int32 max_size = 24;

    // Pour l'instant on limite la taille de DataType en dur.
    // A terme, il faudrait allouer sur le device et désallouer en fin
    // d'exécution (via cudaMallocAsync/cudaFreeAsync pour gérer l'asynchronisme)
    if (sub_size > max_size)
      ARCANE_THROW(NotSupportedException, "sizeof(type) is too big (v={0} max={1})",
                   sizeof(DataType) * sub_size, sizeof(DataType) * max_size);
    FixedArray<DataType, max_size> local_source;
    for (Int32 z = 0; z < sub_size; ++z)
      local_source[z] = source[z];
    for (Int32 z = sub_size; z < max_size; ++z)
      local_source[z] = {};

    if (nb_index == 0) {
      // Remplit toutes les valeurs du tableau avec la source.
      // Comme le nombre d'éléments de la deuxième dimension dépend de la première,
      // on utilise un noyau par dimension.
      RunQueue::ScopedAsync sc(queue);
      const Int32 nb_dim1 = multi_views.size();
      for (Int32 zz = 0; zz < nb_dim1; ++zz) {
        Span<DataType> orig_view = Arccore::asSpan<DataType>(multi_views[zz]);
        Int32 nb_value = CheckedConvert::toInt32(orig_view.size() / sub_size);
        auto command = makeCommand(queue);
        command << RUNCOMMAND_LOOP1(iter, nb_value)
        {
          auto [i] = iter();
          orig_view[i] = local_source[i % sub_size];
        };
      }
    }
    else {
      auto command = makeCommand(queue);
      command << RUNCOMMAND_LOOP1(iter, nb_index)
      {
        auto [i] = iter();
        Int32 index0 = indexes[i * 2];
        Int32 index1 = indexes[(i * 2) + 1];
        Span<std::byte> orig_view_bytes = multi_views[index0];
        auto* orig_view_data = reinterpret_cast<DataType*>(orig_view_bytes.data());
        // Utilise un span pour tester les débordements de tableau mais on
        // pourrait directement utiliser 'orig_view_data' pour plus de performances
        Span<DataType> orig_view = { orig_view_data, orig_view_bytes.size() / (Int64)sizeof(DataType) };
        Int64 zci = ((Int64)(index1)) * sub_size;
        for (Int32 z = 0, n = sub_size; z < n; ++z)
          orig_view[zci + z] = local_source[z];
      };
    }
  }

  void _copyTo(RunQueue* queue, SmallSpan<const Int32> indexes, Span<const DataType> source,
               Span<DataType> destination)
  {
    ARCANE_CHECK_POINTER(queue);

    ARCANE_CHECK_ACCESSIBLE_POINTER(queue, indexes.data());
    ARCANE_CHECK_ACCESSIBLE_POINTER(queue, source.data());
    ARCANE_CHECK_ACCESSIBLE_POINTER(queue, destination.data());

    Int32 nb_index = indexes.size();
    const Int64 sub_size = m_extent.v;

    auto command = makeCommand(queue);
    command << RUNCOMMAND_LOOP1(iter, nb_index)
    {
      auto [i] = iter();
      Int64 zindex = i * sub_size;
      Int64 zci = indexes[i] * sub_size;
      for (Int32 z = 0; z < sub_size; ++z)
        destination[zci + z] = source[zindex + z];
    };
  }

  void _copyTo(RunQueue* queue, SmallSpan<const Int32> indexes, SmallSpan<const Span<const std::byte>> multi_views,
               Span<DataType> destination)
  {
    ARCANE_CHECK_POINTER(queue);

    if (arcaneIsCheck()) {
      ARCANE_CHECK_ACCESSIBLE_POINTER_ALWAYS(queue, indexes.data());
      ARCANE_CHECK_ACCESSIBLE_POINTER_ALWAYS(queue, destination.data());
      ARCANE_CHECK_ACCESSIBLE_POINTER_ALWAYS(queue, multi_views.data());
      // Idéalement il faudrait tester les valeurs des éléments de multi_views
      // mais si on fait cela on peut potentiellement faire des transferts
      // entre l'accélérateur et le CPU.
    }

    const Int32 nb_index = indexes.size() / 2;
    // On devrait pouvoir utiliser 'm_extent.v' mais avec CUDA 12.1 cela génère
    // une erreur lors de l'exécution: error 98 : invalid device function
    const Int32 sub_size = m_extent.v;

    auto command = makeCommand(queue);
    command << RUNCOMMAND_LOOP1(iter, nb_index)
    {
      auto [i] = iter();
      Int32 index0 = indexes[i * 2];
      Int32 index1 = indexes[(i * 2) + 1];
      Span<const std::byte> orig_view_bytes = multi_views[index0];
      auto* orig_view_data = reinterpret_cast<const DataType*>(orig_view_bytes.data());
      // Utilise un span pour tester les débordements de tableau mais on
      // pourrait directement utiliser 'orig_view_data' pour plus de performances
      Span<const DataType> orig_view = { orig_view_data, orig_view_bytes.size() / (Int64)sizeof(DataType) };
      Int64 zci = ((Int64)(index1)) * sub_size;
      Int64 z_index = (Int64)i * sub_size;
      for (Int32 z = 0, n = sub_size; z < n; ++z)
        destination[z_index + z] = orig_view[zci + z];
    };
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class AcceleratorIndexedCopyTraits
{
 public:

  using InterfaceType = Arcane::impl::ISpecificMemoryCopy;
  template <typename DataType, typename Extent> using SpecificType = AcceleratorSpecificMemoryCopy<DataType, Extent>;
  using RefType = Arcane::impl::SpecificMemoryCopyRef<AcceleratorIndexedCopyTraits>;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class AcceleratorSpecificMemoryCopyList
{
 public:

  AcceleratorSpecificMemoryCopyList()
  {
    Arcane::impl::ISpecificMemoryCopyList::setDefaultCopyListIfNotSet(&m_copy_list);
  }
  Arcane::impl::SpecificMemoryCopyList<AcceleratorIndexedCopyTraits> m_copy_list;
};

namespace
{
  AcceleratorSpecificMemoryCopyList global_copy_list;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
