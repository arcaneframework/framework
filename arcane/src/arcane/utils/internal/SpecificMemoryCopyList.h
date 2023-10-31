﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SpecificMemoryCopyList.h                                    (C) 2000-2023 */
/*                                                                           */
/* Classe template pour gérer des fonctions spécialisées de copie mémoire.   */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_INTERNAL_SPECIFICMEMORYCOPYLIST_H
#define ARCANE_UTILS_INTERNAL_SPECIFICMEMORYCOPYLIST_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/String.h"
#include "arcane/utils/ArrayExtentsValue.h"
#include "arcane/utils/FatalErrorException.h"

#include <atomic>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Arguments pour une copie de certains indices entre deux zones mémoire.
 */
class ARCANE_UTILS_EXPORT IndexedMemoryCopyArgs
{
 public:

  using RunQueue = Arcane::Accelerator::RunQueue;

 public:

  IndexedMemoryCopyArgs(SmallSpan<const Int32> indexes, Span<const std::byte> source,
                        Span<std::byte> destination, RunQueue* run_queue)
  : m_indexes(indexes)
  , m_source(source)
  , m_destination(destination)
  , m_queue(run_queue)
  {}

 public:

  SmallSpan<const Int32> m_indexes;
  Span<const std::byte> m_source;
  Span<std::byte> m_destination;
  RunQueue* m_queue = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Arguments pour une copie de certains indices vers/depuis
 * une zone mémoire multiple.
 */
class ARCANE_UTILS_EXPORT IndexedMultiMemoryCopyArgs
{
 public:

  //! Constructeur pour copyTo
  IndexedMultiMemoryCopyArgs(SmallSpan<const Int32> indexes,
                             SmallSpan<const Span<const std::byte>> multi_memory,
                             Span<std::byte> destination,
                             RunQueue* run_queue)
  : m_indexes(indexes)
  , m_const_multi_memory(multi_memory)
  , m_destination_buffer(destination)
  , m_queue(run_queue)
  {}

  //! Constructor pour copyFrom
  IndexedMultiMemoryCopyArgs(SmallSpan<const Int32> indexes,
                             SmallSpan<Span<std::byte>> multi_memory,
                             Span<const std::byte> source,
                             RunQueue* run_queue)
  : m_indexes(indexes)
  , m_multi_memory(multi_memory)
  , m_source_buffer(source)
  , m_queue(run_queue)
  {}

 public:

  SmallSpan<const Int32> m_indexes;
  SmallSpan<const Span<const std::byte>> m_const_multi_memory;
  SmallSpan<Span<std::byte>> m_multi_memory;
  Span<const std::byte> m_source_buffer;
  Span<std::byte> m_destination_buffer;
  RunQueue* m_queue = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface d'un copieur mémoire spécialisé pour une taille de donnée.
 */
class ARCANE_UTILS_EXPORT ISpecificMemoryCopy
{
 public:

  virtual ~ISpecificMemoryCopy() = default;

 public:

  virtual void copyFrom(const IndexedMemoryCopyArgs& args) = 0;
  virtual void copyTo(const IndexedMemoryCopyArgs& args) = 0;
  virtual void fill(const IndexedMemoryCopyArgs& args) = 0;
  virtual void copyFrom(const IndexedMultiMemoryCopyArgs&) = 0;
  virtual void copyTo(const IndexedMultiMemoryCopyArgs&) = 0;
  virtual Int32 datatypeSize() const = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface d'une liste d'instances de ISpecificMemoryCopy spécialisées.
 */
class ARCANE_UTILS_EXPORT ISpecificMemoryCopyList
{
 public:

  /*!
   * \brief Positionne l'instance par défaut pour les copies.
   *
   * Cette méthode est normalement appelée par l'API accélérateur pour
   * fournir des noyaux de copie spécifiques à chaque device.
   */
  static void setDefaultCopyListIfNotSet(ISpecificMemoryCopyList* ptr);

 public:

  virtual void copyTo(Int32 datatype_size, const IndexedMemoryCopyArgs& args) = 0;
  virtual void copyFrom(Int32 datatype_size, const IndexedMemoryCopyArgs& args) = 0;
  virtual void fill(Int32 datatype_size, const IndexedMemoryCopyArgs& args) = 0;
  virtual void copyTo(Int32 datatype_size, const IndexedMultiMemoryCopyArgs& args) = 0;
  virtual void copyFrom(Int32 datatype_size, const IndexedMultiMemoryCopyArgs& args) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Liste d'instances de ISpecificMemoryCopy spécialisées.
 *
 * Cette classe contient des instances de ISpecificMemoryCopy spécialisées
 * pour une taille de de type de données. Cela permet au compilateur de
 * connaitre précisément la taille d'un type de donnée et ainsi de mieux
 * optimiser les boucles.
 */
template <typename Traits>
class SpecificMemoryCopyList
: public ISpecificMemoryCopyList
{
 public:

  using InterfaceType = typename Traits::InterfaceType;
  template <typename DataType, typename Extent> using SpecificType = typename Traits::template SpecificType<DataType, Extent>;
  using RefType = typename Traits::RefType;

 public:

  static constexpr Int32 NB_COPIER = 128;

 public:

  SpecificMemoryCopyList()
  {
    m_copier.fill(nullptr);

    m_copier[1] = &m_s1;
    m_copier[2] = &m_s2;
    m_copier[3] = &m_s3;
    m_copier[4] = &m_s4;
    m_copier[5] = &m_s5;
    m_copier[6] = &m_s6;
    m_copier[7] = &m_s7;
    m_copier[8] = &m_s8;
    m_copier[9] = &m_s9;
    m_copier[10] = &m_s10;

    m_copier[12] = &m_s12;
    m_copier[16] = &m_s16;
    m_copier[24] = &m_s24;
    m_copier[32] = &m_s32;
    m_copier[40] = &m_s40;
    m_copier[48] = &m_s48;
    m_copier[56] = &m_s56;
    m_copier[64] = &m_s64;
    m_copier[72] = &m_s72;
  }

 public:

  void printStats()
  {
    std::cout << "SpecificMemory::nb_specialized=" << m_nb_specialized
              << " nb_generic=" << m_nb_generic << "\n";
  }

  void checkValid()
  {
    // Vérifie que les taille sont correctes
    for (Int32 i = 0; i < NB_COPIER; ++i) {
      auto* x = m_copier[i];
      if (x && (x->datatypeSize() != i))
        ARCANE_FATAL("Incoherent datatype size v={0} expected={1}", x->datatypeSize(), i);
    }
  }

 private:

  RefType _copier(Int32 v)
  {
    if (v < 0)
      ARCANE_FATAL("Bad value {0} for datasize", v);

    InterfaceType* x = nullptr;
    if (v < NB_COPIER)
      x = m_copier[v];
    if (x) {
      if (x->datatypeSize() != v)
        ARCANE_FATAL("Incoherent datatype size v={0} expected={1}", x->datatypeSize(), v);
      ++m_nb_specialized;
    }
    else
      ++m_nb_generic;
    return RefType(x, v);
  }

 public:

  void copyTo(Int32 datatype_size, const IndexedMemoryCopyArgs& args) override
  {
    auto c = _copier(datatype_size);
    c.copyTo(args);
  }
  void copyTo(Int32 datatype_size, const IndexedMultiMemoryCopyArgs& args) override
  {
    auto c = _copier(datatype_size);
    c.copyTo(args);
  }
  void copyFrom(Int32 datatype_size, const IndexedMemoryCopyArgs& args) override
  {
    auto c = _copier(datatype_size);
    c.copyFrom(args);
  }
  void copyFrom(Int32 datatype_size, const IndexedMultiMemoryCopyArgs& args) override
  {
    auto c = _copier(datatype_size);
    c.copyFrom(args);
  }
  void fill(Int32 datatype_size, const IndexedMemoryCopyArgs& args) override
  {
    auto c = _copier(datatype_size);
    c.fill(args);
  }

 private:

  SpecificType<std::byte, ExtentValue<1>> m_s1;
  SpecificType<Int16, ExtentValue<1>> m_s2;
  SpecificType<std::byte, ExtentValue<3>> m_s3;
  SpecificType<Int32, ExtentValue<1>> m_s4;
  SpecificType<std::byte, ExtentValue<5>> m_s5;
  SpecificType<Int16, ExtentValue<3>> m_s6;
  SpecificType<std::byte, ExtentValue<7>> m_s7;
  SpecificType<Int64, ExtentValue<1>> m_s8;
  SpecificType<std::byte, ExtentValue<9>> m_s9;
  SpecificType<Int16, ExtentValue<5>> m_s10;
  SpecificType<Int32, ExtentValue<3>> m_s12;

  SpecificType<Int64, ExtentValue<2>> m_s16;
  SpecificType<Int64, ExtentValue<3>> m_s24;
  SpecificType<Int64, ExtentValue<4>> m_s32;
  SpecificType<Int64, ExtentValue<5>> m_s40;
  SpecificType<Int64, ExtentValue<6>> m_s48;
  SpecificType<Int64, ExtentValue<7>> m_s56;
  SpecificType<Int64, ExtentValue<8>> m_s64;
  SpecificType<Int64, ExtentValue<9>> m_s72;

  std::array<InterfaceType*, NB_COPIER> m_copier;
  std::atomic<Int32> m_nb_specialized = 0;
  std::atomic<Int32> m_nb_generic = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename DataType, typename Extent>
class SpecificMemoryCopyBase
: public ISpecificMemoryCopy
{
  static Int32 typeSize() { return static_cast<Int32>(sizeof(DataType)); }

 public:

  Int32 datatypeSize() const override { return m_extent.v * typeSize(); }

 public:

  Extent m_extent;

 protected:

  static Span<const DataType> _toTrueType(Span<const std::byte> a)
  {
    return { reinterpret_cast<const DataType*>(a.data()), a.size() / typeSize() };
  }
  static Span<DataType> _toTrueType(Span<std::byte> a)
  {
    return { reinterpret_cast<DataType*>(a.data()), a.size() / typeSize() };
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename DataType, typename Extent>
class SpecificMemoryCopy
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

 public:

  void _copyFrom(SmallSpan<const Int32> indexes, Span<const DataType> source,
                 Span<DataType> destination)
  {
    ARCANE_CHECK_POINTER(indexes.data());
    ARCANE_CHECK_POINTER(source.data());
    ARCANE_CHECK_POINTER(destination.data());

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
    ARCANE_CHECK_POINTER(indexes.data());
    ARCANE_CHECK_POINTER(source.data());
    ARCANE_CHECK_POINTER(multi_views.data());

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

  void _fill(SmallSpan<const Int32> indexes, Span<const DataType> source,
             Span<DataType> destination)
  {
    ARCANE_CHECK_POINTER(indexes.data());
    ARCANE_CHECK_POINTER(source.data());
    ARCANE_CHECK_POINTER(destination.data());

    Int32 nb_index = indexes.size();
    for (Int32 i = 0; i < nb_index; ++i) {
      Int64 zci = (Int64)(indexes[i]) * m_extent.v;
      for (Int32 z = 0, n = m_extent.v; z < n; ++z)
        destination[zci + z] = source[z];
    }
  }

  void _copyTo(SmallSpan<const Int32> indexes, Span<const DataType> source,
               Span<DataType> destination)
  {
    ARCANE_CHECK_POINTER(indexes.data());
    ARCANE_CHECK_POINTER(source.data());
    ARCANE_CHECK_POINTER(destination.data());

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
    ARCANE_CHECK_POINTER(indexes.data());
    ARCANE_CHECK_POINTER(destination.data());
    ARCANE_CHECK_POINTER(multi_views.data());

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

template <typename Traits>
class SpecificMemoryCopyRef
{
  template <typename DataType, typename Extent> using SpecificType = typename Traits::template SpecificType<DataType, Extent>;

 public:

  SpecificMemoryCopyRef(ISpecificMemoryCopy* specialized_copier, Int32 datatype_size)
  : m_specialized_copier(specialized_copier)
  , m_used_copier(specialized_copier)
  {
    m_generic_copier.m_extent.v = datatype_size;
    if (!m_used_copier)
      m_used_copier = &m_generic_copier;
  }

  void copyFrom(const IndexedMemoryCopyArgs& args)
  {
    m_used_copier->copyFrom(args);
  }

  void copyTo(const IndexedMemoryCopyArgs& args)
  {
    m_used_copier->copyTo(args);
  }

  void fill(const IndexedMemoryCopyArgs& args)
  {
    m_used_copier->fill(args);
  }

  void copyFrom(const IndexedMultiMemoryCopyArgs& args)
  {
    m_used_copier->copyFrom(args);
  }

  void copyTo(const IndexedMultiMemoryCopyArgs& args)
  {
    m_used_copier->copyTo(args);
  }

 private:

  ISpecificMemoryCopy* m_specialized_copier = nullptr;
  SpecificType<std::byte, ExtentValue<DynExtent>> m_generic_copier;
  ISpecificMemoryCopy* m_used_copier = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
