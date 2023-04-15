// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
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

#include <atomic>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename Traits>
class SpecificMemoryCopyList
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

    m_nb_specialized = 0;
    m_nb_generic = 0;
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

 public:

  RefType copier(Int32 v)
  {
    if (v < 0)
      ARCANE_FATAL("Bad value {0} for datasize", v);

    InterfaceType* x = nullptr;
    if (v >= 0 && v < NB_COPIER)
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
  std::atomic<Int32> m_nb_specialized;
  std::atomic<Int32> m_nb_generic;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_UTILS_EXPORT ISpecificMemoryCopy
{
 public:

  virtual ~ISpecificMemoryCopy() = default;

 public:

  virtual void copyFrom(Span<const Int32> indexes, Span<const std::byte> source,
                        Span<std::byte> destination) = 0;
  virtual void copyTo(Span<const Int32> indexes, Span<const std::byte> source,
                      Span<std::byte> destination) = 0;
  virtual Int32 datatypeSize() const = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename DataType, typename Extent>
class SpecificMemoryCopy
: public ISpecificMemoryCopy
{
  static Int32 typeSize() { return static_cast<Int32>(sizeof(DataType)); }

 public:

  void copyFrom(Span<const Int32> indexes, Span<const std::byte> source,
                Span<std::byte> destination) override
  {
    _copyFrom(indexes, _toTrueType(source), _toTrueType(destination));
  }

  void copyTo(Span<const Int32> indexes, Span<const std::byte> source,
              Span<std::byte> destination) override
  {
    _copyTo(indexes, _toTrueType(source), _toTrueType(destination));
  }

  Int32 datatypeSize() const override { return m_extent.v * typeSize(); }

 public:

  void _copyFrom(Span<const Int32> indexes, Span<const DataType> source,
                 Span<DataType> destination)
  {
    Int64 nb_index = indexes.size();
    for (Int32 i = 0; i < nb_index; ++i) {
      Int64 zindex = i * m_extent.v;
      Int64 zci = indexes[i] * m_extent.v;
      for (Int32 z = 0, n = m_extent.v; z < n; ++z)
        destination[zindex + z] = source[zci + z];
    }
  }

  void _copyTo(Span<const Int32> indexes, Span<const DataType> source,
               Span<DataType> destination)
  {
    Int64 nb_index = indexes.size();

    for (Int32 i = 0; i < nb_index; ++i) {
      Int64 zindex = i * m_extent.v;
      Int64 zci = indexes[i] * m_extent.v;
      for (Int32 z = 0, n = m_extent.v; z < n; ++z)
        destination[zci + z] = source[zindex + z];
    }
  }

 public:

  Extent m_extent;

 private:

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

  void copyFrom(Span<const Int32> indexes, Span<const std::byte> source,
                Span<std::byte> destination)
  {
    m_used_copier->copyFrom(indexes, source, destination);
  }
  void copyTo(Span<const Int32> indexes, Span<const std::byte> source,
              Span<std::byte> destination)
  {
    m_used_copier->copyTo(indexes, source, destination);
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
