// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MemoryView.cc                                               (C) 2000-2023 */
/*                                                                           */
/* Vues constantes ou modifiables sur une zone mémoire.                      */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/MemoryView.h"

#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/ArrayExtentsValue.h"

#include <cstring>

// TODO: ajouter statistiques sur les tailles de 'datatype' utilisées.
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ISpecificMemoryCopy
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

class SpecificMemoryCopyRef
{
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
  SpecificMemoryCopy<std::byte, ExtentValue<DynExtent>> m_generic_copier;
  ISpecificMemoryCopy* m_used_copier = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class SpecificMemoryCopyList
{
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

  SpecificMemoryCopyRef copier(Int32 v)
  {
    if (v < 0)
      ARCANE_FATAL("Bad value {0} for datasize", v);

    ISpecificMemoryCopy* x = nullptr;
    if (v >= 0 && v < NB_COPIER)
      x = m_copier[v];
    if (x) {
      if (x->datatypeSize() != v)
        ARCANE_FATAL("Incoherent datatype size v={0} expected={1}", x->datatypeSize(), v);
      ++m_nb_specialized;
    }
    else
      ++m_nb_generic;
    return SpecificMemoryCopyRef(x, v);
  }

 private:

  SpecificMemoryCopy<std::byte, ExtentValue<1>> m_s1;
  SpecificMemoryCopy<Int16, ExtentValue<1>> m_s2;
  SpecificMemoryCopy<std::byte, ExtentValue<3>> m_s3;
  SpecificMemoryCopy<Int32, ExtentValue<1>> m_s4;
  SpecificMemoryCopy<std::byte, ExtentValue<5>> m_s5;
  SpecificMemoryCopy<Int16, ExtentValue<3>> m_s6;
  SpecificMemoryCopy<std::byte, ExtentValue<7>> m_s7;
  SpecificMemoryCopy<Int64, ExtentValue<1>> m_s8;
  SpecificMemoryCopy<std::byte, ExtentValue<9>> m_s9;
  SpecificMemoryCopy<Int16, ExtentValue<5>> m_s10;
  SpecificMemoryCopy<Int32, ExtentValue<3>> m_s12;

  SpecificMemoryCopy<Int64, ExtentValue<2>> m_s16;
  SpecificMemoryCopy<Int64, ExtentValue<3>> m_s24;
  SpecificMemoryCopy<Int64, ExtentValue<4>> m_s32;
  SpecificMemoryCopy<Int64, ExtentValue<5>> m_s40;
  SpecificMemoryCopy<Int64, ExtentValue<6>> m_s48;
  SpecificMemoryCopy<Int64, ExtentValue<7>> m_s56;
  SpecificMemoryCopy<Int64, ExtentValue<8>> m_s64;
  SpecificMemoryCopy<Int64, ExtentValue<9>> m_s72;

  std::array<ISpecificMemoryCopy*, NB_COPIER> m_copier;
  std::atomic<Int32> m_nb_specialized;
  std::atomic<Int32> m_nb_generic;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::impl

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{
  impl::SpecificMemoryCopyList global_copy_list;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MutableMemoryView::
copyHost(ConstMemoryView v)
{
  auto source = v.bytes();
  auto destination = bytes();
  Int64 source_size = source.size();
  if (source_size==0)
    return;
  Int64 destination_size = destination.size();
  if (source_size>destination_size)
    ARCANE_FATAL("Destination is too small source_size={0} destination_size={1}",
                 source_size,destination_size);
  auto* dest_data = destination.data();
  auto* source_data = source.data();
  ARCANE_CHECK_POINTER(dest_data);
  ARCANE_CHECK_POINTER(source_data);
  std::memmove(destination.data(), source.data(), source_size);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MutableMemoryView::
copyFromIndexesHost(ConstMemoryView v, Span<const Int32> indexes)
{
  Int32 one_data_size = m_datatype_size;
  Int64 v_one_data_size = v.datatypeSize();
  if (one_data_size != v_one_data_size)
    ARCANE_FATAL("Datatype size are not equal this={0} v={1}",
                 one_data_size, v_one_data_size);

  Int64 nb_index = indexes.size();
  if (nb_index == 0)
    return;

  auto source = v.bytes();
  auto destination = bytes();

  impl::SpecificMemoryCopyRef copier = global_copy_list.copier(one_data_size);
  copier.copyFrom(indexes, source, destination);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ConstMemoryView::
copyToIndexesHost(MutableMemoryView v, Span<const Int32> indexes)
{
  Int32 one_data_size = m_datatype_size;
  Int64 v_one_data_size = v.datatypeSize();
  if (one_data_size != v_one_data_size)
    ARCANE_FATAL("Datatype size are not equal this={0} v={1}",
                 one_data_size, v_one_data_size);

  Int64 nb_index = indexes.size();
  if (nb_index == 0)
    return;

  auto source = bytes();
  auto destination = v.bytes();

  impl::SpecificMemoryCopyRef copier = global_copy_list.copier(one_data_size);
  copier.copyTo(indexes, source, destination);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MutableMemoryView
makeMutableMemoryView(void* ptr, Int32 datatype_size, Int64 nb_element)
{
  Span<std::byte> bytes(reinterpret_cast<std::byte*>(ptr), datatype_size * nb_element);
  return MutableMemoryView(bytes, datatype_size, nb_element);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ConstMemoryView
makeConstMemoryView(const void* ptr, Int32 datatype_size, Int64 nb_element)
{
  Span<const std::byte> bytes(reinterpret_cast<const std::byte*>(ptr), datatype_size * nb_element);
  return ConstMemoryView(bytes, datatype_size, nb_element);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_UTILS_EXPORT void
arcanePrintSpecificMemoryStats()
{
  if (arcaneIsCheck()) {
    // N'affiche que pour les tests
    //global_copy_list.printStats();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
