// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MemoryView.cc                                               (C) 2000-2023 */
/*                                                                           */
/* Vues constantes ou modifiables sur une zone mémoire.                       */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/MemoryView.h"

#include "arcane/utils/FatalErrorException.h"

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

template <typename DataType, Int32 N>
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

  Int32 datatypeSize() const override { return N * typeSize(); }

 public:

  void _copyFrom(Span<const Int32> indexes, Span<const DataType> source,
                 Span<DataType> destination)
  {
    Int64 nb_index = indexes.size();
    for (Int32 i = 0; i < nb_index; ++i) {
      Int64 zindex = i * N;
      Int64 zci = indexes[i] * N;
      for (Int32 z = 0; z < N; ++z)
        destination[zindex + z] = source[zci + z];
    }
  }

  void _copyTo(Span<const Int32> indexes, Span<const DataType> source,
               Span<DataType> destination)
  {
    Int64 nb_index = indexes.size();

    for (Int32 i = 0; i < nb_index; ++i) {
      Int64 zindex = i * N;
      Int64 zci = indexes[i] * N;
      for (Int32 z = 0; z < N; ++z)
        destination[zci + z] = source[zindex + z];
    }
  }

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
  }

 public:

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

  ISpecificMemoryCopy* copier(Int64 v)
  {
    if (v <= 0)
      return nullptr;
    if (v >= NB_COPIER)
      return nullptr;
    auto* x = m_copier[v];
#ifdef ARCANE_CHECK
    if (x) {
      if (x->datatypeSize() != v)
        ARCANE_FATAL("Incoherent datatype size v={0} expected={1}", x->datatypeSize(), v);
    }
#endif
    return x;
  }

 private:

  SpecificMemoryCopy<std::byte, 1> m_s1;
  SpecificMemoryCopy<Int16, 1> m_s2;
  SpecificMemoryCopy<std::byte, 3> m_s3;
  SpecificMemoryCopy<Int32, 1> m_s4;
  SpecificMemoryCopy<std::byte, 5> m_s5;
  SpecificMemoryCopy<Int16, 3> m_s6;
  SpecificMemoryCopy<std::byte, 7> m_s7;
  SpecificMemoryCopy<Int64, 1> m_s8;
  SpecificMemoryCopy<std::byte, 9> m_s9;
  SpecificMemoryCopy<Int16, 5> m_s10;
  SpecificMemoryCopy<Int32, 3> m_s12;

  SpecificMemoryCopy<Int64, 2> m_s16;
  SpecificMemoryCopy<Int64, 3> m_s24;
  SpecificMemoryCopy<Int64, 4> m_s32;
  SpecificMemoryCopy<Int64, 5> m_s40;
  SpecificMemoryCopy<Int64, 6> m_s48;
  SpecificMemoryCopy<Int64, 7> m_s56;
  SpecificMemoryCopy<Int64, 8> m_s64;
  SpecificMemoryCopy<Int64, 9> m_s72;

  std::array<ISpecificMemoryCopy*, NB_COPIER> m_copier;
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
  Int64 one_data_size = m_datatype_size;
  Int64 v_one_data_size = v.datatypeSize();
  if (one_data_size != v_one_data_size)
    ARCANE_FATAL("Datatype size are not equal this={0} v={1}",
                 one_data_size, v_one_data_size);

  Int64 nb_index = indexes.size();
  if (nb_index == 0)
    return;

  auto source = v.bytes();
  auto destination = bytes();

  auto* copier = global_copy_list.copier(one_data_size);
  if (copier) {
    copier->copyFrom(indexes, source, destination);
    return;
  }

  for (Int32 i = 0; i < nb_index; ++i) {
    Int64 zindex = i * one_data_size;
    Int64 zci = indexes[i] * one_data_size;
    for (Integer z = 0; z < one_data_size; ++z)
      destination[zindex + z] = source[zci + z];
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ConstMemoryView::
copyToIndexesHost(MutableMemoryView v, Span<const Int32> indexes)
{
  Int64 one_data_size = m_datatype_size;
  Int64 v_one_data_size = v.datatypeSize();
  if (one_data_size != v_one_data_size)
    ARCANE_FATAL("Datatype size are not equal this={0} v={1}",
                 one_data_size, v_one_data_size);

  Int64 nb_index = indexes.size();
  if (nb_index == 0)
    return;

  auto source = bytes();
  auto destination = v.bytes();

  auto* copier = global_copy_list.copier(one_data_size);
  if (copier) {
    copier->copyTo(indexes, source, destination);
    return;
  }

  for (Int32 i = 0; i < nb_index; ++i) {
    Int64 zindex = i * one_data_size;
    Int64 zci = indexes[i] * one_data_size;
    for (Integer z = 0; z < one_data_size; ++z)
      destination[zci + z] = source[zindex + z];
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
