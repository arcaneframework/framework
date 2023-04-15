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
#include "arcane/utils/internal/SpecificMemoryCopyList.h"

#include <cstring>

// TODO: ajouter statistiques sur les tailles de 'datatype' utilisées.
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IndexedCopyTraits
{
 public:

  using InterfaceType = ISpecificMemoryCopy;
  template <typename DataType, typename Extent> using SpecificType = SpecificMemoryCopy<DataType, Extent>;
  using RefType = SpecificMemoryCopyRef<IndexedCopyTraits>;
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
  impl::SpecificMemoryCopyList<impl::IndexedCopyTraits> global_copy_list;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MutableMemoryView::
copyHost(ConstMemoryView v)
{
  auto source = v.bytes();
  auto destination = bytes();
  Int64 source_size = source.size();
  if (source_size == 0)
    return;
  Int64 destination_size = destination.size();
  if (source_size > destination_size)
    ARCANE_FATAL("Destination is too small source_size={0} destination_size={1}",
                 source_size, destination_size);
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
