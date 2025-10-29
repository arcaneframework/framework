// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MemoryUtils.cc                                              (C) 2000-2025 */
/*                                                                           */
/* Fonctions utilitaires de gestion mémoire.                                 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/common/MemoryUtils.h"

#include "arccore/base/FatalErrorException.h"
#include "arccore/common/internal/SpecificMemoryCopyList.h"

// Pour std::memmove
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
using RunQueue = Accelerator::RunQueue;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{
  impl::SpecificMemoryCopyList<impl::IndexedCopyTraits> global_copy_list;
  impl::ISpecificMemoryCopyList* default_global_copy_list = nullptr;

  impl::ISpecificMemoryCopyList* _getDefaultCopyList(const RunQueue* queue)
  {
    if (queue && !default_global_copy_list)
      ARCCORE_FATAL("No instance of copier is available for RunQueue");
    if (default_global_copy_list && queue)
      return default_global_copy_list;
    return &global_copy_list;
  }
  Int32 _checkDataTypeSize(const TraceInfo& trace, Int32 data_size1, Int32 data_size2)
  {
    if (data_size1 != data_size2)
      throw FatalErrorException(trace, String::format("Datatype size are not equal this={0} v={1}", data_size1, data_size2));
    return data_size1;
  }
} // namespace

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void impl::ISpecificMemoryCopyList::
setDefaultCopyListIfNotSet(ISpecificMemoryCopyList* ptr)
{
  if (!default_global_copy_list) {
    default_global_copy_list = ptr;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MemoryUtils::
copyHost(MutableMemoryView destination, ConstMemoryView source)
{
  auto b_source = source.bytes();
  auto b_destination = destination.bytes();
  Int64 source_size = b_source.size();
  if (source_size == 0)
    return;
  Int64 destination_size = b_destination.size();
  if (source_size > destination_size)
    ARCCORE_FATAL("Destination is too small source_size={0} destination_size={1}",
                 source_size, destination_size);
  auto* destination_data = b_destination.data();
  auto* source_data = b_source.data();
  ARCCORE_CHECK_POINTER(destination_data);
  ARCCORE_CHECK_POINTER(source_data);
  std::memmove(destination_data, source_data, source_size);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MemoryUtils::
copyHostWithIndexedDestination(MutableMemoryView destination, ConstMemoryView source,
                    Span<const Int32> indexes)
{
  copyWithIndexedDestination(destination, source, indexes.smallView(), nullptr);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MemoryUtils::
copyWithIndexedDestination(MutableMemoryView destination, ConstMemoryView source,
                SmallSpan<const Int32> indexes, RunQueue* queue)
{

  Int32 one_data_size = _checkDataTypeSize(A_FUNCINFO, destination.datatypeSize(), source.datatypeSize());

  Int64 nb_index = indexes.size();
  if (nb_index == 0)
    return;

  auto b_source = source.bytes();
  auto b_destination = destination.bytes();

  _getDefaultCopyList(queue)->copyFrom(one_data_size, { indexes, b_source, b_destination, queue });
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MemoryUtils::
fillIndexed(MutableMemoryView destination, ConstMemoryView source,
            SmallSpan<const Int32> indexes, const RunQueue* queue)
{
  Int32 one_data_size = _checkDataTypeSize(A_FUNCINFO, destination.datatypeSize(), source.datatypeSize());

  Int64 nb_index = indexes.size();
  if (nb_index == 0)
    return;

  auto b_source = source.bytes();
  auto b_destination = destination.bytes();

  _getDefaultCopyList(queue)->fill(one_data_size, { indexes, b_source, b_destination, queue });
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MemoryUtils::
fill(MutableMemoryView destination, ConstMemoryView source, const RunQueue* queue)
{
  Int32 one_data_size = _checkDataTypeSize(A_FUNCINFO, destination.datatypeSize(), source.datatypeSize());

  auto b_source = source.bytes();
  auto b_destination = destination.bytes();

  _getDefaultCopyList(queue)->fill(one_data_size, { {}, b_source, b_destination, queue });
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MemoryUtils::
copyHostWithIndexedSource(MutableMemoryView destination, ConstMemoryView source,
                  Span<const Int32> indexes)
{
  copyWithIndexedSource(destination, source, indexes.smallView(), nullptr);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MemoryUtils::
copyWithIndexedSource(MutableMemoryView destination, ConstMemoryView source,
              SmallSpan<const Int32> indexes,
              RunQueue* queue)
{
  Int32 one_data_size = _checkDataTypeSize(A_FUNCINFO, source.datatypeSize(), destination.datatypeSize());

  Int64 nb_index = indexes.size();
  if (nb_index == 0)
    return;

  auto b_source = source.bytes();
  auto b_destination = destination.bytes();

  _getDefaultCopyList(queue)->copyTo(one_data_size, { indexes, b_source, b_destination, queue });
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MemoryUtils::
copyWithIndexedDestination(MutableMultiMemoryView destination, ConstMemoryView source,
                SmallSpan<const Int32> indexes, RunQueue* queue)
{
  Int32 one_data_size = _checkDataTypeSize(A_FUNCINFO, destination.datatypeSize(), source.datatypeSize());

  Int64 nb_index = indexes.size();
  if (nb_index == 0)
    return;

  _getDefaultCopyList(queue)->copyFrom(one_data_size, { indexes, destination.views(), source.bytes(), queue });
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MemoryUtils::
fillIndexed(MutableMultiMemoryView destination, ConstMemoryView source,
            SmallSpan<const Int32> indexes, RunQueue* queue)
{
  Int32 one_data_size = _checkDataTypeSize(A_FUNCINFO, destination.datatypeSize(), source.datatypeSize());

  Int64 nb_index = indexes.size();
  if (nb_index == 0)
    return;

  _getDefaultCopyList(queue)->fill(one_data_size, { indexes, destination.views(), source.bytes(), queue });
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MemoryUtils::
fill(MutableMultiMemoryView destination, ConstMemoryView source, RunQueue* queue)
{
  Int32 one_data_size = _checkDataTypeSize(A_FUNCINFO, destination.datatypeSize(), source.datatypeSize());

  _getDefaultCopyList(queue)->fill(one_data_size, { {}, destination.views(), source.bytes(), queue });
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MemoryUtils::
copyWithIndexedSource(MutableMemoryView destination, ConstMultiMemoryView source,
              SmallSpan<const Int32> indexes, RunQueue* queue)
{
  Int32 one_data_size = _checkDataTypeSize(A_FUNCINFO, destination.datatypeSize(), source.datatypeSize());

  Int64 nb_index = indexes.size();
  if (nb_index == 0)
    return;

  _getDefaultCopyList(queue)->copyTo(one_data_size, { indexes, source.views(), destination.bytes(), queue });
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCCORE_COMMON_EXPORT void
arccorePrintSpecificMemoryStats()
{
  if (arccoreIsCheck()) {
    // N'affiche que pour les tests
    //global_copy_list.printStats();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
