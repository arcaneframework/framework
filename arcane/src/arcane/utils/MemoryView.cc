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

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MutableMemoryView::
copyHost(MemoryView v)
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
copyFromIndexesHost(MemoryView v, Span<const Int32> indexes)
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

  for (Int32 i = 0; i < nb_index; ++i) {
    Int64 zindex = i * one_data_size;
    Int64 zci = indexes[i] * one_data_size;
    for (Integer z = 0; z < one_data_size; ++z)
      destination[zindex + z] = source[zci + z];
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MemoryView::
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
