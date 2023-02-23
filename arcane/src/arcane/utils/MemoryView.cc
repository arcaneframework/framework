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

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
