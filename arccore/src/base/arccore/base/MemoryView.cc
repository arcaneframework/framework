// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MemoryView.cc                                               (C) 2000-2025 */
/*                                                                           */
/* Vues constantes ou modifiables sur une zone mémoire.                      */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/MemoryView.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MutableMemoryView
makeMutableMemoryView(void* ptr, Int32 datatype_size, Int64 nb_element)
{
  Span<std::byte> bytes(reinterpret_cast<std::byte*>(ptr), datatype_size * nb_element);
  return { bytes, datatype_size, nb_element };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ConstMemoryView
makeConstMemoryView(const void* ptr, Int32 datatype_size, Int64 nb_element)
{
  Span<const std::byte> bytes(reinterpret_cast<const std::byte*>(ptr), datatype_size * nb_element);
  return { bytes, datatype_size, nb_element };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
