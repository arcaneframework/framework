// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DataSynchronizeMemory.h                                     (C) 2000-2023 */
/*                                                                           */
/* Gestion des allocations mémoire pour les synchronisations.                */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IMPL_DATASYNCHRONIZEMEMORY_H
#define ARCANE_IMPL_DATASYNCHRONIZEMEMORY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Array.h"
#include "arcane/utils/Ref.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
namespace Arcane
{
class IBufferCopier;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Gestion des allocations mémoire pour les synchronisations.
 */
class ARCANE_IMPL_EXPORT DataSynchronizeMemory
{
 public:

  explicit DataSynchronizeMemory(IMemoryAllocator* allocator)
  : m_buffer(allocator)
  {}

 public:

  void resize(Int64 new_size) { m_buffer.resize(new_size); }
  Span<const std::byte> bytes() const { return m_buffer; }
  Span<std::byte> bytes() { return m_buffer; }
  IMemoryAllocator* allocator() const { return m_buffer.allocator(); }

 private:

  //! Buffer contenant les données.
  UniqueArray<std::byte> m_buffer;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
