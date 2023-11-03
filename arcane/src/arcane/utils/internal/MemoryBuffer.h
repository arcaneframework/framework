// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MemoryBuffer.h                                              (C) 2000-2023 */
/*                                                                           */
/* Buffer mémoire.                                                           */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IMPL_MEMORYBUFFER_H
#define ARCANE_IMPL_MEMORYBUFFER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Array.h"
#include "arcane/utils/Ref.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Gestion d'un buffer mémoire.
 *
 */
class ARCANE_UTILS_EXPORT MemoryBuffer
{
 private:

  explicit MemoryBuffer(IMemoryAllocator* allocator)
  : m_buffer(allocator)
  {}

 public:

  /*!
  * \brief Créé une instance de \a MemoryBuffer.
  *
  * L'allocateur \a allocator doit rester valide durant toute
  * la durée de vie de l'instance créée.
  */
  static Ref<MemoryBuffer> create(IMemoryAllocator* allocator)
  {
    auto* memory = new MemoryBuffer(allocator);
    Ref<MemoryBuffer> ref_memory = makeRef<MemoryBuffer>(memory);
    return ref_memory;
  }

 public:

  /*!
   * \brief Redimensionne la zone mémoire.
   *
   * Aucune initialisation n'est effectuée. Si la taille diminue
   * le resize() est sans effet.
   */
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
