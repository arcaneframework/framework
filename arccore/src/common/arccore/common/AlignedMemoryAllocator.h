// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AlignedMemoryAllocator.h                                    (C) 2000-2025 */
/*                                                                           */
/* Allocateur mémoire par défaut.                                            */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_COMMON_ALIGNEDMEMORYALLOCATOR_H
#define ARCCORE_COMMON_ALIGNEDMEMORYALLOCATOR_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/common/IMemoryAllocator.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Impl
{
  extern "C++" ARCCORE_COMMON_EXPORT size_t
  adjustMemoryCapacity(size_t wanted_capacity, size_t element_size);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Allocateur mémoire avec alignement mémoire spécifique.
 *
 * Cette classe s'utilise via les deux méthodes publiques Simd()
 * et CacheLine() qui retournent respectivement un allocateur avec
 * un alignement adéquat pour autoriser la vectorisation et un allocateur
 * aligné sur une ligne de cache.
 */
class ARCCORE_COMMON_EXPORT AlignedMemoryAllocator
: public IMemoryAllocator
{
 private:

  static AlignedMemoryAllocator SimdAllocator;
  static AlignedMemoryAllocator CacheLineAllocator;

 public:

  // TODO: essayer de trouver les bonnes valeurs en fonction de la cible.
  // 64 est OK pour toutes les architectures x64 à la fois pour le SIMD
  // et la ligne de cache.

  // IMPORTANT : Si on change la valeur ici, il faut changer la taille de
  // l'alignement de ArrayImplBase.

  // TODO Pour l'instant seul un alignement sur 64 est autorisé. Pour
  // autoriser d'autres valeurs, il faut modifier l'implémentation dans
  // ArrayImplBase.

  // TODO marquer les méthodes comme 'final'.

  //! Alignement pour les structures utilisant la vectorisation
  static constexpr Integer simdAlignment() { return 64; }
  //! Alignement pour une ligne de cache.
  static constexpr Integer cacheLineAlignment() { return 64; }

  /*!
   * \brief Allocateur garantissant l'alignement pour utiliser
   * la vectorisation sur la plateforme cible.
   *
   * Il s'agit de l'alignement pour le type plus restrictif et donc il
   * est possible d'utiliser cet allocateur pour toutes les structures vectorielles.
   */
  static AlignedMemoryAllocator* Simd()
  {
    return &SimdAllocator;
  }

  /*!
   * \brief Allocateur garantissant l'alignement sur une ligne de cache.
   */
  static AlignedMemoryAllocator* CacheLine()
  {
    return &CacheLineAllocator;
  }

 protected:

  explicit AlignedMemoryAllocator(Int32 alignment)
  : m_alignment(static_cast<size_t>(alignment))
  {}

 public:

  bool hasRealloc(MemoryAllocationArgs) const override { return false; }
  AllocatedMemoryInfo allocate(MemoryAllocationArgs args, Int64 new_size) override;
  AllocatedMemoryInfo reallocate(MemoryAllocationArgs args, AllocatedMemoryInfo current_ptr, Int64 new_size) override;
  void deallocate(MemoryAllocationArgs args, AllocatedMemoryInfo ptr) override;
  Int64 adjustedCapacity(MemoryAllocationArgs args, Int64 wanted_capacity, Int64 element_size) const override;
  size_t guaranteedAlignment(MemoryAllocationArgs) const override { return m_alignment; }
  eMemoryResource memoryResource() const override { return eMemoryResource::Host; }

 private:

  size_t m_alignment;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

