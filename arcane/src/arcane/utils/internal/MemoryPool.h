// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MemoryPool.h                                                (C) 2000-2024 */
/*                                                                           */
/* Classe pour gérer une liste de zone allouées.                             */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_INTERNAL_MEMORYPOOL_H
#define ARCANE_UTILS_INTERNAL_MEMORYPOOL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcaneGlobal.h"

#include <memory>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Interface d'un allocateur pour un MemoryPool.
 *
 * Cette interface fonctionne à la manière d'un malloc/free à ceci prêt qu'il
 * faut fournir la taille alloué pour un bloc pour la libération de ce dernier.
 * L'utilisateur de cette interface doit donc gérer la conservation de cette
 * information.
 */
class ARCANE_UTILS_EXPORT IMemoryPoolAllocator
{
 public:

  virtual ~IMemoryPoolAllocator() = default;

 public:

  //! Alloue un bloc pour \a size octets
  virtual void* allocateMemory(size_t size) = 0;
  //! Libère le bloc situé à l'adresse \a address contenant \a size octets
  virtual void freeMemory(void* address, size_t size) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Classe pour gérer une liste de zone allouées.
 *
 * Cette classe utilise une sémantique par référence.
 */
class ARCANE_UTILS_EXPORT MemoryPool
: public IMemoryPoolAllocator
{
  class Impl;

 public:

  explicit MemoryPool(IMemoryPoolAllocator* allocator, const String& name);
  ~MemoryPool();

 public:

  void* allocateMemory(size_t size) override;
  void freeMemory(void* ptr, size_t size) override;
  void dumpStats();
  String name() const;

 private:

  std::shared_ptr<Impl> m_p;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
