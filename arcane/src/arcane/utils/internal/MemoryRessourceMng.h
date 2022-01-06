// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMemoryRessourceMng.h                                       (C) 2000-2022 */
/*                                                                           */
/* Gestion des ressources mémoire pour les CPU et accélérateurs.             */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_INTERNAL_MEMORYRESSOURCEMNG_H
#define ARCANE_UTILS_INTERNAL_MEMORYRESSOURCEMNG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/IMemoryRessourceMng.h"
#include "arcane/utils/internal/IMemoryRessourceMngInternal.h"

#include <array>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Gestion des ressources mémoire pour les CPU et accélérateurs.
 */
class ARCANE_UTILS_EXPORT MemoryRessourceMng
: public IMemoryRessourceMng
, public IMemoryRessourceMngInternal
{
 public:

  MemoryRessourceMng();

 public:

  IMemoryAllocator* getAllocator(eMemoryRessource r) override;

 public:

  void setAllocator(eMemoryRessource r, IMemoryAllocator* allocator) override;

 public:

  //! Interface interne
  virtual IMemoryRessourceMngInternal* _internal() { return this; }

 private:

  std::array<IMemoryAllocator*, NB_MEMORY_RESSOURCE> m_allocators;

 private:

  inline int _checkValidRessource(eMemoryRessource r);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
