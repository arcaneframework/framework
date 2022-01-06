// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MemoryRessourceMng.cc                                       (C) 2000-2021 */
/*                                                                           */
/* Gestion des ressources mémoire pour les CPU et accélérateurs.             */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/internal/MemoryRessourceMng.h"

#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/Array.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MemoryRessourceMng::
MemoryRessourceMng()
{
  std::fill(m_allocators.begin(), m_allocators.end(), nullptr);
  // Par défaut on utilise l'allocateur CPU. Les allocateurs spécifiques pour
  // les accélérateurs seront positionnés lorsqu'on aura choisi le runtime
  // accélérateur
  IMemoryAllocator* a = AlignedMemoryAllocator::Simd();
  setAllocator(eMemoryRessource::Host, a);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

int MemoryRessourceMng::
_checkValidRessource(eMemoryRessource r)
{
  int x = (int)r;
  if (x <= 0 || x >= NB_MEMORY_RESSOURCE)
    ARCANE_FATAL("Invalid value '{0}'. Valid range is '1' to '{1}'", x, NB_MEMORY_RESSOURCE - 1);
  return x;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IMemoryAllocator* MemoryRessourceMng::
getAllocator(eMemoryRessource r)
{
  int x = _checkValidRessource(r);
  IMemoryAllocator* a = m_allocators[(int)r];

  // Si pas d'allocateur spécifique, utilise platform::getAcceleratorHostMemoryAllocator()
  // pour compatibilité avec l'existant
  if (r == eMemoryRessource::UnifiedMemory && !a) {
    a = platform::getAcceleratorHostMemoryAllocator();
    if (!a)
      a = m_allocators[(int)eMemoryRessource::Host];
  }

  if (!a)
    ARCANE_FATAL("Allocator for ressource '{0}' is not available", x);

  return a;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MemoryRessourceMng::
setAllocator(eMemoryRessource r, IMemoryAllocator* allocator)
{
  int x = _checkValidRessource(r);
  m_allocators[x] = allocator;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
