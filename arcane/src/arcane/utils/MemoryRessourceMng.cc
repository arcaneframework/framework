// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MemoryRessourceMng.cc                                       (C) 2000-2023 */
/*                                                                           */
/* Gestion des ressources mémoire pour les CPU et accélérateurs.             */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/internal/MemoryRessourceMng.h"

#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/Array.h"
#include "arcane/utils/MemoryView.h"
#include "arcane/utils/MemoryAllocator.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
namespace
{
  const char* _toName(eMemoryRessource r)
  {
    switch (r) {
    case eMemoryRessource::Unknown:
      return "Unknown";
    case eMemoryRessource::Host:
      return "Host";
    case eMemoryRessource::HostPinned:
      return "HostPinned";
    case eMemoryRessource::Device:
      return "Device";
    case eMemoryRessource::UnifiedMemory:
      return "UnifiedMemory";
    }
    return "Invalid";
  }

} // namespace

extern "C++" ARCANE_UTILS_EXPORT std::ostream&
operator<<(std::ostream& o, eMemoryRessource r)
{
  o << _toName(r);
  return o;
}

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

namespace
{
  inline bool _isHost(eMemoryRessource r)
  {
    // Si on sait pas, considère qu'on est accessible de puis l'hôte.
    if (r == eMemoryRessource::Unknown)
      return true;
    if (r == eMemoryRessource::Host || r == eMemoryRessource::UnifiedMemory || r == eMemoryRessource::HostPinned)
      return true;
    return false;
  }
} // namespace

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MemoryRessourceMng::
copy(ConstMemoryView from, eMemoryRessource from_mem,
     MutableMemoryView to, eMemoryRessource to_mem, RunQueue* queue)
{
  Int64 from_size = from.bytes().size();
  Int64 to_size = to.bytes().size();
  if (from_size > to_size)
    ARCANE_FATAL("Destination copy is too small (to_size={0} from_size={1})", to_size, from_size);

  // Utilise l'instance spécifique si elle disponible
  if (m_copier) {
    m_copier->copy(from, from_mem, to, to_mem, queue);
    return;
  }

  // Sinon, on peut juste faire un 'memcpy' si la mémoire est accessible
  // depuis le CPU

  if (!_isHost(from_mem))
    ARCANE_FATAL("Source buffer is not accessible from host and no copier provided (location={0})",
                 from_mem);

  if (!_isHost(to_mem))
    ARCANE_FATAL("Destination buffer is not accessible from host and no copier provided (location={0})",
                 to_mem);

  to.copyHost(from);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MemoryRessourceMng::
genericCopy(ConstMemoryView from, MutableMemoryView to)
{
  IMemoryRessourceMng* mrm = platform::getDataMemoryRessourceMng();
  eMemoryRessource mem_type = eMemoryRessource::Unknown;
  mrm->_internal()->copy(from, mem_type, to, mem_type, nullptr);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
