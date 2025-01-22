// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MemoryResourceMng.cc                                       (C) 2000-2025 */
/*                                                                           */
/* Gestion des ressources mémoire pour les CPU et accélérateurs.             */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/internal/MemoryResourceMng.h"

#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/Array.h"
#include "arcane/utils/MemoryView.h"
#include "arcane/utils/MemoryAllocator.h"
#include "arcane/utils/MemoryUtils.h"
#include "arcane/utils/internal/MemoryUtilsInternal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{
  inline bool _isHost(eMemoryRessource r)
  {
    // Si on sait pas, considère qu'on est accessible de puis l'hôte.
    if (r == eMemoryResource::Unknown)
      return true;
    if (r == eMemoryResource::Host || r == eMemoryResource::UnifiedMemory || r == eMemoryResource::HostPinned)
      return true;
    return false;
  }

} // namespace

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class DefaultHostMemoryCopier
: public IMemoryCopier
{
 public:

  void copy(ConstMemoryView from, eMemoryResource from_mem,
            MutableMemoryView to, eMemoryResource to_mem,
            [[maybe_unused]] const RunQueue* queue) override
  {
    // Sans support accélérateur, on peut juste faire un 'memcpy' si la mémoire
    // est accessible depuis le CPU

    if (!_isHost(from_mem))
      ARCANE_FATAL("Source buffer is not accessible from host and no copier provided (location={0})",
                   from_mem);

    if (!_isHost(to_mem))
      ARCANE_FATAL("Destination buffer is not accessible from host and no copier provided (location={0})",
                   to_mem);

    to.copyHost(from);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MemoryResourceMng::
MemoryResourceMng()
: m_default_memory_copier(new DefaultHostMemoryCopier())
, m_copier(m_default_memory_copier.get())
{
  // Par défaut on utilise l'allocateur CPU. Les allocateurs spécifiques pour
  // les accélérateurs seront positionnés lorsqu'on aura choisi le runtime
  // accélérateur
  IMemoryAllocator* a = AlignedMemoryAllocator::Simd();
  setAllocator(eMemoryRessource::Host, a);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

int MemoryResourceMng::
_checkValidResource(eMemoryResource r)
{
  int x = (int)r;
  if (x <= 0 || x >= Arccore::ARCCORE_NB_MEMORY_RESOURCE)
    ARCANE_FATAL("Invalid value '{0}'. Valid range is '1' to '{1}'", x, Arccore::ARCCORE_NB_MEMORY_RESOURCE - 1);
  return x;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IMemoryAllocator* MemoryResourceMng::
getAllocator(eMemoryResource r, bool throw_if_not_found)
{
  int x = _checkValidResource(r);
  IMemoryAllocator* a = m_allocators[x];

  // Si pas d'allocateur spécifique et qu'on n'est pas sur accélérateur,
  // utilise platform::getAcceleratorHostMemoryAllocator().
  if (!a && !m_is_accelerator) {
    if (r == eMemoryResource::UnifiedMemory || r == eMemoryResource::HostPinned) {
      eMemoryResource mem = MemoryUtils::getDefaultDataMemoryResource();
      a = m_allocators[static_cast<int>(mem)];
      if (!a)
        a = m_allocators[static_cast<int>(eMemoryResource::Host)];
    }
  }

  if (!a && throw_if_not_found)
    ARCANE_FATAL("Allocator for resource '{0}' is not available", r);

  return a;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IMemoryAllocator* MemoryResourceMng::
getAllocator(eMemoryResource r)
{
  return getAllocator(r, true);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MemoryResourceMng::
setAllocator(eMemoryResource r, IMemoryAllocator* allocator)
{
  int x = _checkValidResource(r);
  m_allocators[x] = allocator;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MemoryResourceMng::
copy(ConstMemoryView from, eMemoryResource from_mem,
     MutableMemoryView to, eMemoryResource to_mem, const RunQueue* queue)
{
  Int64 from_size = from.bytes().size();
  Int64 to_size = to.bytes().size();
  if (from_size > to_size)
    ARCANE_FATAL("Destination copy is too small (to_size={0} from_size={1})", to_size, from_size);

  m_copier->copy(from, from_mem, to, to_mem, queue);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MemoryResourceMng::
genericCopy(ConstMemoryView from, MutableMemoryView to)
{
  IMemoryResourceMng* mrm = MemoryUtils::getDataMemoryResourceMng();
  eMemoryResource mem_type = eMemoryResource::Unknown;
  mrm->_internal()->copy(from, mem_type, to, mem_type, nullptr);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
