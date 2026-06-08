// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MemoryResourceMng.cc                                        (C) 2000-2025 */
/*                                                                           */
/* Memory resource management for CPUs and accelerators.                     */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/common/internal/MemoryResourceMng.h"

#include "arccore/base/FatalErrorException.h"
#include "arccore/base/PlatformUtils.h"
#include "arccore/base/MemoryView.h"
#include "arccore/common/Array.h"
#include "arccore/common/AlignedMemoryAllocator.h"
#include "arccore/common/MemoryUtils.h"
#include "arccore/common/internal/MemoryUtilsInternal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{
  bool _isHost(eMemoryResource r)
  {
    // If unknown, assume it is accessible from the host.
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
    // Without accelerator support, we can just perform a 'memcpy' if the memory
    // is accessible from the CPU

    if (!_isHost(from_mem))
      ARCCORE_FATAL("Source buffer is not accessible from host and no copier provided (location={0})",
                    from_mem);

    if (!_isHost(to_mem))
      ARCCORE_FATAL("Destination buffer is not accessible from host and no copier provided (location={0})",
                    to_mem);

    MemoryUtils::copyHost(to, from);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MemoryResourceMng::
MemoryResourceMng()
: m_default_memory_copier(new DefaultHostMemoryCopier())
, m_copier(m_default_memory_copier.get())
{
  // By default, we use the CPU allocator. Specific allocators for
  // accelerators will be set when the accelerator runtime is chosen
  IMemoryAllocator* a = AlignedMemoryAllocator::Simd();
  setAllocator(eMemoryResource::Host, a);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

int MemoryResourceMng::
_checkValidResource(eMemoryResource r)
{
  int x = (int)r;
  if (x <= 0 || x >= ARCCORE_NB_MEMORY_RESOURCE)
    ARCCORE_FATAL("Invalid value '{0}'. Valid range is '1' to '{1}'", x, ARCCORE_NB_MEMORY_RESOURCE - 1);
  return x;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IMemoryAllocator* MemoryResourceMng::
getAllocator(eMemoryResource r, bool throw_if_not_found)
{
  int x = _checkValidResource(r);
  IMemoryAllocator* a = m_allocators[x];

  // If no specific allocator is found and we are not on an accelerator,
  // use Platform::getAcceleratorHostMemoryAllocator().
  if (!a && !m_is_accelerator) {
    if (r == eMemoryResource::UnifiedMemory || r == eMemoryResource::HostPinned) {
      eMemoryResource mem = MemoryUtils::getDefaultDataMemoryResource();
      a = m_allocators[static_cast<int>(mem)];
      if (!a)
        a = m_allocators[static_cast<int>(eMemoryResource::Host)];
    }
  }

  if (!a && throw_if_not_found)
    ARCCORE_FATAL("Allocator for resource '{0}' is not available", r);

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

IMemoryPool* MemoryResourceMng::
getMemoryPoolOrNull(eMemoryResource r)
{
  int x = _checkValidResource(r);
  return m_memory_pools[x];
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
setMemoryPool(eMemoryResource r, IMemoryPool* pool)
{
  int x = _checkValidResource(r);
  m_memory_pools[x] = pool;
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
    ARCCORE_FATAL("Destination copy is too small (to_size={0} from_size={1})", to_size, from_size);

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

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
