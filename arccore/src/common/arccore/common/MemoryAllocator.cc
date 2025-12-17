// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MemoryAllocator.cc                                          (C) 2000-2025 */
/*                                                                           */
/* Allocateurs mémoires.                                                     */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ArgumentException.h"
#include "arccore/base/NotSupportedException.h"
#include "arccore/base/NotImplementedException.h"
#include "arccore/base/TraceInfo.h"

#include "arccore/common/DefaultMemoryAllocator.h"
#include "arccore/common/AlignedMemoryAllocator.h"
#include "arccore/common/AllocatedMemoryInfo.h"

#include <cstdlib>
#include <cstring>
#include <errno.h>

#if defined(ARCCORE_OS_WIN32)
#include <malloc.h>
#endif

#include <iostream>

#if defined(ARCCORE_OS_LINUX) || defined(ARCCORE_OS_MACOS)
#define ARCCORE_USE_POSIX_MEMALIGN
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

DefaultMemoryAllocator DefaultMemoryAllocator::shared_null_instance;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

size_t IMemoryAllocator::
guarantedAlignment(MemoryAllocationArgs args) const
{
  return this->guaranteedAlignment(args);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void IMemoryAllocator::
notifyMemoryArgsChanged(MemoryAllocationArgs, MemoryAllocationArgs, AllocatedMemoryInfo)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void IMemoryAllocator::
copyMemory([[maybe_unused]] MemoryAllocationArgs args, AllocatedMemoryInfo destination,
           AllocatedMemoryInfo source)
{
  std::memcpy(destination.baseAddress(), source.baseAddress(), source.size());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool DefaultMemoryAllocator::
hasRealloc(MemoryAllocationArgs) const
{
  return true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AllocatedMemoryInfo DefaultMemoryAllocator::
allocate(MemoryAllocationArgs, Int64 new_size)
{
  return { ::malloc(new_size), new_size };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AllocatedMemoryInfo DefaultMemoryAllocator::
reallocate(MemoryAllocationArgs, AllocatedMemoryInfo current_ptr, Int64 new_size)
{
  return { ::realloc(current_ptr.baseAddress(), new_size), new_size };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DefaultMemoryAllocator::
deallocate(MemoryAllocationArgs,AllocatedMemoryInfo ptr)
{
  ::free(ptr.baseAddress());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 DefaultMemoryAllocator::
adjustedCapacity(MemoryAllocationArgs, Int64 wanted_capacity, Int64 ) const
{
  return wanted_capacity;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AlignedMemoryAllocator AlignedMemoryAllocator::
SimdAllocator(AlignedMemoryAllocator::simdAlignment());

AlignedMemoryAllocator AlignedMemoryAllocator::
CacheLineAllocator(AlignedMemoryAllocator::cacheLineAlignment());

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * NOTE: Normalement les fonctions _mm_alloc() et _mm_free() permettent
 * d'allouer de la mémoire alignée. Il faudrait vérifier si elles sont
 * disponibles partout.
 */

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AllocatedMemoryInfo AlignedMemoryAllocator::
allocate([[maybe_unused]] MemoryAllocationArgs args, Int64 new_size)
{
#if defined(ARCCORE_USE_POSIX_MEMALIGN)
  void* ptr = nullptr;
  int e = ::posix_memalign(&ptr, m_alignment, new_size);
  if (e == EINVAL)
    throw ArgumentException(A_FUNCINFO, "Invalid argument to posix_memalign");
  if (e == ENOMEM)
    return AllocatedMemoryInfo(nullptr);
  return AllocatedMemoryInfo(ptr, new_size);
#elif defined(ARCCORE_OS_WIN32)
  return AllocatedMemoryInfo(_aligned_malloc(new_size, m_alignment), new_size);
#else
  throw NotImplementedException(A_FUNCINFO);
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AllocatedMemoryInfo AlignedMemoryAllocator::
reallocate([[maybe_unused]] MemoryAllocationArgs args, AllocatedMemoryInfo current_ptr, Int64 new_size)
{
#if defined(ARCCORE_USE_POSIX_MEMALIGN)
  ARCCORE_UNUSED(current_ptr);
  ARCCORE_UNUSED(new_size);
  throw NotSupportedException(A_FUNCINFO);
#elif defined(ARCCORE_OS_WIN32)
  return AllocatedMemoryInfo(_aligned_realloc(current_ptr.baseAddress(), new_size, m_alignment), new_size);
#else
  throw NotImplementedException(A_FUNCINFO);
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AlignedMemoryAllocator::
deallocate([[maybe_unused]] MemoryAllocationArgs args, AllocatedMemoryInfo ptr)
{
#if defined(ARCCORE_USE_POSIX_MEMALIGN)
  ::free(ptr.baseAddress());
#elif defined(ARCCORE_OS_WIN32)
  return _aligned_free(ptr.baseAddress());
#else
  throw NotImplementedException(A_FUNCINFO);
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{
  bool is_verbose = false;
}

size_t
adjustMemoryCapacity(size_t wanted_capacity, size_t element_size, size_t alignment)
{
  if (element_size == 0)
    return wanted_capacity;
  // Si \a element_size est plus petit que \a m_alignment, considère que
  // la mémoire allouée doit être un multiple de l'alignement.
  // (On pourrait être plus restrictif suivant les types mais ce n'est en
  // général pas utile).
  size_t block_size = alignment / element_size;
  if (block_size <= 1)
    return wanted_capacity;

  // Si l'alignement n'est pas un multiple de la taille d'un élément,
  // cela signifie que l'élément ne sera pas utilisé pour la vectorisation.
  // Il n'est donc pas nécessaire dans ce cas de modifier la capacité.
  size_t nb_element = alignment % element_size;
  if (nb_element != 0)
    return wanted_capacity;

  if (is_verbose)
    std::cout << " wanted_capacity=" << wanted_capacity
              << " element_size=" << element_size
              << " block_size=" << block_size << '\n';

  // Ajoute à la capacité ce qu'il faut pour que le module soit 0.
  size_t modulo = wanted_capacity % block_size;
  if (modulo != 0)
    wanted_capacity += (block_size - modulo);
  if (is_verbose)
    std::cout << " final_wanted_capacity=" << wanted_capacity
              << " modulo=" << modulo << '\n';
  ARCCORE_ASSERT(((wanted_capacity % block_size) == 0), ("Bad capacity"));
  return wanted_capacity;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 AlignedMemoryAllocator::
adjustedCapacity([[maybe_unused]] MemoryAllocationArgs args, Int64 wanted_capacity, Int64 element_size) const
{
  return adjustMemoryCapacity(wanted_capacity, element_size, m_alignment);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AllocatedMemoryInfo PrintableMemoryAllocator::
allocate(MemoryAllocationArgs args, Int64 new_size)
{
  AllocatedMemoryInfo mem_info = Base::allocate(args, new_size);
  std::cout << "DEF_ARRAY_ALLOCATE new_size=" << new_size << " ptr=" << mem_info.baseAddress() << '\n';
  return mem_info;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AllocatedMemoryInfo PrintableMemoryAllocator::
reallocate(MemoryAllocationArgs args, AllocatedMemoryInfo current_ptr, Int64 new_size)
{
  AllocatedMemoryInfo mem_info = Base::reallocate(args, current_ptr, new_size);
  std::cout << "DEF_ARRAY_REALLOCATE new_size=" << new_size
            << " current_ptr=" << current_ptr.baseAddress()
            << " new_ptr=" << mem_info.baseAddress() << '\n';
  return mem_info;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void PrintableMemoryAllocator::
deallocate(MemoryAllocationArgs args, AllocatedMemoryInfo ptr)
{
  std::cout << "DEF_ARRAY_DEALLOCATE ptr=" << ptr.baseAddress() << '\n';
  Base::deallocate(args, ptr);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
