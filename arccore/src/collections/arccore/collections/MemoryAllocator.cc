// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MemoryAllocator.cc                                          (C) 2000-2023 */
/*                                                                           */
/* Allocateurs mémoires.                                                     */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ArgumentException.h"
#include "arccore/base/NotSupportedException.h"
#include "arccore/base/NotImplementedException.h"
#include "arccore/base/TraceInfo.h"

#include "arccore/collections/IMemoryAllocator.h"

#include <cstdlib>
#include <errno.h>

#if defined(ARCCORE_OS_WIN32)
#include <malloc.h>
#endif

#include <iostream>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

DefaultMemoryAllocator DefaultMemoryAllocator::shared_null_instance;
DefaultMemoryAllocator2 DefaultMemoryAllocator2::shared_null_instance;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool DefaultMemoryAllocator::
hasRealloc() const
{
  return true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void* DefaultMemoryAllocator::
allocate(size_t new_size)
{
  return ::malloc(new_size);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void* DefaultMemoryAllocator::
reallocate(void* current_ptr, size_t new_size)
{
  return ::realloc(current_ptr, new_size);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DefaultMemoryAllocator::
deallocate(void* ptr)
{
  ::free(ptr);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

size_t DefaultMemoryAllocator::
adjustCapacity(size_t wanted_capacity, size_t)
{
  return wanted_capacity;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool DefaultMemoryAllocator2::
hasRealloc(MemoryAllocationArgs) const
{
  return true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void* DefaultMemoryAllocator2::
allocate(size_t new_size, MemoryAllocationArgs)
{
  return ::malloc(new_size);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void* DefaultMemoryAllocator2::
reallocate(void* current_ptr, size_t new_size, MemoryAllocationArgs)
{
  return ::realloc(current_ptr, new_size);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DefaultMemoryAllocator2::
deallocate(void* ptr, MemoryAllocationArgs)
{
  ::free(ptr);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

size_t DefaultMemoryAllocator2::
adjustCapacity(size_t wanted_capacity, size_t, MemoryAllocationArgs)
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

AlignedMemoryAllocator2 AlignedMemoryAllocator2::
SimdAllocator(AlignedMemoryAllocator2::simdAlignment());

AlignedMemoryAllocator2 AlignedMemoryAllocator2::
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

bool AlignedMemoryAllocator::
hasRealloc() const
{
  return false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool AlignedMemoryAllocator2::
hasRealloc(MemoryAllocationArgs) const
{
  return false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void* AlignedMemoryAllocator::
allocate(size_t new_size)
{
#ifdef ARCCORE_OS_LINUX
  void* ptr = nullptr;
  int e = ::posix_memalign(&ptr, m_alignment, new_size);
  if (e == EINVAL)
    throw ArgumentException(A_FUNCINFO, "Invalid argument to posix_memalign");
  if (e == ENOMEM)
    return nullptr;
  return ptr;
#elif defined(ARCCORE_OS_WIN32)
  return _aligned_malloc(new_size, m_alignment);
#else
  throw NotImplementedException(A_FUNCINFO);
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void* AlignedMemoryAllocator2::
allocate(size_t new_size,MemoryAllocationArgs)
{
#ifdef ARCCORE_OS_LINUX
  void* ptr = nullptr;
  int e = ::posix_memalign(&ptr, m_alignment, new_size);
  if (e == EINVAL)
    throw ArgumentException(A_FUNCINFO, "Invalid argument to posix_memalign");
  if (e == ENOMEM)
    return nullptr;
  return ptr;
#elif defined(ARCCORE_OS_WIN32)
  return _aligned_malloc(new_size, m_alignment);
#else
  throw NotImplementedException(A_FUNCINFO);
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void* AlignedMemoryAllocator::
reallocate(void* current_ptr, size_t new_size)
{
#ifdef ARCCORE_OS_LINUX
  ARCCORE_UNUSED(current_ptr);
  ARCCORE_UNUSED(new_size);
  throw NotSupportedException(A_FUNCINFO);
#elif defined(ARCCORE_OS_WIN32)
  return _aligned_realloc(current_ptr, new_size, m_alignment);
#else
  throw NotImplementedException(A_FUNCINFO);
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void* AlignedMemoryAllocator2::
reallocate(void* current_ptr, size_t new_size, MemoryAllocationArgs)
{
#ifdef ARCCORE_OS_LINUX
  ARCCORE_UNUSED(current_ptr);
  ARCCORE_UNUSED(new_size);
  throw NotSupportedException(A_FUNCINFO);
#elif defined(ARCCORE_OS_WIN32)
  return _aligned_realloc(current_ptr, new_size, m_alignment);
#else
  throw NotImplementedException(A_FUNCINFO);
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AlignedMemoryAllocator::
deallocate(void* ptr)
{
#ifdef ARCCORE_OS_LINUX
  ::free(ptr);
#elif defined(ARCCORE_OS_WIN32)
  return _aligned_free(ptr);
#else
  throw NotImplementedException(A_FUNCINFO);
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AlignedMemoryAllocator2::
deallocate(void* ptr, MemoryAllocationArgs)
{
#ifdef ARCCORE_OS_LINUX
  ::free(ptr);
#elif defined(ARCCORE_OS_WIN32)
  return _aligned_free(ptr);
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

size_t AlignedMemoryAllocator::
adjustCapacity(size_t wanted_capacity, size_t element_size)
{
  return adjustMemoryCapacity(wanted_capacity, element_size, m_alignment);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

size_t AlignedMemoryAllocator2::
adjustCapacity(size_t wanted_capacity, size_t element_size, MemoryAllocationArgs)
{
  return adjustMemoryCapacity(wanted_capacity, element_size, m_alignment);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void* PrintableMemoryAllocator::
allocate(size_t new_size)
{
  void* ptr = Base::allocate(new_size);
  std::cout << "DEF_ARRAY_ALLOCATE new_size=" << new_size << " ptr=" << ptr << '\n';
  return ptr;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void* PrintableMemoryAllocator::
reallocate(void* current_ptr, size_t new_size)
{
  void* ptr = Base::reallocate(current_ptr, new_size);
  std::cout << "DEF_ARRAY_REALLOCATE new_size=" << new_size
            << " current_ptr=" << current_ptr
            << " new_ptr=" << ptr << '\n';
  return ptr;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void PrintableMemoryAllocator::
deallocate(void* ptr)
{
  std::cout << "DEF_ARRAY_DEALLOCATE ptr=" << ptr << '\n';
  Base::deallocate(ptr);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool IMemoryAllocator::
hasRealloc(MemoryAllocationArgs) const
{
  return hasRealloc();
}

void* IMemoryAllocator::
allocate(size_t new_size, MemoryAllocationArgs)
{
  return allocate(new_size);
}

void* IMemoryAllocator::
reallocate(void* current_ptr, size_t new_size, MemoryAllocationArgs)
{
  return reallocate(current_ptr, new_size);
}

void IMemoryAllocator::
deallocate(void* ptr, MemoryAllocationArgs)
{
  return deallocate(ptr);
}

size_t IMemoryAllocator::
adjustCapacity(size_t wanted_capacity, size_t element_size, MemoryAllocationArgs)
{
  return adjustCapacity(wanted_capacity, element_size);
}

size_t IMemoryAllocator::
guarantedAlignment(MemoryAllocationArgs)
{
  return guarantedAlignment();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
