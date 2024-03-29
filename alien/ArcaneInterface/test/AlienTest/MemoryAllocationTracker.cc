﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#include "MemoryAllocationTracker.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include <stdio.h>
#include <malloc.h>
#include <map>
#include <algorithm>

static bool need_analysis = false;
size_t memory_peak;
size_t total_allocation = 0;
size_t allocation_count = 0;
size_t reallocation_count = 0;
size_t deallocation_count = 0;
std::map<void*, size_t> allocation_database;

/*---------------------------------------------------------------------------*/
#if defined(TEST_HAS_MALLOC_HOOKS)
/*---------------------------------------------------------------------------*/

/* Prototypes de nos routines */
static void my_local_init_hook(void);
static void* my_local_malloc_hook(size_t, const void*);
static void my_local_free_hook(void*, const void*);
static void* my_local_realloc_hook(void* ptr, size_t size, const void* caller);

/* Variables pour sauver la routine originale */
static void* (*old_malloc_hook)(size_t, const void*);
static void (*old_free_hook)(void*, const void*);
static void* (*old_realloc_hook)(void* ptr, size_t size, const void* caller);

// /* Ecrasement de la routine d'initialisaiton glibg */
// void (*__malloc_initialize_hook) (void) = my_local_init_hook;

/*---------------------------------------------------------------------------*/

void
pushHooks()
{
  __malloc_hook = old_malloc_hook;
  __free_hook = old_free_hook;
  __realloc_hook = old_realloc_hook;
}

/*---------------------------------------------------------------------------*/

void
popHooks()
{
  __malloc_hook = my_local_malloc_hook;
  __free_hook = my_local_free_hook;
  __realloc_hook = my_local_realloc_hook;
}

/*---------------------------------------------------------------------------*/

static void
my_local_init_hook(void)
{
  old_malloc_hook = __malloc_hook;
  __malloc_hook = my_local_malloc_hook;
  old_free_hook = __free_hook;
  __free_hook = my_local_free_hook;
  old_realloc_hook = __realloc_hook;
  __realloc_hook = my_local_realloc_hook;
  need_analysis = false;
  total_allocation = 0;
  allocation_count = reallocation_count = deallocation_count = 0;
}

/*---------------------------------------------------------------------------*/

static void
my_local_restore_hook(void)
{
  __malloc_hook = my_local_malloc_hook;
  __free_hook = my_local_free_hook;
  __realloc_hook = my_local_realloc_hook;
}

/*---------------------------------------------------------------------------*/

static void*
my_local_malloc_hook(size_t size, const void* caller)
{
  void* result;

  /* Replacer la routine originale */
  pushHooks();

  /* Appel de la routine originale) */
  result = malloc(size);

  //       /* 'printf' peut appeler 'malloc'... a proteger. */
  //       printf ("malloc0(%u) called from %p returns %p\n",(unsigned int) size, caller,
  //       result);

  if (need_analysis) {
    //       /* 'printf' peut appeler 'malloc'... a proteger. */
    //       printf ("malloc(%u) called from %p returns %p\n",(unsigned int) size, caller,
    //       result);

    allocation_database[result] = size;
    total_allocation += size;
    allocation_count++;
    memory_peak = std::max(memory_peak, total_allocation);
  }

  popHooks();

  return result;
}

/*---------------------------------------------------------------------------*/

static void
my_local_free_hook(void* ptr, const void* caller)
{
  /* Replacer la routine originale */
  pushHooks();

  /* Appel de la routine originale) */
  free(ptr);

  /* Sauver la routine originale */
  old_free_hook = __free_hook;

  if (need_analysis) {
    //     /* 'printf' peut appeler 'malloc'... a proteger. */
    //     printf ("free(%u) called from %p\n",(unsigned int)0, caller);

    std::map<void*, size_t>::iterator finder = allocation_database.find(ptr);
    if (finder != allocation_database.end()) {
      const size_t size = finder->second;
      allocation_database.erase(finder);
      total_allocation -= size;
      deallocation_count++;
    }
  }

  /* Replacer notre routine */
  popHooks();
}

/*---------------------------------------------------------------------------*/

void*
my_local_realloc_hook(void* ptr, size_t size, const void* caller)
{
  void* result;

  /* Replacer la routine originale */
  pushHooks();

  /* Appel de la routine originale) */
  result = realloc(ptr, size);

  if (need_analysis) {
    //     /* 'printf' peut appeler 'malloc'... a proteger. */
    //     printf ("realloc(%p,%u) called from %p returns %p\n",ptr,size, caller,result);

    std::map<void*, size_t>::iterator finder = allocation_database.find(ptr);
    if (finder != allocation_database.end()) {
      const size_t old_size = finder->second;
      allocation_database.erase(finder);
      total_allocation -= old_size;
      allocation_database[result] = size;
      total_allocation += size;
      reallocation_count++;
      memory_peak = std::max(memory_peak, total_allocation);
    }
  }

  /* Replacer notre routine */
  popHooks();

  return result;
}

/*---------------------------------------------------------------------------*/
#else // TEST_HAS_MALLOC_HOOKS
/*---------------------------------------------------------------------------*/

static void
my_local_init_hook(void)
{
  need_analysis = false;
  total_allocation = 0;
  allocation_count = reallocation_count = deallocation_count = 0;
}

/*---------------------------------------------------------------------------*/

static void
my_local_restore_hook(void)
{
}

/*---------------------------------------------------------------------------*/

void
pushHooks()
{
}

/*---------------------------------------------------------------------------*/

void
popHooks()
{
}

/*---------------------------------------------------------------------------*/
#endif // TEST_HAS_MALLOC_HOOKS
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MemoryAllocationTracker::MemoryAllocationTracker()
{
  my_local_init_hook();
}

/*---------------------------------------------------------------------------*/

MemoryAllocationTracker::~MemoryAllocationTracker()
{
  my_local_restore_hook();
}

/*---------------------------------------------------------------------------*/

void
MemoryAllocationTracker::beginCollect()
{
  need_analysis = true;
  popHooks();
}

/*---------------------------------------------------------------------------*/

void
MemoryAllocationTracker::endCollect()
{
  need_analysis = false;
  pushHooks();
}

/*---------------------------------------------------------------------------*/

size_t
MemoryAllocationTracker::getPeakAllocation() const
{
  return memory_peak;
}

/*---------------------------------------------------------------------------*/

size_t
MemoryAllocationTracker::getTotalAllocation() const
{
  return total_allocation;
}

/*---------------------------------------------------------------------------*/

void
MemoryAllocationTracker::resetPeakAllocation()
{
  memory_peak = total_allocation;
}

/*---------------------------------------------------------------------------*/

size_t
MemoryAllocationTracker::getAllocationCount() const
{
  return allocation_count;
}

/*---------------------------------------------------------------------------*/

size_t
MemoryAllocationTracker::getReallocationCount() const
{
  return reallocation_count;
}

/*---------------------------------------------------------------------------*/

size_t
MemoryAllocationTracker::getDeallocationCount() const
{
  return deallocation_count;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
