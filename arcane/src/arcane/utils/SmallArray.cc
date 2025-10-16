// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SmallArray.cc                                               (C) 2000-2025 */
/*                                                                           */
/* Tableau 1D de données avec buffer pré-alloué.                             */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/SmallArray.h"
#include "arcane/utils/FatalErrorException.h"

#include <cstdlib>
#include <cstring>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Impl
{

namespace
{
  //! A mettre à true si on souhaite activer les traces d'allocation
  const bool is_verbose = false;
} // namespace

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void* StackMemoryAllocator::
_allocateMemory(Int64 new_size)
{
  if (is_verbose)
    std::cout << "ALLOCATE: use malloc s=" << new_size << "\n";
  void* ptr = std::malloc(new_size);
  if (!ptr)
    ARCANE_FATAL("Can not allocated memory for size '{0}'", new_size);
  return ptr;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AllocatedMemoryInfo StackMemoryAllocator::
reallocate(MemoryAllocationArgs, AllocatedMemoryInfo current_ptr_info, Int64 new_size)
{
  void* current_ptr = current_ptr_info.baseAddress();
  if (current_ptr != m_preallocated_buffer) {
    if (new_size < m_preallocated_size) {
      // On passe d'un pointeur alloué vers notre buffer interne.
      // Il faut recopier les valeurs. On ne connait pas exactement
      // la taille de 'current_ptr' mais on est certain qu'elle est
      // supérieure à 'm_preallocated_size' donc à 'new_size'.
      // On ne recopie donc que ces valeurs là
      if (is_verbose)
        std::cout << "REALLOCATE: use own buffer from realloc s=" << new_size << "\n";
      std::memcpy(m_preallocated_buffer, current_ptr, new_size);
      std::free(current_ptr);
      return { m_preallocated_buffer, new_size };
    }
    if (is_verbose)
      std::cout << "REALLOCATE: use realloc s=" << new_size << "\n";
    return { std::realloc(current_ptr, new_size), new_size };
  }

  if (new_size <= m_preallocated_size) {
    if (is_verbose)
      std::cout << "REALLOCATE: use buffer because small size s=" << new_size << "\n";
    return { m_preallocated_buffer, new_size };
  }

  // Il faut allouer et recopier depuis le buffer pré-alloué.
  if (is_verbose)
    std::cout << "REALLOCATE: use malloc and copy s=" << new_size << "\n";
  void* new_ptr = std::malloc(new_size);
  std::memcpy(new_ptr, m_preallocated_buffer, m_preallocated_size);
  return { new_ptr, new_size };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void StackMemoryAllocator::
_freeMemory(void* ptr)
{
  std::free(ptr);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
