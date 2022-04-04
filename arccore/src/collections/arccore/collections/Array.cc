﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Array.cc                                                    (C) 2000-2021 */
/*                                                                           */
/* Vecteur de données 1D.                                                    */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ArrayView.h"
#include "arccore/base/FatalErrorException.h"
#include "arccore/base/TraceInfo.h"

#include "arccore/collections/Array.h"

#include <algorithm>
#include <iostream>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class BadAllocException
: public std::bad_alloc
{
 public:
  BadAllocException(const std::string& str) : m_message(str){}
  virtual const char* what() const ARCCORE_NOEXCEPT
  {
    return m_message.c_str();
  }
 public:
  std::string m_message;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArrayMetaData::
_checkAllocator() const
{
  if (!allocator)
    throw BadAllocException("Null allocator");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*
 * TODO: pour les allocations, faire en sorte que le
 * début du tableau soit aligné sur 16 octets dans tous les cas.
 * Attention dans ce cas a bien traiter les problèmes avec realloc().
 * TODO: pour les grosses allocations qui correspondantes la
 * plupart du temps à des variables, ajouter un random sur le début
 * du tableau pour éviter les conflits de bancs mémoire ou de cache
 */

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ArrayMetaData::MemoryPointer ArrayMetaData::
_allocate(Int64 new_capacity,Int64 sizeof_true_type)
{
  _checkAllocator();

  size_t s_new_capacity = (size_t)new_capacity;
  s_new_capacity = allocator->adjustCapacity(s_new_capacity,sizeof_true_type);
  size_t s_sizeof_true_type = (size_t)sizeof_true_type;
  size_t elem_size = s_new_capacity * s_sizeof_true_type;
  MemoryPointer p = allocator->allocate(elem_size);
#ifdef ARCCORE_DEBUG_ARRAY
  std::cout << "ArrayImplBase::ALLOCATE: elemsize=" << elem_size
            << " typesize=" << sizeof_true_type
            << " size=" << new_capacity << " datasize=" << sizeof_true_impl
            << " p=" << p << '\n';
#endif
  if (!p){
    std::ostringstream ostr;
    ostr << " Bad ArrayImplBase::allocate() size=" << elem_size << " capacity=" << new_capacity
         << " sizeof_true_type=" << sizeof_true_type << '\n';
    throw BadAllocException(ostr.str());
  }

  this->capacity = (Int64)s_new_capacity;

  return p;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ArrayMetaData::MemoryPointer ArrayMetaData::
_reallocate(Int64 new_capacity,Int64 sizeof_true_type,MemoryPointer current)
{
  _checkAllocator();

  size_t s_new_capacity = (size_t)new_capacity;
  s_new_capacity = allocator->adjustCapacity(s_new_capacity,sizeof_true_type);
  size_t s_sizeof_true_type = (size_t)sizeof_true_type;
  size_t elem_size = s_new_capacity * s_sizeof_true_type;
  
  MemoryPointer p = nullptr;
  {
    const bool use_realloc = allocator->hasRealloc();
    // Lorsqu'on voudra implémenter un realloc avec alignement, il faut passer
    // par use_realloc = false car sous Linux il n'existe pas de méthode realloc
    // garantissant l'alignmenent (alors que sous Win32 si :) ).
    // use_realloc = false;
    if (use_realloc){
      p = allocator->reallocate(current,elem_size);
    }
    else{
      p = allocator->allocate(elem_size);
      //GG: TODO: regarder si 'current' peut etre nul (a priori je ne pense pas...)
      if (p && current){
        size_t current_size = this->size * s_sizeof_true_type;
        ::memcpy(p,current,current_size);
        allocator->deallocate(current);
      }
    }
  }
#ifdef ARCCORE_DEBUG_ARRAY
  std::cout << " ArrayImplBase::REALLOCATE: elemsize=" << elem_size
            << " typesize=" << sizeof_true_type
            << " size=" << new_capacity << " datasize=" << sizeof_true_impl
            << " ptr=" << current << " new_p=" << p << '\n';
#endif
  if (!p){
    std::ostringstream ostr;
    ostr << " Bad ArrayImplBase::reallocate() size=" << elem_size
         << " capacity=" << new_capacity
         << " sizeof_true_type=" << sizeof_true_type
         << " old_ptr=" << current << '\n';
    throw BadAllocException(ostr.str());
  }
  this->capacity = (Int64)s_new_capacity;
  return p;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArrayMetaData::
_deallocate(MemoryPointer current) noexcept
{
  if (allocator)
    allocator->deallocate(current);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArrayMetaData::
overlapError(const void* begin1,Int64 size1,
             const void* begin2,Int64 size2)
{
  ARCCORE_UNUSED(begin1);
  ARCCORE_UNUSED(begin2);
  ARCCORE_UNUSED(size1);
  ARCCORE_UNUSED(size2);
  ARCCORE_FATAL("source and destinations overlaps");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArrayMetaData::
throwNullExpected()
{
  throw BadAllocException("ArrayMetaData should be null");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArrayMetaData::
throwNotNullExpected()
{
  throw BadAllocException("ArrayMetaData should be not be null");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
