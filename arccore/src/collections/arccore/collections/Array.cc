// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Array.cc                                                    (C) 2000-2023 */
/*                                                                           */
/* Vecteur de données 1D.                                                    */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/FatalErrorException.h"
#include "arccore/base/TraceInfo.h"

#include "arccore/collections/Array.h"
#include "arccore/collections/IMemoryAllocator.h"

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
  explicit BadAllocException(std::string str) : m_message(std::move(str)){}
  const char* what() const ARCCORE_NOEXCEPT override
  {
    return m_message.c_str();
  }
 public:
  std::string m_message;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IMemoryAllocator* ArrayMetaData::
_defaultAllocator()
{
  return &DefaultMemoryAllocator::shared_null_instance;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArrayMetaData::
_checkAllocator() const
{
  if (!allocation_options.m_allocator)
    throw BadAllocException("Null allocator");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MemoryAllocationArgs ArrayMetaData::
_getAllocationArgs() const
{
  return allocation_options.allocationArgs();
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

MemoryAllocationArgs MemoryAllocationOptions::
allocationArgs() const
{
  MemoryAllocationArgs x;
  x.setMemoryLocationHint(m_memory_location_hint);
  return x;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AbstractArrayBase::
setMemoryAllocationHint(eMemoryLocationHint new_hint)
{
  MemoryAllocationArgs old_args = m_md->_getAllocationArgs();
  m_md->allocation_options.setMemoryLocationHint(new_hint);
  MemoryAllocationArgs new_args = m_md->_getAllocationArgs();
  m_md->_allocator()->notifyMemoryArgsChanged(old_args,new_args,AllocatedMemoryInfo{});
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ArrayMetaData::MemoryPointer ArrayMetaData::
_allocate(Int64 new_capacity,Int64 sizeof_true_type)
{
  _checkAllocator();
  MemoryAllocationArgs alloc_args = _getAllocationArgs();
  IMemoryAllocator* a = _allocator();
  size_t s_new_capacity = (size_t)new_capacity;
  s_new_capacity = a->adjustCapacity(s_new_capacity,sizeof_true_type,alloc_args);
  size_t s_sizeof_true_type = (size_t)sizeof_true_type;
  size_t elem_size = s_new_capacity * s_sizeof_true_type;
  MemoryPointer p = a->allocate(elem_size,alloc_args);
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
  MemoryAllocationArgs alloc_args = _getAllocationArgs();
  IMemoryAllocator* a = _allocator();

  size_t s_new_capacity = (size_t)new_capacity;
  s_new_capacity = a->adjustCapacity(s_new_capacity,sizeof_true_type,alloc_args);
  size_t s_sizeof_true_type = (size_t)sizeof_true_type;
  size_t elem_size = s_new_capacity * s_sizeof_true_type;
  
  MemoryPointer p = nullptr;
  {
    const bool use_realloc = a->hasRealloc(alloc_args);
    // Lorsqu'on voudra implémenter un realloc avec alignement, il faut passer
    // par use_realloc = false car sous Linux il n'existe pas de méthode realloc
    // garantissant l'alignement (alors que sous Win32 si :) ).
    // use_realloc = false;
    if (use_realloc) {
      p = a->reallocate(current, elem_size, alloc_args);
    }
    else {
      p = a->allocate(elem_size, alloc_args);
      //GG: TODO: regarder si 'current' peut être nul (a priori je ne pense pas...)
      if (p && current) {
        size_t current_size = this->size * s_sizeof_true_type;
        ::memcpy(p, current, current_size);
        a->deallocate(current, alloc_args);
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
  if (_allocator()){
    MemoryAllocationArgs alloc_args = _getAllocationArgs();
    _allocator()->deallocate(current,alloc_args);
  }
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

void ArrayMetaData::
throwUnsupportedSpecificAllocator()
{
  throw BadAllocException("Changing allocator is only supported for UniqueArray");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
