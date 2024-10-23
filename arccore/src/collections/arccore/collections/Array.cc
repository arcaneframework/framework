﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Array.cc                                                    (C) 2000-2024 */
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
  return &DefaultMemoryAllocator3::shared_null_instance;
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

MemoryAllocationArgs ArrayMetaData::
_getAllocationArgs(RunQueue* queue) const
{
  return allocation_options.allocationArgs(queue);
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

void ArrayMetaData::
_setMemoryLocationHint(eMemoryLocationHint new_hint,void* ptr,Int64 sizeof_true_type)
{
  MemoryAllocationArgs old_args = _getAllocationArgs();
  allocation_options.setMemoryLocationHint(new_hint);
  MemoryAllocationArgs new_args = _getAllocationArgs();
  AllocatedMemoryInfo mem_info(ptr,size*sizeof_true_type,capacity*sizeof_true_type);
  _allocator()->notifyMemoryArgsChanged(old_args,new_args,mem_info);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArrayMetaData::
_copyFromMemory(MemoryPointer destination, ConstMemoryPointer source, Int64 sizeof_true_type, RunQueue* queue)
{
  MemoryAllocationArgs args = _getAllocationArgs(queue);
  Int64 full_size = size * sizeof_true_type;
  AllocatedMemoryInfo source_info(const_cast<void*>(source), full_size, full_size);
  AllocatedMemoryInfo destination_info(destination, full_size, full_size);
  _allocator()->copyMemory(args, destination_info, source_info);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ArrayMetaData::MemoryPointer ArrayMetaData::
_allocate(Int64 new_capacity, Int64 sizeof_true_type, RunQueue* queue)
{
  _checkAllocator();
  MemoryAllocationArgs alloc_args = _getAllocationArgs(queue);
  IMemoryAllocator* a = _allocator();
  new_capacity = a->adjustedCapacity(alloc_args, new_capacity, sizeof_true_type);
  Int64 elem_size = new_capacity * sizeof_true_type;
  MemoryPointer p = a->allocate(alloc_args, elem_size).baseAddress();
#ifdef ARCCORE_DEBUG_ARRAY
  std::cout << "ArrayImplBase::ALLOCATE: elemsize=" << elem_size
            << " typesize=" << sizeof_true_type
            << " size=" << new_capacity << " datasize=" << sizeof_true_impl
            << " p=" << p << '\n';
#endif
  if (!p) {
    std::ostringstream ostr;
    ostr << " Bad ArrayImplBase::allocate() size=" << elem_size << " capacity=" << new_capacity
         << " sizeof_true_type=" << sizeof_true_type << '\n';
    throw BadAllocException(ostr.str());
  }

  this->capacity = new_capacity;

  return p;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ArrayMetaData::MemoryPointer ArrayMetaData::
_reallocate(const AllocatedMemoryInfo& current_info, Int64 new_capacity, Int64 sizeof_true_type, RunQueue* queue)
{
  _checkAllocator();
  MemoryPointer current = current_info.baseAddress();
  MemoryAllocationArgs alloc_args = _getAllocationArgs(queue);
  IMemoryAllocator* a = _allocator();
  new_capacity = a->adjustedCapacity(alloc_args, new_capacity, sizeof_true_type);
  size_t elem_size = new_capacity * sizeof_true_type;
  MemoryPointer p = nullptr;
  {
    const bool use_realloc = a->hasRealloc(alloc_args);
    // Lorsqu'on voudra implémenter un realloc avec alignement, il faut passer
    // par use_realloc = false car sous Linux il n'existe pas de méthode realloc
    // garantissant l'alignement (alors que sous Win32 si :) ).
    // use_realloc = false;
    if (use_realloc) {
      p = a->reallocate(alloc_args, current_info, elem_size).baseAddress();
    }
    else {
      AllocatedMemoryInfo new_alloc_info = a->allocate(alloc_args, elem_size);
      p = new_alloc_info.baseAddress();
      //GG: TODO: regarder si 'current' peut être nul (a priori je ne pense pas...)
      if (p && current) {
        a->copyMemory(alloc_args, new_alloc_info, current_info);
        a->deallocate(alloc_args, current_info);
      }
    }
  }
#ifdef ARCCORE_DEBUG_ARRAY
  std::cout << " ArrayImplBase::REALLOCATE: elemsize=" << elem_size
            << " typesize=" << sizeof_true_type
            << " size=" << new_capacity << " datasize=" << sizeof_true_impl
            << " ptr=" << current << " new_p=" << p << '\n';
#endif
  if (!p) {
    std::ostringstream ostr;
    ostr << " Bad ArrayImplBase::reallocate() size=" << elem_size
         << " capacity=" << new_capacity
         << " sizeof_true_type=" << sizeof_true_type
         << " old_ptr=" << current << '\n';
    throw BadAllocException(ostr.str());
  }
  this->capacity = new_capacity;
  return p;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArrayMetaData::
_deallocate(const AllocatedMemoryInfo& mem_info, RunQueue* queue) noexcept
{
  if (_allocator()) {
    MemoryAllocationArgs alloc_args = _getAllocationArgs(queue);
    _allocator()->deallocate(alloc_args, mem_info);
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

void ArrayMetaData::
throwInvalidMetaDataForSharedArray()
{
  throw BadAllocException("MetaData for SharedArray are not allocated");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AbstractArrayBase::
setDebugName(const String& name)
{
  m_md->allocation_options.setArrayName(name);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String AbstractArrayBase::
debugName() const
{
  return m_md->allocation_options.arrayName();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
