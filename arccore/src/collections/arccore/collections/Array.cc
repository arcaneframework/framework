// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2020 IFPEN-CEA
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Array.cc                                                    (C) 2000-2018 */
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

ArrayImplBase ArrayImplBase::shared_null_instance = ArrayImplBase();
ArrayImplBase* ArrayImplBase::shared_null = &ArrayImplBase::shared_null_instance;

ArrayMetaData ArrayMetaData::shared_null_instance = ArrayMetaData();
ArrayMetaData* ArrayMetaData::shared_null = &ArrayMetaData::shared_null_instance;

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

ArrayImplBase* ArrayImplBase::
allocate(Int64 sizeof_true_impl,Int64 new_capacity,
         Int64 sizeof_true_type,ArrayImplBase* init,ArrayMetaData* init_meta_data)
{
  return allocate(sizeof_true_impl,new_capacity,sizeof_true_type,init,init_meta_data,nullptr);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ArrayImplBase* ArrayImplBase::
allocate(Int64 sizeof_true_impl,Int64 new_capacity,Int64 sizeof_true_type,
         ArrayImplBase* init,ArrayMetaData* init_meta_data,
         IMemoryAllocator* allocator)
{
  if (!allocator)
    allocator = init_meta_data->allocator;

  size_t s_sizeof_true_impl = (size_t)sizeof_true_impl;
  size_t s_new_capacity = (size_t)new_capacity;
  s_new_capacity = allocator->adjustCapacity(s_new_capacity,sizeof_true_type);
  size_t s_sizeof_true_type = (size_t)sizeof_true_type;
  size_t elem_size = s_sizeof_true_impl + (s_new_capacity - 1) * s_sizeof_true_type;
  ArrayImplBase* p = (ArrayImplBase*)(allocator->allocate(elem_size));
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

  *p = *init;

  init_meta_data->capacity = (Int64)s_new_capacity;
  return p;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ArrayImplBase* ArrayImplBase::
reallocate(Int64 sizeof_true_impl,Int64 new_capacity,Int64 sizeof_true_type,
           ArrayImplBase* current,ArrayMetaData* current_meta_data)
{
  IMemoryAllocator* allocator = current_meta_data->allocator;
  size_t s_sizeof_true_impl = (size_t)sizeof_true_impl;
  size_t s_new_capacity = (size_t)new_capacity;
  s_new_capacity = allocator->adjustCapacity(s_new_capacity,sizeof_true_type);
  size_t s_sizeof_true_type = (size_t)sizeof_true_type;
  size_t elem_size = s_sizeof_true_impl + (s_new_capacity - 1) * s_sizeof_true_type;
  
  ArrayImplBase* p = 0;
  {
    const bool use_realloc = allocator->hasRealloc();
    // Lorsqu'on voudra implémenter un realloc avec alignement, il faut passer
    // par use_realloc = false car sous Linux il n'existe pas de méthode realloc
    // garantissant l'alignmenent (alors que sous Win32 si :) ).
    // use_realloc = false;
    if (use_realloc){
      p = (ArrayImplBase*)(allocator->reallocate(current,elem_size));
    }
    else{
      p = (ArrayImplBase*)(allocator->allocate(elem_size));
      //GG: TODO: regarder si 'current' peut etre nul (a priori je ne pense pas...)
      if (p && current){
        size_t current_size = s_sizeof_true_impl + (current_meta_data->size - 1) * s_sizeof_true_type;
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
  current_meta_data->capacity = (Int64)s_new_capacity;
  return p;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArrayImplBase::
deallocate(ArrayImplBase* current,ArrayMetaData* current_meta_data)
{
  current_meta_data->allocator->deallocate(current);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArrayImplBase::
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
throwBadSharedNull()
{
  throw BadAllocException("corrupted ArrayMetaData::shared_null");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
