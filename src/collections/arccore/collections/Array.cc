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

ArrayImplBase* DefaultArrayAllocator::
allocate(Int64 sizeof_true_impl,Int64 new_capacity,
         Int64 sizeof_true_type,ArrayImplBase* init)
{
  Int64 elem_size = sizeof_true_impl + (new_capacity - 1) * sizeof_true_type;
  /*std::cout << " ALLOCATE: elemsize=" << elem_size
            << " typesize=" << sizeofTypedData
            << " size=" << size << " datasize=" << sizeofT << '\n';*/
  ArrayImplBase* p = (ArrayImplBase*)::malloc(elem_size);
  //std::cout << " RETURN p=" << p << '\n';
  if (!p){
    std::ostringstream ostr;
    ostr << " Bad DefaultArrayAllocator::allocate() size=" << elem_size << " capacity=" << new_capacity
         << " sizeof_true_type=" << sizeof_true_type << '\n';
    throw BadAllocException(ostr.str());
  }
  Int64 s = (new_capacity>init->capacity) ? init->capacity : new_capacity;
  ::memcpy(p, init,sizeof_true_impl + (s - 1) * sizeof_true_type);
  return p;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DefaultArrayAllocator::
deallocate(ArrayImplBase* ptr)
{
  ::free(ptr);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ArrayImplBase* DefaultArrayAllocator::
reallocate(Int64 sizeof_true_impl,Int64 new_capacity,
           Int64 sizeof_true_type,ArrayImplBase* ptr)
{
  Int64 elem_size = sizeof_true_impl + (new_capacity - 1) * sizeof_true_type;
  //Integer elem_size = sizeofTypedData + (size - 1) * sizeofT;
  std::cout << " REALLOCATE: elemsize=" << elem_size
            << " typesize=" << sizeof_true_type
            << " size=" << new_capacity << " datasize=" << sizeof_true_impl
            << " ptr=" << ptr << '\n';
  ArrayImplBase* p = (ArrayImplBase*)::realloc(ptr,elem_size);
  if (!p){
    std::ostringstream ostr;
    ostr << " Bad DefaultArrayAllocator::reallocate() size=" << elem_size << " capacity=" << new_capacity
         << " sizeof_true_type=" << sizeof_true_type
         << " old_ptr=" << ptr << '\n';
    throw BadAllocException(ostr.str());
  }
  //std::cout << " RETURN p=" << ((Int64)p%16) << '\n';
  return p;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 DefaultArrayAllocator::
computeCapacity(Int64 current,Int64 wanted)
{
  Int64 capacity = current;
  //std::cout << " REALLOC: want=" << wanted_size << " current_capacity=" << capacity << '\n';
  while (wanted>capacity)
    capacity = (capacity==0) ? 4 : (capacity + 1 + capacity / 2);
  //std::cout << " REALLOC: want=" << wanted_size << " new_capacity=" << capacity << '\n';
  return capacity;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ArrayImplBase* ArrayImplBase::
allocate(Int64 sizeof_true_impl,Int64 new_capacity,
         Int64 sizeof_true_type,ArrayImplBase* init)
{
  return allocate(sizeof_true_impl,new_capacity,sizeof_true_type,init,nullptr);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ArrayImplBase* ArrayImplBase::
allocate(Int64 sizeof_true_impl,Int64 new_capacity,
         Int64 sizeof_true_type,ArrayImplBase* init,IMemoryAllocator* allocator)
{
  if (!allocator)
    allocator = init->allocator;

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

  p->capacity = (Int64)s_new_capacity;
  return p;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ArrayImplBase* ArrayImplBase::
reallocate(Int64 sizeof_true_impl,Int64 new_capacity,Int64 sizeof_true_type,
           ArrayImplBase* current)
{
  IMemoryAllocator* allocator = current->allocator;
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
        size_t current_size = s_sizeof_true_impl + (current->size - 1) * s_sizeof_true_type;
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
  p->capacity = (Int64)s_new_capacity;
  return p;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArrayImplBase::
deallocate(ArrayImplBase* current)
{
  current->allocator->deallocate(current);
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
  throw FatalErrorException(A_FUNCINFO,"source and destinations overlaps");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArrayImplBase::
throwBadSharedNull()
{
  throw BadAllocException("corrupted ArrayImplBase::shared_null");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
