// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DefaultAllocator.h                                          (C) 2000-2018 */
/*                                                                           */
/* Allocateur par défaut.                                                    */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_DEFAULTALLOCATOR_H
#define ARCANE_UTILS_DEFAULTALLOCATOR_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/IAllocator.h"

#include <new>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Allocateur par défaut (utilise new[] et delete[])
 */
template<typename T>
class DefaultAllocatorT
: public IAllocatorT<T>
{
 public:

  DefaultAllocatorT() {}
  virtual ~DefaultAllocatorT() {}

 public:
  static DefaultAllocatorT<T>* globalInstance()
  {
    static DefaultAllocatorT<T> _global_instance;
    return &_global_instance;
  }

 public:

  virtual void destroy()
  {
    delete this;
  }

  virtual T* allocate(Integer new_capacity)
  {
    if (new_capacity<=0)
      throw std::bad_alloc();
    T* x = new T[(size_t)new_capacity];
    return x;
  }

  virtual void deallocate(const T* ptr,Integer capacity)
  {
	  ARCANE_UNUSED(capacity);
    delete[] ptr;
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

