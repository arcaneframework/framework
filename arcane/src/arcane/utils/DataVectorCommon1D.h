// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DataVectorCommon1D.h                                        (C) 2000-2006 */
/*                                                                           */
/* Vecteur de données 1D.                                                    */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_DATAVECTORCOMMON1D_H
#define ARCANE_DATAVECTORCOMMON1D_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArrayView.h"
#include "arcane/utils/IAllocator.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Vecteur de données 1D d'un type \a T.
 *
 * Cette classe est réservée à la partie interne de Arcane. Pour utiliser
 * un vecteur de données, utiliser la classe DataVector1D.
 *
 * C'est à la classe dérivée de libérer la mémoire allouée ainsi que
 * l'allocateur (voir _destroy()).
 *
 */
template<typename T>
class DataVectorCommon1D
: public ArrayView<T>
{
 public:

  //! Type de la classe de base
  typedef ArrayView<T> ArrayType;

  //! Type de l'allocateur
  typedef IAllocatorT<T> AllocatorType;

 private:
  
  typedef DataVectorCommon1D<T> ThatClassType;

 public:

  DataVectorCommon1D(AllocatorType* allocator)
  : ArrayType(), m_capacity(0), m_allocator(allocator)
    {
      ARCANE_CHECK_PTR(allocator);
    }

  DataVectorCommon1D(Integer nb_element,AllocatorType* allocator)
  : ArrayType(), m_capacity(nb_element), m_allocator(allocator)
    {
      ARCANE_CHECK_PTR(allocator);
      this->_setPtr(_allocate(nb_element));
      this->_setSize(nb_element);
    }

  DataVectorCommon1D(ThatClassType& from)
  : ArrayType(from), m_capacity(from.m_capacity), m_allocator(from.m_allocator)
    {
      ARCANE_CHECK_PTR(m_allocator);
    }

  const ThatClassType& operator=(ThatClassType& from)
    {
      if (&from!=this){
        this->_setArray(from.data(),from.size());
        m_capacity = from.m_capacity;
        m_allocator = from.m_allocator;
      }
      return (*this);
    }

  virtual ~DataVectorCommon1D()
    {
    }

 protected:

  void _add(const T& element)
    {
      Integer s = this->size();
      if (s>=m_capacity)
        _reserve((m_capacity==0) ? 4 : m_capacity*2);
      this->_ptr()[s] = element;
      this->_setSize(s+1);
    }

  void _resize(Integer new_size)
    {
      if (new_size>m_capacity)
        _reserve(new_size);
      this->_setSize(new_size);
    }

  void _reserve(Integer new_capacity)
    {
      T* new_ptr = _allocate(new_capacity);
      T* current_ptr = this->_ptr();
      for( Integer i=0, is=m_capacity; i<is; ++i )
        new_ptr[i] = current_ptr[i];
      _deallocate(current_ptr,m_capacity);
      m_capacity = new_capacity;
      this->_setPtr(new_ptr);
    }

  void _destroy()
    {
      _deallocate(this->_ptr(),m_capacity);
      m_allocator->destroy();
    }

 private:

  Integer m_capacity;
  AllocatorType* m_allocator;

 private:

  void _deallocate(const T* ptr,Integer capacity)
    {
      m_allocator->deallocate(ptr,capacity);
    }

  T* _allocate(Integer capacity)
    {
      return m_allocator->allocate(capacity);
    }

};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

