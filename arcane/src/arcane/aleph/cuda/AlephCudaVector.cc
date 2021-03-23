// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#include "arcane/aleph/AlephArcane.h"
#include "arcane/aleph/cuda/AlephCuda.h"
 

//---------------------------------------------------------------------------//


ARCANE_BEGIN_NAMESPACE

template <class T> CNC_Vector<T>::CNC_Vector(unsigned int size, unsigned int alignment) {
  data_ = NULL ;
  base_mem_ = NULL ;
  size_ = 0 ;
  allocate(size, alignment) ;
} 

template <class T> CNC_Vector<T>::~CNC_Vector() { deallocate() ; }

//---------------------------------------------------------------------------//

/** does not preserve previous values stored in the array */
template <class T> void CNC_Vector<T>::allocate(unsigned int size, unsigned int alignment) {
  deallocate() ;
  if(size != 0) {
    base_mem_ = (char*)malloc(size * sizeof(T) + alignment -1) ;
    char* p = base_mem_ ;
    // GMY 20090825 original: while(unsigned __int64(p) % alignment) {  p++ ; }
    while (((unsigned long long) p) % alignment) { ++p; }
    data_ = (T*)p ;
    for(unsigned int i=0; i<size; i++) {
      // Direct call to the constructor, see dlist.h for more explanations.
      new(&data_[i])T() ;                    
    }
  }
  size_ = size ;
}

//---------------------------------------------------------------------------//

template <class T> void CNC_Vector<T>::set_all(const T& value) {
  for(unsigned int i=0; i<size_; i++) {
    data_[i] = value ;
  }
}

//---------------------------------------------------------------------------//

template <class T> T& CNC_Vector<T>::operator()(unsigned int i) {
  return data_[i] ;
}

//---------------------------------------------------------------------------//

template <class T> const T& CNC_Vector<T>::operator()(unsigned int i) const {
  return data_[i] ;
}

//---------------------------------------------------------------------------//

template <class T> T& CNC_Vector<T>::operator[](unsigned int index) {
  return data_[index] ;
}

//---------------------------------------------------------------------------//

template <class T> const T& CNC_Vector<T>::operator[](unsigned int index) const {
  return data_[index] ;
}

//---------------------------------------------------------------------------//

template <class T> T& CNC_Vector<T>::from_linear_index(unsigned int index) {
  return data_[index] ;
}

//---------------------------------------------------------------------------//

template <class T> const T& CNC_Vector<T>::from_linear_index(unsigned int index) const {
  return data_[index] ;
}

//---------------------------------------------------------------------------//

template <class T> unsigned int CNC_Vector<T>::size() const { return size_ ; }

//---------------------------------------------------------------------------//
	
template <class T> unsigned int CNC_Vector<T>::alignment() const { return alignment_ ; }

//---------------------------------------------------------------------------//

template <class T> void CNC_Vector<T>::clear() { allocate(0) ; }

//---------------------------------------------------------------------------//

/** low-level access, for experts only. */
template <class T> const T* CNC_Vector<T>::data() const { return data_ ; }

//---------------------------------------------------------------------------//

/** low-level access, for experts only. */
template <class T> T* CNC_Vector<T>::data() { return data_ ; }

//---------------------------------------------------------------------------//

template <class T> unsigned int CNC_Vector<T>::mem_usage() const {
  return size_ * sizeof(T) + sizeof(thisclass) ;
}

//---------------------------------------------------------------------------//

template <class T> void CNC_Vector<T>::print () const {
  for(unsigned int index=0; index<size_; index++){
    //printf("\t[%d]=%ld\n", index, data_[index]);
  }
}


//---------------------------------------------------------------------------//

template <class T> void CNC_Vector<T>::deallocate() {
  if(size_ != 0) {
    for(unsigned int i=0; i<size_; i++) {
      // direct call to the destructor
      data_[i].~T() ;
    }
    free(base_mem_) ;
    data_ = NULL ;
    base_mem_ = NULL ;
    size_ = 0 ;
  }
}


template class CNC_Vector<double>;
template class CNC_Vector<long>;
template class CNC_Vector<std::set<unsigned int> >;


ARCANE_END_NAMESPACE
