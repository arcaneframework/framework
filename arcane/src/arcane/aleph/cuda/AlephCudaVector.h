// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* cnc_vector.h                                                (C) 2000-2012 */
/*                                                                           */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#ifndef _CNC_INTERFACE_VECTOR_H_
#define _CNC_INTERFACE_VECTOR_H_


ARCANE_BEGIN_NAMESPACE
  

template <class T> class CNC_Vector{
public:
  typedef CNC_Vector<T> thisclass ;

  CNC_Vector(unsigned int size = 0, unsigned int alignment = 1);

  ~CNC_Vector();

//---------------------------------------------------------------------------//

    /** does not preserve previous values stored in the array */
  void allocate(unsigned int size, unsigned int alignment = 1);

//---------------------------------------------------------------------------//

  void set_all(const T& value) ;

//---------------------------------------------------------------------------//

  T& operator()(unsigned int i) ;
//---------------------------------------------------------------------------//

  const T& operator()(unsigned int i) const ;

//---------------------------------------------------------------------------//

  T& operator[](unsigned int index) ;
//---------------------------------------------------------------------------//

  const T& operator[](unsigned int index) const;

//---------------------------------------------------------------------------//

  T& from_linear_index(unsigned int index);

//---------------------------------------------------------------------------//

  const T& from_linear_index(unsigned int index) const;
//---------------------------------------------------------------------------//

  unsigned int size() const;

//---------------------------------------------------------------------------//
	
  unsigned int alignment() const;

//---------------------------------------------------------------------------//

  void clear();

//---------------------------------------------------------------------------//

    /** low-level access, for experts only. */
  const T* data() const ;

//---------------------------------------------------------------------------//

    /** low-level access, for experts only. */
  T* data();

//---------------------------------------------------------------------------//

  unsigned int mem_usage() const ;
//---------------------------------------------------------------------------//

  void print () const;

//---------------------------------------------------------------------------//

protected:
    T* data_ ;
    unsigned int size_ ;
    char* base_mem_ ;
    unsigned int alignment_ ;

//---------------------------------------------------------------------------//

  void deallocate() ;

//---------------------------------------------------------------------------//

private:
    CNC_Vector(const thisclass& rhs) ;
    thisclass& operator=(const thisclass& rhs) ;
} ;


ARCANE_END_NAMESPACE

#endif

