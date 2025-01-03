// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* RawCopy.h                                                   (C) 2000-2024 */
/*                                                                           */
/* Structure de copie brute sans controle arithmétique.                      */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_RAWCOPY_H
#define ARCANE_CORE_RAWCOPY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename T> struct RawCopy { };

// Specify specialization for all types to remove all implicit implementation
// (when a new Arcane POD type will appear)

template<>
struct RawCopy<Byte> {
  typedef Byte T;
  inline static void copy(T & dst, const T & src) { dst = src; }
};

// Types without problem

template<>
struct RawCopy<Int16> {
  typedef Int16 T;
  inline static void copy(T & dst, const T & src) { dst = src; }
};

template<>
struct RawCopy<Int8> {
  typedef Int8 T;
  inline static void copy(T & dst, const T & src) { dst = src; }
};

template<>
struct RawCopy<Int32> {
  typedef Int32 T;
  inline static void copy(T & dst, const T & src) { dst = src; }
};

template<>
struct RawCopy<Int64> {
  typedef Int64 T;
  inline static void copy(T & dst, const T & src) { dst = src; }
};

template<>
struct RawCopy<String> {
  typedef String T;
  inline static void copy(T & dst, const T & src) { dst = src; }
};

// Types with problems

#ifdef ARCANE_RAWCOPY

#ifndef NO_USER_WARNING
#warning "Using RawCopy for compact Variables"
#endif /* NO_USER_WARNING */

template<>
struct RawCopy<Real> {
  typedef Real T;
  inline static void copy(T & dst, const T & src) { 
    reinterpret_cast<Int64&>(dst) = reinterpret_cast<const Int64&>(src); 
  }
};

template<>
struct RawCopy<Real2> {
  typedef Real2 T;
  inline static void copy(T & dst, const T & src) { 
    reinterpret_cast<Int64&>(dst.x) = reinterpret_cast<const Int64&>(src.x); 
    reinterpret_cast<Int64&>(dst.y) = reinterpret_cast<const Int64&>(src.y); 
  }
};

template<>
struct RawCopy<Real3> {
  typedef Real3 T;
  inline static void copy(T & dst, const T & src) {
    reinterpret_cast<Int64&>(dst.x) = reinterpret_cast<const Int64&>(src.x);
    reinterpret_cast<Int64&>(dst.y) = reinterpret_cast<const Int64&>(src.y);
    reinterpret_cast<Int64&>(dst.z) = reinterpret_cast<const Int64&>(src.z);
  }
};

template<>
struct RawCopy<Real2x2> {
  typedef Real2x2 T;
  inline static void copy(T & dst, const T & src) {
    reinterpret_cast<Int64&>(dst.x.x) = reinterpret_cast<const Int64&>(src.x.x); 
    reinterpret_cast<Int64&>(dst.x.y) = reinterpret_cast<const Int64&>(src.x.y); 
    reinterpret_cast<Int64&>(dst.y.x) = reinterpret_cast<const Int64&>(src.y.x); 
    reinterpret_cast<Int64&>(dst.y.y) = reinterpret_cast<const Int64&>(src.y.y); 
  }
};

template<>
struct RawCopy<Real3x3> {
  typedef Real3x3 T;
  inline static void copy(T & dst, const T & src) { 
    reinterpret_cast<Int64&>(dst.x.x) = reinterpret_cast<const Int64&>(src.x.x); 
    reinterpret_cast<Int64&>(dst.x.y) = reinterpret_cast<const Int64&>(src.x.y); 
    reinterpret_cast<Int64&>(dst.x.z) = reinterpret_cast<const Int64&>(src.x.z); 
    reinterpret_cast<Int64&>(dst.y.x) = reinterpret_cast<const Int64&>(src.y.x); 
    reinterpret_cast<Int64&>(dst.y.y) = reinterpret_cast<const Int64&>(src.y.y); 
    reinterpret_cast<Int64&>(dst.y.z) = reinterpret_cast<const Int64&>(src.y.z); 
    reinterpret_cast<Int64&>(dst.z.x) = reinterpret_cast<const Int64&>(src.z.x); 
    reinterpret_cast<Int64&>(dst.z.y) = reinterpret_cast<const Int64&>(src.z.y); 
    reinterpret_cast<Int64&>(dst.z.z) = reinterpret_cast<const Int64&>(src.z.z); 
  }
};

#else /* ARCANE_RAWCOPY */

template<>
struct RawCopy<Real> {
  typedef Real T;
  inline static void copy(T & dst, const T & src) { dst = src; }
};

template<>
struct RawCopy<BFloat16> {
  typedef BFloat16 T;
  inline static void copy(T & dst, const T & src) { dst = src; }
};

template<>
struct RawCopy<Float16> {
  typedef Float16 T;
  inline static void copy(T & dst, const T & src) { dst = src; }
};

template<>
struct RawCopy<Float32> {
  typedef Float32 T;
  inline static void copy(T & dst, const T & src) { dst = src; }
};


template<>
struct RawCopy<Real2> {
  typedef Real2 T;
  inline static void copy(T & dst, const T & src) { dst = src; }
};

template<>
struct RawCopy<Real3> {
  typedef Real3 T;
  inline static void copy(T & dst, const T & src) { dst = src; }
};

template<>
struct RawCopy<Real2x2> {
  typedef Real2x2 T;
  inline static void copy(T & dst, const T & src) { dst = src; }
};

template<>
struct RawCopy<Real3x3> {
  typedef Real3x3 T;
  inline static void copy(T & dst, const T & src) { dst = src; }
};

#endif /* ARCANE_RAWCOPY */


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

