// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ArrayTraits.h                                               (C) 2000-2025 */
/*                                                                           */
/* Characteristics of a 1D array.                                            */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_COMMON_ARRAYTRAITS_H
#define ARCCORE_COMMON_ARRAYTRAITS_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/common/CommonGlobal.h"

// For std::byte
#include <cstddef>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Characteristics for an array.
 *
 * This template class can be specialized to indicate that a type must
 * be considered a POD type for array classes (Array, Array2, ...).
 *
 * For these classes, if a type is a POD type, then constructors, destructors
 * and copy operators are not called to manage instances of this
 * type in arrays.
 *
 * To indicate that a type must be considered a POD type, you must
 * use the ARCCORE_DEFINE_ARRAY_PODTYPE macro.
 */
template <typename T>
class ArrayTraits
{
 public:

  typedef const T& ConstReferenceType;
  typedef FalseType IsPODType;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Macro to indicate that the type \a datatype is a POD type for arrays.
 *
 * This macro specializes \a Arccore::ArrayTraits for the type \a datatype. It
 * must therefore be used in the \a Arccore namespace and before the use of
 * the \a datatype type.
 */
#define ARCCORE_DEFINE_ARRAY_PODTYPE(datatype) \
  template <> \
  class ArrayTraits<datatype>           \
  { \
   public: \
\
    typedef datatype ConstReferenceType; \
    typedef TrueType IsPODType; \
  }

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Default implementation indicating that a type is not POD
template <typename T>
class ArrayTraits<T*>
{
 public:

  typedef T* Ptr;
  typedef const Ptr& ConstReferenceType;
  typedef FalseType IsPODType;
};

//! Default implementation indicating that a type is not POD
template <typename T>
class ArrayTraits<const T*>
{
 public:

  typedef T* Ptr;
  typedef const T* ConstReferenceType;
  typedef FalseType IsPODType;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCCORE_DEFINE_ARRAY_PODTYPE(char);
ARCCORE_DEFINE_ARRAY_PODTYPE(signed char);
ARCCORE_DEFINE_ARRAY_PODTYPE(unsigned char);
ARCCORE_DEFINE_ARRAY_PODTYPE(short);
ARCCORE_DEFINE_ARRAY_PODTYPE(int);
ARCCORE_DEFINE_ARRAY_PODTYPE(long);
ARCCORE_DEFINE_ARRAY_PODTYPE(unsigned short);
ARCCORE_DEFINE_ARRAY_PODTYPE(unsigned int);
ARCCORE_DEFINE_ARRAY_PODTYPE(unsigned long);
ARCCORE_DEFINE_ARRAY_PODTYPE(float);
ARCCORE_DEFINE_ARRAY_PODTYPE(double);
ARCCORE_DEFINE_ARRAY_PODTYPE(long double);
ARCCORE_DEFINE_ARRAY_PODTYPE(std::byte);
ARCCORE_DEFINE_ARRAY_PODTYPE(Float16);
ARCCORE_DEFINE_ARRAY_PODTYPE(BFloat16);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
