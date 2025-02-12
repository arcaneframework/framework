// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ArrayTraits.h                                               (C) 2000-2025 */
/*                                                                           */
/* Caractéristiques d'un tableau 1D.                                         */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_COLLECTIONS_ARRAYTRAITS_H
#define ARCCORE_COLLECTIONS_ARRAYTRAITS_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/collections/CollectionsGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Caractéristiques pour un tableau.
 *
 * Cette classe template peut être spécialisée pour indiquer qu'un type doit
 * être considéré comme un type POD pour les classes tableaux (Array, Array2, ...).
 *
 * Pour ces classes, si un type est un type POD, alors les constructeurs, destructeurs
 * et opérateurs de recopies ne sont pas appelés pour gérer les instances de ce
 * type dans les tableaux.
 *
 * Pour indiquer qu'un type doit être considéré comme un type POD, il faut
 * utiliser la macro ARCCORE_DEFINE_ARRAY_PODTYPE.
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
 * \brief Macro pour indiquer que le type \a datatype est un type POD pour les tableaux.
 *
 * Cette macro spécialise \a Arccore::ArrayTraits pour le type \a datatype. Elle
 * donc être utilisé dans le namespace \a Arccore et avant l'utilisation du
 * type \a datatype.
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
//! Implémentation par défaut indiquant qu'un type n'est pas POD
template <typename T>
class ArrayTraits<T*>
{
 public:

  typedef T* Ptr;
  typedef const Ptr& ConstReferenceType;
  typedef FalseType IsPODType;
};

//! Implémentation par défaut indiquant qu'un type n'est pas POD
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
