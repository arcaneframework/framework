// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IndexedSpace.h                                                   (C) 2014 */
/*                                                                           */
/* Space for linear algebra.                                                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_INDEXEDSPACE_H
#define ARCANE_INDEXEDSPACE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//#include "arcane/ArcaneTypes.h"
#include "fake.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
  * \brief Indexed set/space to define matrix and vector support.
 */
class IndexedSpace
{
 public:
  IndexedSpace() {}
  IndexedSpace(const IndexedSpace& src) {}

  virtual ~IndexedSpace() {}

  IndexedSpace& operator=(const IndexedSpace& src) {return *this;}

  /*!
   * \brief Return whether the given space is a subspace.
   */
  virtual bool isSubspace(const IndexedSpace& other) const { return true;}

  /*!
   * \brief Return if the given space and us are the same.
   */
  virtual bool isSame(const IndexedSpace& other) const { return true;}
  
  /*!
   * \brief Return if the spaces can interoperate.
   */
  virtual bool isCompatible(const IndexedSpace& other) const { return true;}  
};


class EmptyIndexedSpace: public IndexedSpace
{
 public:
  EmptyIndexedSpace() {}
  EmptyIndexedSpace(const IndexedSpace& src) {}

  ~EmptyIndexedSpace() {}

  /*!
   * \brief Return whether the given space is a subspace.
   */
  bool isSubspace(const IndexedSpace& other) const { return false;}

  /*!
   * \brief Return if the given space and us are the same.
   */
  bool isSame(const IndexedSpace& other) const { return false;}
  
  /*!
   * \brief Return if the spaces can interoperate.
   */
  bool isCompatible(const IndexedSpace& other) const { return true;}
};







bool 
operator<(const IndexedSpace& s1, const IndexedSpace& s2) 
{
  return s2.isSubspace(s1);
}

const IndexedSpace&
operator+( const IndexedSpace& s1, const IndexedSpace& s2)
{
  if (!s1.isCompatible(s2)) { // Error.
    return s1; // TODO: Throw an exception.
  }

  if (s1 < s2) { // m_out is a subspace of s2
    return s2;
  }
  return s1;
}



/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif // ARCANE_INDEXEDSPACE_H
