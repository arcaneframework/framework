// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ArrayBoundsIndex.h                                          (C) 2000-2022 */
/*                                                                           */
/* Gestion des indices des tableaux multi-dimensionnels.                     */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_ARRAYBOUNDSINDEX_H
#define ARCANE_UTILS_ARRAYBOUNDSINDEX_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcaneGlobal.h"

#include <array>

/*
 * ATTENTION:
 *
 * Toutes les classes de ce fichier sont expérimentales et l'API n'est pas
 * figée. A NE PAS UTILISER EN DEHORS DE ARCANE.
 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<int RankValue>
class ArrayBoundsIndexBase
{
 public:
  ARCCORE_HOST_DEVICE std::array<Int32,RankValue> operator()() const { return m_indexes; }
 protected:
  ARCCORE_HOST_DEVICE constexpr ArrayBoundsIndexBase()
  {
    for( int i=0; i<RankValue; ++i )
      m_indexes[i] = 0;
  }
  ARCCORE_HOST_DEVICE constexpr ArrayBoundsIndexBase(std::array<Int32,RankValue> _id) : m_indexes(_id){}
 public:
  ARCCORE_HOST_DEVICE constexpr Int32 operator[](int i) const
  {
    ARCCORE_CHECK_AT(i,RankValue);
    return m_indexes[i];
  }
  ARCCORE_HOST_DEVICE constexpr Int64 asInt64(int i) const
  {
    ARCCORE_CHECK_AT(i,RankValue);
    return m_indexes[i];
  }
  ARCCORE_HOST_DEVICE constexpr void add(const ArrayBoundsIndexBase<RankValue>& rhs)
  {
    for( int i=0; i<RankValue; ++i )
      m_indexes[i] += rhs[i];
  }
 protected:
  std::array<Int32,RankValue> m_indexes;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<>
class ArrayBoundsIndex<1>
: public ArrayBoundsIndexBase<1>
{
 public:
  ARCCORE_HOST_DEVICE constexpr ArrayBoundsIndex() : ArrayBoundsIndexBase<1>(){}
  ARCCORE_HOST_DEVICE constexpr ArrayBoundsIndex(Int32 _id0) : ArrayBoundsIndexBase<1>()
  {
    m_indexes[0] = _id0;
  }
  ARCCORE_HOST_DEVICE constexpr ArrayBoundsIndex(std::array<Int32,1> _id)
  : ArrayBoundsIndexBase<1>(_id) {}
 public:
  ARCCORE_HOST_DEVICE constexpr Int32 id0() const { return m_indexes[0]; }
  ARCCORE_HOST_DEVICE constexpr Int64 largeId0() const { return m_indexes[0]; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<>
class ArrayBoundsIndex<2>
: public ArrayBoundsIndexBase<2>
{
 public:
  ARCCORE_HOST_DEVICE constexpr ArrayBoundsIndex() : ArrayBoundsIndexBase<2>(){}
  ARCCORE_HOST_DEVICE constexpr ArrayBoundsIndex(Int32 _id0,Int32 _id1)
  : ArrayBoundsIndexBase<2>()
  {
    m_indexes[0] = _id0;
    m_indexes[1] = _id1;
  }
  ARCCORE_HOST_DEVICE constexpr ArrayBoundsIndex(std::array<Int32,2> _id)
  : ArrayBoundsIndexBase<2>(_id) {}
 public:
  ARCCORE_HOST_DEVICE constexpr Int32 id0() const { return m_indexes[0]; }
  ARCCORE_HOST_DEVICE constexpr Int32 id1() const { return m_indexes[1]; }
  ARCCORE_HOST_DEVICE constexpr Int64 largeId0() const { return m_indexes[0]; }
  ARCCORE_HOST_DEVICE constexpr Int64 largeId1() const { return m_indexes[1]; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<>
class ArrayBoundsIndex<3>
: public ArrayBoundsIndexBase<3>
{
 public:
  ARCCORE_HOST_DEVICE constexpr ArrayBoundsIndex() : ArrayBoundsIndexBase<3>(){}
  ARCCORE_HOST_DEVICE constexpr ArrayBoundsIndex(Int32 _id0,Int32 _id1,Int32 _id2)
  : ArrayBoundsIndexBase<3>()
  {
    m_indexes[0] = _id0;
    m_indexes[1] = _id1;
    m_indexes[2] = _id2;
  }
  ARCCORE_HOST_DEVICE constexpr ArrayBoundsIndex(std::array<Int32,3> _id)
  : ArrayBoundsIndexBase<3>(_id) {}
 public:
  ARCCORE_HOST_DEVICE constexpr Int32 id0() const { return m_indexes[0]; }
  ARCCORE_HOST_DEVICE constexpr Int32 id1() const { return m_indexes[1]; }
  ARCCORE_HOST_DEVICE constexpr Int32 id2() const { return m_indexes[2]; }
  ARCCORE_HOST_DEVICE constexpr Int64 largeId0() const { return m_indexes[0]; }
  ARCCORE_HOST_DEVICE constexpr Int64 largeId1() const { return m_indexes[1]; }
  ARCCORE_HOST_DEVICE constexpr Int64 largeId2() const { return m_indexes[2]; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<>
class ArrayBoundsIndex<4>
: public ArrayBoundsIndexBase<4>
{
 public:
  ARCCORE_HOST_DEVICE constexpr ArrayBoundsIndex() : ArrayBoundsIndexBase<4>(){}
  ARCCORE_HOST_DEVICE constexpr ArrayBoundsIndex(Int32 _id0,Int32 _id1,Int32 _id2,Int32 _id3)
  : ArrayBoundsIndexBase<4>()
  {
    m_indexes[0] = _id0;
    m_indexes[1] = _id1;
    m_indexes[2] = _id2;
    m_indexes[3] = _id3;
  }
  ARCCORE_HOST_DEVICE constexpr ArrayBoundsIndex(std::array<Int32,4> _id)
  : ArrayBoundsIndexBase<4>(_id) {}
 public:
  ARCCORE_HOST_DEVICE constexpr Int32 id0() const { return m_indexes[0]; }
  ARCCORE_HOST_DEVICE constexpr Int32 id1() const { return m_indexes[1]; }
  ARCCORE_HOST_DEVICE constexpr Int32 id2() const { return m_indexes[2]; }
  ARCCORE_HOST_DEVICE constexpr Int32 id3() const { return m_indexes[3]; }
  ARCCORE_HOST_DEVICE constexpr Int64 largeId0() const { return m_indexes[0]; }
  ARCCORE_HOST_DEVICE constexpr Int64 largeId1() const { return m_indexes[1]; }
  ARCCORE_HOST_DEVICE constexpr Int64 largeId2() const { return m_indexes[2]; }
  ARCCORE_HOST_DEVICE constexpr Int64 largeId3() const { return m_indexes[3]; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
