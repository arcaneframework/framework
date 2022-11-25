// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ArrayIndex.h                                                (C) 2000-2022 */
/*                                                                           */
/* Gestion des indices des tableaux multi-dimensionnels.                     */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_ARRAYINDEX_H
#define ARCANE_UTILS_ARRAYINDEX_H
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

template <int RankValue>
class ArrayIndexBase
{
 public:
 protected:

  constexpr ARCCORE_HOST_DEVICE ArrayIndexBase() = default;
  constexpr ARCCORE_HOST_DEVICE ArrayIndexBase(std::array<Int32, RankValue> _id)
  : m_indexes(_id)
  {}

 public:

  constexpr std::array<Int32, RankValue> operator()() const { return m_indexes; }

  constexpr ARCCORE_HOST_DEVICE Int32 operator[](int i) const
  {
    ARCCORE_CHECK_AT(i, RankValue);
    return m_indexes[i];
  }
  constexpr ARCCORE_HOST_DEVICE Int64 asInt64(int i) const
  {
    ARCCORE_CHECK_AT(i, RankValue);
    return m_indexes[i];
  }
  constexpr ARCCORE_HOST_DEVICE void add(const ArrayIndexBase<RankValue>& rhs)
  {
    for (int i = 0; i < RankValue; ++i)
      m_indexes[i] += rhs[i];
  }

 protected:

  std::array<Int32,RankValue> m_indexes = { };
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<>
class ArrayIndex<0>
: public ArrayIndexBase<0>
{
 public:
  ARCCORE_HOST_DEVICE constexpr ArrayIndex() : ArrayIndexBase<0>(){}
  ARCCORE_HOST_DEVICE constexpr ArrayIndex(std::array<Int32,0> _id)
  : ArrayIndexBase<0>(_id) {}
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<>
class ArrayIndex<1>
: public ArrayIndexBase<1>
{
 public:
  ARCCORE_HOST_DEVICE constexpr ArrayIndex() : ArrayIndexBase<1>(){}
  ARCCORE_HOST_DEVICE constexpr ArrayIndex(Int32 _id0) : ArrayIndexBase<1>()
  {
    m_indexes[0] = _id0;
  }
  ARCCORE_HOST_DEVICE constexpr ArrayIndex(std::array<Int32,1> _id)
  : ArrayIndexBase<1>(_id) {}
 public:
  ARCCORE_HOST_DEVICE constexpr Int32 id0() const { return m_indexes[0]; }
  ARCCORE_HOST_DEVICE constexpr Int64 largeId0() const { return m_indexes[0]; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<>
class ArrayIndex<2>
: public ArrayIndexBase<2>
{
 public:
  ARCCORE_HOST_DEVICE constexpr ArrayIndex() : ArrayIndexBase<2>(){}
  ARCCORE_HOST_DEVICE constexpr ArrayIndex(Int32 _id0,Int32 _id1)
  : ArrayIndexBase<2>()
  {
    m_indexes[0] = _id0;
    m_indexes[1] = _id1;
  }
  ARCCORE_HOST_DEVICE constexpr ArrayIndex(std::array<Int32,2> _id)
  : ArrayIndexBase<2>(_id) {}
 public:
  ARCCORE_HOST_DEVICE constexpr Int32 id0() const { return m_indexes[0]; }
  ARCCORE_HOST_DEVICE constexpr Int32 id1() const { return m_indexes[1]; }
  ARCCORE_HOST_DEVICE constexpr Int64 largeId0() const { return m_indexes[0]; }
  ARCCORE_HOST_DEVICE constexpr Int64 largeId1() const { return m_indexes[1]; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<>
class ArrayIndex<3>
: public ArrayIndexBase<3>
{
 public:
  ARCCORE_HOST_DEVICE constexpr ArrayIndex() : ArrayIndexBase<3>(){}
  ARCCORE_HOST_DEVICE constexpr ArrayIndex(Int32 _id0,Int32 _id1,Int32 _id2)
  : ArrayIndexBase<3>()
  {
    m_indexes[0] = _id0;
    m_indexes[1] = _id1;
    m_indexes[2] = _id2;
  }
  ARCCORE_HOST_DEVICE constexpr ArrayIndex(std::array<Int32,3> _id)
  : ArrayIndexBase<3>(_id) {}
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
class ArrayIndex<4>
: public ArrayIndexBase<4>
{
 public:
  ARCCORE_HOST_DEVICE constexpr ArrayIndex() : ArrayIndexBase<4>(){}
  ARCCORE_HOST_DEVICE constexpr ArrayIndex(Int32 _id0,Int32 _id1,Int32 _id2,Int32 _id3)
  : ArrayIndexBase<4>()
  {
    m_indexes[0] = _id0;
    m_indexes[1] = _id1;
    m_indexes[2] = _id2;
    m_indexes[3] = _id3;
  }
  ARCCORE_HOST_DEVICE constexpr ArrayIndex(std::array<Int32,4> _id)
  : ArrayIndexBase<4>(_id) {}
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
