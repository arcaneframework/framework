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

#include "arcane/utils/UtilsTypes.h"

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
/*!
 * \brief Classe de base de la gestion des indices d'un tableau N-dimension.
 */
template <int RankValue>
class ArrayIndexBase
{
 protected:

  // Note: on pourrait utiliser '= default' mais cela ne passe pas
  // avec VS2017 car pour lui le constructeur n'est pas 'constexpr'
  constexpr ArrayIndexBase() {}
  constexpr ArrayIndexBase(std::array<Int32, RankValue> _id)
  : m_indexes(_id)
  {}

 public:

  //! Liste des indices
  constexpr std::array<Int32, RankValue> operator()() const { return m_indexes; }

  //! Retourne le i-ème indice
  constexpr ARCCORE_HOST_DEVICE Int32 operator[](int i) const
  {
    ARCCORE_CHECK_AT(i, RankValue);
    return m_indexes[i];
  }
  //! Retourne le i-ème indice sous la forme d'un Int64
  constexpr ARCCORE_HOST_DEVICE Int64 asInt64(int i) const
  {
    ARCCORE_CHECK_AT(i, RankValue);
    return m_indexes[i];
  }

  //! Ajoute \a rhs aux valeurs des indices de l'instance.
  constexpr ARCCORE_HOST_DEVICE void add(const ArrayIndexBase<RankValue>& rhs)
  {
    for (int i = 0; i < RankValue; ++i)
      m_indexes[i] += rhs[i];
  }

 protected:

  std::array<Int32, RankValue> m_indexes = {};
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <>
class ArrayIndex<0>
: public ArrayIndexBase<0>
{
 public:

  ArrayIndex() = default;
  ARCCORE_HOST_DEVICE constexpr ArrayIndex(std::array<Int32, 0> _id)
  : ArrayIndexBase<0>(_id)
  {}
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <>
class ArrayIndex<1>
: public ArrayIndexBase<1>
{
 public:

  ArrayIndex() = default;
  ARCCORE_HOST_DEVICE constexpr ArrayIndex(Int32 _id0)
  : ArrayIndexBase<1>()
  {
    m_indexes[0] = _id0;
  }
  ARCCORE_HOST_DEVICE constexpr ArrayIndex(std::array<Int32, 1> _id)
  : ArrayIndexBase<1>(_id)
  {}

 public:

  ARCCORE_HOST_DEVICE constexpr Int32 id0() const { return m_indexes[0]; }
  ARCCORE_HOST_DEVICE constexpr Int64 largeId0() const { return m_indexes[0]; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <>
class ArrayIndex<2>
: public ArrayIndexBase<2>
{
 public:

  ArrayIndex() = default;
  ARCCORE_HOST_DEVICE constexpr ArrayIndex(Int32 _id0, Int32 _id1)
  : ArrayIndexBase<2>()
  {
    m_indexes[0] = _id0;
    m_indexes[1] = _id1;
  }
  ARCCORE_HOST_DEVICE constexpr ArrayIndex(std::array<Int32, 2> _id)
  : ArrayIndexBase<2>(_id)
  {}

 public:

  ARCCORE_HOST_DEVICE constexpr Int32 id0() const { return m_indexes[0]; }
  ARCCORE_HOST_DEVICE constexpr Int32 id1() const { return m_indexes[1]; }
  ARCCORE_HOST_DEVICE constexpr Int64 largeId0() const { return m_indexes[0]; }
  ARCCORE_HOST_DEVICE constexpr Int64 largeId1() const { return m_indexes[1]; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <>
class ArrayIndex<3>
: public ArrayIndexBase<3>
{
 public:

  ArrayIndex() = default;
  ARCCORE_HOST_DEVICE constexpr ArrayIndex(Int32 _id0, Int32 _id1, Int32 _id2)
  : ArrayIndexBase<3>()
  {
    m_indexes[0] = _id0;
    m_indexes[1] = _id1;
    m_indexes[2] = _id2;
  }
  ARCCORE_HOST_DEVICE constexpr ArrayIndex(std::array<Int32, 3> _id)
  : ArrayIndexBase<3>(_id)
  {}

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

template <>
class ArrayIndex<4>
: public ArrayIndexBase<4>
{
 public:

  ArrayIndex() = default;
  ARCCORE_HOST_DEVICE constexpr ArrayIndex(Int32 _id0, Int32 _id1, Int32 _id2, Int32 _id3)
  : ArrayIndexBase<4>()
  {
    m_indexes[0] = _id0;
    m_indexes[1] = _id1;
    m_indexes[2] = _id2;
    m_indexes[3] = _id3;
  }
  ARCCORE_HOST_DEVICE constexpr ArrayIndex(std::array<Int32, 4> _id)
  : ArrayIndexBase<4>(_id)
  {}

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

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
