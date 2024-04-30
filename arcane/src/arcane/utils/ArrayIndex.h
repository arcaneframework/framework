// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ArrayIndex.h                                                (C) 2000-2024 */
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
template <int RankValue, typename IndexType_>
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

template <typename IndexType_>
class ArrayIndex<0, IndexType_>
: public ArrayIndexBase<0, IndexType_>
{
 public:

  ArrayIndex() = default;
  ARCCORE_HOST_DEVICE constexpr ArrayIndex(std::array<Int32, 0> _id)
  : ArrayIndexBase<0, IndexType_>(_id)
  {}
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename IndexType_>
class ArrayIndex<1, IndexType_>
: public ArrayIndexBase<1, IndexType_>
{
  using BaseClass = ArrayIndexBase<1, IndexType_>;
  using BaseClass::m_indexes;

 public:

  ArrayIndex() = default;
  ARCCORE_HOST_DEVICE constexpr ArrayIndex(Int32 _id0)
  : BaseClass()
  {
    m_indexes[0] = _id0;
  }
  ARCCORE_HOST_DEVICE constexpr ArrayIndex(std::array<Int32, 1> _id)
  : BaseClass(_id)
  {}

 public:

  ARCCORE_HOST_DEVICE constexpr Int32 id0() const { return m_indexes[0]; }
  ARCCORE_HOST_DEVICE constexpr Int64 largeId0() const { return m_indexes[0]; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename IndexType_>
class ArrayIndex<2, IndexType_>
: public ArrayIndexBase<2, IndexType_>
{
  using BaseClass = ArrayIndexBase<2, IndexType_>;
  using BaseClass::m_indexes;

 public:

  ArrayIndex() = default;
  ARCCORE_HOST_DEVICE constexpr ArrayIndex(Int32 _id0, Int32 _id1)
  : BaseClass()
  {
    m_indexes[0] = _id0;
    m_indexes[1] = _id1;
  }
  ARCCORE_HOST_DEVICE constexpr ArrayIndex(std::array<Int32, 2> _id)
  : BaseClass(_id)
  {}

 public:

  ARCCORE_HOST_DEVICE constexpr Int32 id0() const { return m_indexes[0]; }
  ARCCORE_HOST_DEVICE constexpr Int32 id1() const { return m_indexes[1]; }
  ARCCORE_HOST_DEVICE constexpr Int64 largeId0() const { return m_indexes[0]; }
  ARCCORE_HOST_DEVICE constexpr Int64 largeId1() const { return m_indexes[1]; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename IndexType_>
class ArrayIndex<3, IndexType_>
: public ArrayIndexBase<3, IndexType_>
{
  using BaseClass = ArrayIndexBase<3, IndexType_>;
  using BaseClass::m_indexes;

 public:

  ArrayIndex() = default;
  ARCCORE_HOST_DEVICE constexpr ArrayIndex(Int32 _id0, Int32 _id1, Int32 _id2)
  : BaseClass()
  {
    m_indexes[0] = _id0;
    m_indexes[1] = _id1;
    m_indexes[2] = _id2;
  }
  ARCCORE_HOST_DEVICE constexpr ArrayIndex(std::array<Int32, 3> _id)
  : BaseClass(_id)
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

template <typename IndexType_>
class ArrayIndex<4, IndexType_>
: public ArrayIndexBase<4, IndexType_>
{
  using BaseClass = ArrayIndexBase<4, IndexType_>;
  using BaseClass::m_indexes;

 public:

  ArrayIndex() = default;
  ARCCORE_HOST_DEVICE constexpr ArrayIndex(Int32 _id0, Int32 _id1, Int32 _id2, Int32 _id3)
  : ArrayIndexBase<4, IndexType_>()
  {
    m_indexes[0] = _id0;
    m_indexes[1] = _id1;
    m_indexes[2] = _id2;
    m_indexes[3] = _id3;
  }
  ARCCORE_HOST_DEVICE constexpr ArrayIndex(std::array<Int32, 4> _id)
  : ArrayIndexBase<4, IndexType_>(_id)
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
