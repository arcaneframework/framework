// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MDIndex.h                                                   (C) 2000-2026 */
/*                                                                           */
/* Management of multi-dimensional array indices.                            */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_BASE_MDINDEX_H
#define ARCCORE_BASE_MDINDEX_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/BaseTypes.h"

#include <array>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Base class for managing indices of an N-dimensional array.
 */
template <int RankValue, typename IndexType_>
class MDIndexBase
{
 public:

  using ExtentIndexType = IndexType_;

 protected:

  // Note: on pourrait utiliser '= default' mais cela ne passe pas
  // with VS2017 car pour lui le constructeur n'est pas 'constexpr'
  constexpr MDIndexBase() {}
  constexpr MDIndexBase(std::array<ExtentIndexType, RankValue> _id)
  : m_indexes(_id)
  {}

 public:

  //! List of indices
  constexpr std::array<ExtentIndexType, RankValue> operator()() const { return m_indexes; }

  //! Returns the i-th index
  constexpr ExtentIndexType operator[](int i) const
  {
    ARCCORE_CHECK_AT(i, RankValue);
    return m_indexes[i];
  }
  //! Returns the i-th index as an Int64
  constexpr Int64 asInt64(int i) const
  {
    ARCCORE_CHECK_AT(i, RankValue);
    return m_indexes[i];
  }

  //! Adds \a rhs to the index values of the instance.
  constexpr void add(const MDIndexBase<RankValue, ExtentIndexType>& rhs)
  {
    for (int i = 0; i < RankValue; ++i)
      m_indexes[i] += rhs[i];
  }

 protected:

  std::array<ExtentIndexType, RankValue> m_indexes = {};
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename IndexType_>
class MDIndex<0, IndexType_>
: public MDIndexBase<0, IndexType_>
{
  using BaseClass = MDIndexBase<0, IndexType_>;

 public:

  using ExtentIndexType = BaseClass::ExtentIndexType;

 public:

  MDIndex() = default;
  constexpr MDIndex(std::array<ExtentIndexType, 0> _id)
  : MDIndexBase<0, IndexType_>(_id)
  {}
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename IndexType_>
class MDIndex<1, IndexType_>
: public MDIndexBase<1, IndexType_>
{
  using BaseClass = MDIndexBase<1, IndexType_>;
  using BaseClass::m_indexes;

 public:

  using ExtentIndexType = BaseClass::ExtentIndexType;

 public:

  MDIndex() = default;
  constexpr MDIndex(ExtentIndexType _id0)
  : BaseClass()
  {
    m_indexes[0] = _id0;
  }
  constexpr MDIndex(std::array<ExtentIndexType, 1> _id)
  : BaseClass(_id)
  {}

 public:

  constexpr ExtentIndexType id0() const { return m_indexes[0]; }
  constexpr Int64 largeId0() const { return m_indexes[0]; }

  // Pour l'index de dimension 1, on autorise la conversion vers l'index
  constexpr operator IndexType_() const { return m_indexes[0]; }

 public:

  //! Convert to a MDIndex with different index type
  template <typename OtherIndexType> static constexpr MDIndex
  fromOther(const MDIndex<1, OtherIndexType>& rhs)
  {
    return MDIndex(static_cast<ExtentIndexType>(rhs.id0()));
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename IndexType_>
class MDIndex<2, IndexType_>
: public MDIndexBase<2, IndexType_>
{
  using BaseClass = MDIndexBase<2, IndexType_>;
  using BaseClass::m_indexes;

 public:

  using ExtentIndexType = BaseClass::ExtentIndexType;

 public:

  MDIndex() = default;
  constexpr MDIndex(ExtentIndexType _id0, ExtentIndexType _id1)
  : BaseClass()
  {
    m_indexes[0] = _id0;
    m_indexes[1] = _id1;
  }
  constexpr MDIndex(std::array<ExtentIndexType, 2> _id)
  : BaseClass(_id)
  {}

 public:

  constexpr ExtentIndexType id0() const { return m_indexes[0]; }
  constexpr ExtentIndexType id1() const { return m_indexes[1]; }
  constexpr Int64 largeId0() const { return m_indexes[0]; }
  constexpr Int64 largeId1() const { return m_indexes[1]; }

 public:

  //! Convert to a MDIndex with different index type
  template <typename OtherIndexType> static constexpr MDIndex
  fromOther(const MDIndex<2, OtherIndexType>& rhs)
  {
    return MDIndex(static_cast<ExtentIndexType>(rhs.id0()),
                   static_cast<ExtentIndexType>(rhs.id1()));
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename IndexType_>
class MDIndex<3, IndexType_>
: public MDIndexBase<3, IndexType_>
{
  using BaseClass = MDIndexBase<3, IndexType_>;
  using BaseClass::m_indexes;

 public:

  using ExtentIndexType = BaseClass::ExtentIndexType;

 public:

  MDIndex() = default;
  constexpr MDIndex(ExtentIndexType _id0, ExtentIndexType _id1, ExtentIndexType _id2)
  : BaseClass()
  {
    m_indexes[0] = _id0;
    m_indexes[1] = _id1;
    m_indexes[2] = _id2;
  }
  constexpr MDIndex(std::array<ExtentIndexType, 3> _id)
  : BaseClass(_id)
  {}

 public:

  constexpr ExtentIndexType id0() const { return m_indexes[0]; }
  constexpr ExtentIndexType id1() const { return m_indexes[1]; }
  constexpr ExtentIndexType id2() const { return m_indexes[2]; }
  constexpr Int64 largeId0() const { return m_indexes[0]; }
  constexpr Int64 largeId1() const { return m_indexes[1]; }
  constexpr Int64 largeId2() const { return m_indexes[2]; }

 public:

  //! Convert to a MDIndex with different index type
  template <typename OtherIndexType> static constexpr MDIndex
  fromOther(const MDIndex<3, OtherIndexType>& rhs)
  {
    return MDIndex(static_cast<ExtentIndexType>(rhs.id0()),
                   static_cast<ExtentIndexType>(rhs.id1()),
                   static_cast<ExtentIndexType>(rhs.id2()));
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename IndexType_>
class MDIndex<4, IndexType_>
: public MDIndexBase<4, IndexType_>
{
  using BaseClass = MDIndexBase<4, IndexType_>;
  using BaseClass::m_indexes;

 public:

  using ExtentIndexType = BaseClass::ExtentIndexType;

 public:

  MDIndex() = default;
  constexpr MDIndex(ExtentIndexType _id0, ExtentIndexType _id1,
                    ExtentIndexType _id2, ExtentIndexType _id3)
  : MDIndexBase<4, IndexType_>()
  {
    m_indexes[0] = _id0;
    m_indexes[1] = _id1;
    m_indexes[2] = _id2;
    m_indexes[3] = _id3;
  }
  constexpr MDIndex(std::array<ExtentIndexType, 4> _id)
  : MDIndexBase<4, IndexType_>(_id)
  {}

 public:

  constexpr ExtentIndexType id0() const { return m_indexes[0]; }
  constexpr ExtentIndexType id1() const { return m_indexes[1]; }
  constexpr ExtentIndexType id2() const { return m_indexes[2]; }
  constexpr ExtentIndexType id3() const { return m_indexes[3]; }
  constexpr Int64 largeId0() const { return m_indexes[0]; }
  constexpr Int64 largeId1() const { return m_indexes[1]; }
  constexpr Int64 largeId2() const { return m_indexes[2]; }
  constexpr Int64 largeId3() const { return m_indexes[3]; }

 public:

  //! Convert to a MDIndex with different index type
  template <typename OtherIndexType> static constexpr MDIndex
  fromOther(const MDIndex<4, OtherIndexType>& rhs)
  {
    return MDIndex(static_cast<ExtentIndexType>(rhs.id0()),
                   static_cast<ExtentIndexType>(rhs.id1()),
                   static_cast<ExtentIndexType>(rhs.id2()),
                   static_cast<ExtentIndexType>(rhs.id3()));
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
