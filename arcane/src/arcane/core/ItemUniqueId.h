// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemUniqueId.h                                              (C) 2000-2025 */
/*                                                                           */
/* Type d'un identifiant unique pour une entité.                             */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ITEMUNIQUEID_H
#define ARCANE_CORE_ITEMUNIQUEID_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/HashFunction.h"

#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Identifiant unique d'une entité.
 * \ingroup Mesh
 */
class ARCANE_CORE_EXPORT ItemUniqueId
{
 public:

  ItemUniqueId() = default;
  constexpr ARCCORE_HOST_DEVICE explicit ItemUniqueId(Int64 uid)
  : m_unique_id(uid)
  {}

  constexpr ARCCORE_HOST_DEVICE operator Int64() const { return m_unique_id; }
  ARCANE_DEPRECATED operator Int32() const { return asInt32(); }
  constexpr ARCCORE_HOST_DEVICE Int64 asInt64() const { return m_unique_id; }
  Int32 asInt32() const;
  Integer asInteger() const
  {
#ifdef ARCANE_64BIT
    return asInt64();
#else
    return asInt32();
#endif
  }

 public:

  friend inline bool operator<(ItemUniqueId lhs, ItemUniqueId rhs)
  {
    return lhs.asInt64() < rhs.asInt64();
  }
  friend inline bool operator<(Int64 lhs, ItemUniqueId rhs)
  {
    return lhs < rhs.asInt64();
  }
  friend inline bool operator<(ItemUniqueId lhs, Int32 rhs)
  {
    return lhs.asInt64() < rhs;
  }
  friend inline bool operator<(Int32 lhs, ItemUniqueId rhs)
  {
    return lhs < rhs.asInt64();
  }
  friend inline bool operator<(ItemUniqueId lhs, Int64 rhs)
  {
    return lhs.asInt64() < rhs;
  }

  friend inline bool operator<=(ItemUniqueId lhs, ItemUniqueId rhs)
  {
    return lhs.asInt64() <= rhs.asInt64();
  }
  friend inline bool operator<=(Int64 lhs, ItemUniqueId rhs)
  {
    return lhs <= rhs.asInt64();
  }
  friend inline bool operator<=(ItemUniqueId lhs, Int64 rhs)
  {
    return lhs.asInt64() <= rhs;
  }
  friend inline bool operator<=(Int32 lhs, ItemUniqueId rhs)
  {
    return lhs <= rhs.asInt64();
  }
  friend inline bool operator<=(ItemUniqueId lhs, Int32 rhs)
  {
    return lhs.asInt64() <= rhs;
  }

  friend inline bool operator>(ItemUniqueId lhs, ItemUniqueId rhs)
  {
    return lhs.asInt64() > rhs.asInt64();
  }
  friend inline bool operator>(Int64 lhs, ItemUniqueId rhs)
  {
    return lhs > rhs.asInt64();
  }
  friend inline bool operator>(ItemUniqueId lhs, Int64 rhs)
  {
    return lhs.asInt64() > rhs;
  }
  friend inline bool operator>(Int32 lhs, ItemUniqueId rhs)
  {
    return lhs > rhs.asInt64();
  }
  friend inline bool operator>(ItemUniqueId lhs, Int32 rhs)
  {
    return lhs.asInt64() > rhs;
  }

  friend inline bool operator>=(ItemUniqueId lhs, ItemUniqueId rhs)
  {
    return lhs.asInt64() >= rhs.asInt64();
  }
  friend inline bool operator>=(Int64 lhs, ItemUniqueId rhs)
  {
    return lhs >= rhs.asInt64();
  }
  friend inline bool operator>=(ItemUniqueId lhs, Int64 rhs)
  {
    return lhs.asInt64() >= rhs;
  }
  friend inline bool operator>=(Int32 lhs, ItemUniqueId rhs)
  {
    return lhs >= rhs.asInt64();
  }
  friend inline bool operator>=(ItemUniqueId lhs, Int32 rhs)
  {
    return lhs.asInt64() >= rhs;
  }

  friend inline bool operator!=(ItemUniqueId lhs, ItemUniqueId rhs)
  {
    return lhs.asInt64() != rhs.asInt64();
  }
  friend inline bool operator!=(Int64 lhs, ItemUniqueId rhs)
  {
    return lhs != rhs.asInt64();
  }
  friend inline bool operator!=(ItemUniqueId lhs, Int64 rhs)
  {
    return lhs.asInt64() != rhs;
  }
  friend inline bool operator!=(Int32 lhs, ItemUniqueId rhs)
  {
    return lhs != rhs.asInt64();
  }
  friend inline bool operator!=(ItemUniqueId lhs, Int32 rhs)
  {
    return lhs.asInt64() != rhs;
  }

  friend inline bool operator==(ItemUniqueId lhs, ItemUniqueId rhs)
  {
    return lhs.asInt64() == rhs.asInt64();
  }
  friend inline bool operator==(Int64 lhs, ItemUniqueId rhs)
  {
    return lhs == rhs.asInt64();
  }
  friend inline bool operator==(ItemUniqueId lhs, Int64 rhs)
  {
    return lhs.asInt64() == rhs;
  }
  friend inline bool operator==(Int32 lhs, ItemUniqueId rhs)
  {
    return lhs == rhs.asInt64();
  }
  friend inline bool operator==(ItemUniqueId lhs, Int32 rhs)
  {
    return lhs.asInt64() == rhs;
  }

  friend ARCANE_CORE_EXPORT std::ostream&
  operator<<(std::ostream& o, const ItemUniqueId&);

 private:

  Int64 m_unique_id = NULL_ITEM_ID;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Spécialisation pour les Int64
 */
template <>
class HashTraitsT<ItemUniqueId>
{
 public:

  typedef ItemUniqueId KeyTypeConstRef;
  typedef ItemUniqueId& KeyTypeRef;
  typedef ItemUniqueId KeyTypeValue;
  typedef Int64 HashValueType;
  typedef TrueType Printable;

 public:

  static Int64 hashFunction(ItemUniqueId key)
  {
    return IntegerHashFunctionT<Int64>::hashfunc(key);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
