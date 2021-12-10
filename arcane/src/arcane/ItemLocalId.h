// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemLocalId.h                                               (C) 2000-2021 */
/*                                                                           */
/* Index local sur une entité du maillage.                                   */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ITEMLOCALID_H
#define ARCANE_ITEMLOCALID_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/ItemTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup Mesh
 * \brief Index d'un Item dans une variable.
 */
class ARCANE_CORE_EXPORT ItemLocalId
{
 public:
  constexpr ARCCORE_HOST_DEVICE ItemLocalId() : m_local_id(NULL_ITEM_LOCAL_ID){}
  constexpr ARCCORE_HOST_DEVICE explicit ItemLocalId(Int32 id) : m_local_id(id){}
  // La définition de ce constructeur est dans ItemInternal.h
  inline explicit ItemLocalId(ItemInternal* item);
  inline ItemLocalId(Item item);
  constexpr ARCCORE_HOST_DEVICE operator Int32() const { return m_local_id; }
  constexpr ARCCORE_HOST_DEVICE Int32 asInt32() const { return m_local_id; }
  constexpr ARCCORE_HOST_DEVICE Int32 asInteger() const { return m_local_id; }
 public:
  constexpr ARCCORE_HOST_DEVICE Int32 localId() const { return m_local_id; }
 private:
  Int32 m_local_id;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup Mesh
 * \brief Index d'un Node dans une variable.
 */
class ARCANE_CORE_EXPORT NodeLocalId
: public ItemLocalId
{
 public:
  constexpr ARCCORE_HOST_DEVICE explicit NodeLocalId(Int32 id) : ItemLocalId(id){}
  inline NodeLocalId(Node node);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup Mesh
 * \brief Index d'une Edge dans une variable.
 */
class ARCANE_CORE_EXPORT EdgeLocalId
: public ItemLocalId
{
 public:
  constexpr ARCCORE_HOST_DEVICE explicit EdgeLocalId(Int32 id) : ItemLocalId(id){}
  inline EdgeLocalId(Edge edge);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup Mesh
 * \brief Index d'une Face dans une variable.
 */
class ARCANE_CORE_EXPORT FaceLocalId
: public ItemLocalId
{
 public:
  constexpr ARCCORE_HOST_DEVICE explicit FaceLocalId(Int32 id) : ItemLocalId(id){}
  inline FaceLocalId(Face item);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup Mesh
 * \brief Index d'une Cell dans une variable.
 */
class ARCANE_CORE_EXPORT CellLocalId
: public ItemLocalId
{
 public:
  constexpr ARCCORE_HOST_DEVICE explicit CellLocalId(Int32 id) : ItemLocalId(id){}
  inline CellLocalId(Cell item);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup Mesh
 * \brief Index d'une Particle dans une variable.
 */
class ARCANE_CORE_EXPORT ParticleLocalId
: public ItemLocalId
{
 public:
  constexpr ARCCORE_HOST_DEVICE explicit ParticleLocalId(Int32 id) : ItemLocalId(id){}
  inline ParticleLocalId(Particle item);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vue typée sur une liste d'entités d'une connectivité.
 */
template <typename ItemType>
class ItemLocalIdView
{
 public:
  using LocalIdType = typename ItemType::LocalIdType;
  using SpanType = SmallSpan<const LocalIdType>;
  using iterator = typename SpanType::iterator;
  using const_iterator = typename SpanType::const_iterator;
 public:
  constexpr ARCCORE_HOST_DEVICE ItemLocalIdView(SpanType ids) : m_ids(ids){}
  constexpr ARCCORE_HOST_DEVICE ItemLocalIdView(const LocalIdType* ids,Int32 s) : m_ids(ids,s){}
  ItemLocalIdView() = default;
  constexpr ARCCORE_HOST_DEVICE operator SpanType() const { return m_ids; }
 public:
  constexpr ARCCORE_HOST_DEVICE SpanType ids() const { return m_ids; }
  constexpr ARCCORE_HOST_DEVICE LocalIdType operator[](Int32 i) const { return m_ids[i]; }
  constexpr ARCCORE_HOST_DEVICE Int32 size() const { return m_ids.size(); }
  constexpr ARCCORE_HOST_DEVICE iterator begin() { return m_ids.begin(); }
  constexpr ARCCORE_HOST_DEVICE iterator end() { return m_ids.end(); }
  constexpr ARCCORE_HOST_DEVICE const_iterator begin() const { return m_ids.begin(); }
  constexpr ARCCORE_HOST_DEVICE const_iterator end() const { return m_ids.end(); }
 public:
  constexpr ARCCORE_HOST_DEVICE const LocalIdType* data() const { return m_ids.data(); }
 private:
  SpanType m_ids;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
