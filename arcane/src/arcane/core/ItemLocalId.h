// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemLocalId.h                                               (C) 2000-2025 */
/*                                                                           */
/* Local ID of a mesh entity.                                                */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ITEMLOCALID_H
#define ARCANE_CORE_ITEMLOCALID_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArrayView.h"

#include "arcane/core/ItemTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class ItemSharedInfo;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// TODO: deprecate constructors that take an argument
// an ItemEnumerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup Mesh
 * \brief Index of an Item in a variable.
 */
class ARCANE_CORE_EXPORT ItemLocalId
{
 public:

  ItemLocalId() = default;
  constexpr ARCCORE_HOST_DEVICE explicit ItemLocalId(Int32 id)
  : m_local_id(id)
  {}
  // The definition of this constructor is in ItemInternal.h
  inline ItemLocalId(ItemInternal* item);
  inline ItemLocalId(ItemConnectedEnumerator enumerator);
  template <typename ItemType> inline ItemLocalId(ItemEnumeratorT<ItemType> enumerator);
  template <typename ItemType> inline ItemLocalId(ItemConnectedEnumeratorT<ItemType> enumerator);
  inline ItemLocalId(Item item);
  constexpr ARCCORE_HOST_DEVICE operator Int32() const { return m_local_id; }
  constexpr ARCCORE_HOST_DEVICE Int32 asInt32() const { return m_local_id; }
  constexpr ARCCORE_HOST_DEVICE Int32 asInteger() const { return m_local_id; }

 public:

  constexpr ARCCORE_HOST_DEVICE Int32 localId() const { return m_local_id; }
  constexpr ARCCORE_HOST_DEVICE bool isNull() const { return m_local_id == NULL_ITEM_LOCAL_ID; }

 public:

  static SmallSpan<const ItemLocalId> fromSpanInt32(SmallSpan<const Int32> v)
  {
    auto* ptr = reinterpret_cast<const ItemLocalId*>(v.data());
    return { ptr, v.size() };
  }
  static SmallSpan<const Int32> toSpanInt32(SmallSpan<const ItemLocalId> v)
  {
    auto* ptr = reinterpret_cast<const Int32*>(v.data());
    return { ptr, v.size() };
  }

 private:

  Int32 m_local_id = NULL_ITEM_LOCAL_ID;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup Mesh
 * \brief Index of an \a ItemType entity in a variable.
 */
template <typename ItemType_>
class ItemLocalIdT
: public ItemLocalId
{
 public:

  using ItemType = ItemType_;
  using ThatClass = ItemLocalIdT<ItemType>;

 public:

  ItemLocalIdT() = default;
  constexpr ARCCORE_HOST_DEVICE explicit ItemLocalIdT(Int32 id)
  : ItemLocalId(id)
  {}
  inline ItemLocalIdT(ItemInternal* item);
  inline ItemLocalIdT(ItemConnectedEnumeratorT<ItemType> enumerator);
  inline ItemLocalIdT(ItemType item);

 public:

  static SmallSpan<const ItemLocalId> fromSpanInt32(SmallSpan<const Int32> v)
  {
    auto* ptr = reinterpret_cast<const ThatClass*>(v.data());
    return { ptr, v.size() };
  }

  static SmallSpan<const Int32> toSpanInt32(SmallSpan<const ThatClass> v)
  {
    auto* ptr = reinterpret_cast<const Int32*>(v.data());
    return { ptr, v.size() };
  }

 public:

  ARCANE_DEPRECATED_REASON("Y2022: Use strongly typed 'ItemEnumeratorT<ItemType>' or 'ItemType'")
  inline ItemLocalIdT(ItemEnumerator enumerator);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Class to convert an ItemLocalId to an entity (Item).
 *
 * Instances of this class remain valid throughout the lifetime
 * of the associated family.
 */
class ARCANE_CORE_EXPORT ItemLocalIdToItemConverter
{
  template <typename ItemType_> friend class ItemLocalIdToItemConverterT;

 public:

  explicit ItemLocalIdToItemConverter(IItemFamily* family);
  /*!
   * \brief Default constructor.
   *
   * The instance will not be valid until it has been copied
   * from a valid instance (using the constructor that
   * takes an IItemFamily as an argument).
   */
  ItemLocalIdToItemConverter() = default;

 public:

  //! Entity of local ID \a local_id
  inline constexpr ARCCORE_HOST_DEVICE Item operator[](ItemLocalId local_id) const;
  //! Entity of local ID \a local_id
  inline constexpr ARCCORE_HOST_DEVICE Item operator[](Int32 local_id) const;

 private:

  ItemSharedInfo* m_item_shared_info = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Class to convert an ItemLocalId to an entity (Item).
 *
 * Instances of this class remain valid throughout the lifetime
 * of the associated family.
 */
template <typename ItemType_> class ItemLocalIdToItemConverterT
: public ItemLocalIdToItemConverter
{
 public:

  using ItemType = ItemType_;
  using ItemLocalIdType = ItemLocalIdT<ItemType>;

 public:

  using ItemLocalIdToItemConverter::ItemLocalIdToItemConverter;

 public:

  //! Entity of local ID \a local_id
  inline constexpr ARCCORE_HOST_DEVICE ItemType operator[](ItemLocalIdType local_id) const;
  //! Entity of local ID \a local_id
  inline constexpr ARCCORE_HOST_DEVICE ItemType operator[](Int32 local_id) const;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Class to convert a NodeLocalId to an edge.
class NodeLocalIdToNodeConverter
: public ItemLocalIdToItemConverterT<Node>
{
 public:

  using ItemLocalIdToItemConverterT<Node>::ItemLocalIdToItemConverterT;
};

//! Class to convert an EdgeLocalId to an edge.
class EdgeLocalIdToEdgeConverter
: public ItemLocalIdToItemConverterT<Edge>
{
 public:

  using ItemLocalIdToItemConverterT<Edge>::ItemLocalIdToItemConverterT;
};

//! Class to convert a FaceLocalId to a face.
class FaceLocalIdToFaceConverter
: public ItemLocalIdToItemConverterT<Face>
{
 public:

  using ItemLocalIdToItemConverterT<Face>::ItemLocalIdToItemConverterT;
};

//! Class to convert a CellLocalId to a mesh.
class CellLocalIdToCellConverter
: public ItemLocalIdToItemConverterT<Cell>
{
 public:

  using ItemLocalIdToItemConverterT<Cell>::ItemLocalIdToItemConverterT;
};

//! Class to convert a ParticleLocalId to a particle.
class ParticleLocalIdToParticleConverter
: public ItemLocalIdToItemConverterT<Particle>
{
 public:

  using ItemLocalIdToItemConverterT<Particle>::ItemLocalIdToItemConverterT;
};

//! Class to convert a DoFLocalId to a degree of freedom.
class DoFLocalIdToDoFConverter
: public ItemLocalIdToItemConverterT<DoF>
{
 public:

  using ItemLocalIdToItemConverterT<DoF>::ItemLocalIdToItemConverterT;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
