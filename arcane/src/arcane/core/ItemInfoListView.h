// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemInfoListView.h                                          (C) 2000-2024 */
/*                                                                           */
/* View of a list to obtain information about entities.                      */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ITEMINFOLISTVIEW_H
#define ARCANE_ITEMINFOLISTVIEW_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ItemGenericInfoListView.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief View of a list to obtain information about entities.
 *
 * Like all views, these instances are temporary and should not be
 * kept between two modifications of the associated family.
 *
 * The methods of this class are only valid if the instance has been initialized
 * with a non-null family (IItemFamily).
 *
 * Via this class, it is possible to retrieve an Item instance from
 * an ItemLocalId.
 */
class ARCANE_CORE_EXPORT ItemInfoListView
: public ItemGenericInfoListView
{
  using BaseClass = ItemGenericInfoListView;
  friend class mesh::ItemFamily;
  friend ItemVector;
  friend ItemPairEnumerator;
  friend ItemGenericInfoListView;
  template <int Extent> friend class ItemConnectedListView;
  template <typename ItemType> friend class ItemEnumeratorBaseT;

  // To be deleted when we no longer need _itemsInternal()
  friend ItemVectorView;

 public:

  ItemInfoListView() = default;

  /*!
   * \brief Constructs a view associated with the family \a family.
   *
   * \a family may be \a nullptr in which case the instance is not
   * usable for retrieving entity information
   */
  explicit ItemInfoListView(IItemFamily* family);

 public:

  //! Associated family
  IItemFamily* itemFamily() const { return m_item_shared_info->itemFamily(); }

  // NOTE: The definitions of the two operator[] methods are in Item.h

  //! Entity associated with local ID \a local_id
  inline Item operator[](ItemLocalId local_id) const;

  //! Entity associated with local ID \a local_id
  inline Item operator[](Int32 local_id) const;

 private:

  // Only ItemFamily can create instances via this constructor
  explicit ItemInfoListView(ItemSharedInfo* shared_info)
  : ItemGenericInfoListView(shared_info)
  {}

 protected:

  using BaseClass::m_flags;
  using BaseClass::m_item_shared_info;
  void _checkValid(eItemKind expected_kind);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Base class for specialized views of entity information.
 */
template <typename ItemType>
class ItemInfoListViewT
: public ItemInfoListView
{
 public:

  ItemInfoListViewT() = default;

  //! Constructs a view associated with the family \a family.
  explicit ItemInfoListViewT(IItemFamily* family)
  : ItemInfoListView(family)
  {
    _checkValid(ItemTraitsT<ItemType>::kind());
  }

 public:

  // NOTE: The definitions of the two operator[] methods are in Item.h

  //! Entity associated with local ID \a local_id
  inline ItemType operator[](ItemLocalId local_id) const;

  //! Entity associated with local ID \a local_id
  inline ItemType operator[](Int32 local_id) const;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief View of node information.
 */
class NodeInfoListView
: public ItemInfoListViewT<Node>
{
 public:

  using BaseClass = ItemInfoListViewT<Node>;

 public:

  NodeInfoListView() = default;

  //! Constructs a view associated with the family \a family.
  explicit NodeInfoListView(IItemFamily* family)
  : BaseClass(family)
  {}
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief View of edge information.
 */
class EdgeInfoListView
: public ItemInfoListViewT<Edge>
{
 public:

  using BaseClass = ItemInfoListViewT<Edge>;

 public:

  EdgeInfoListView() = default;

  //! Constructs a view associated with the family \a family.
  explicit EdgeInfoListView(IItemFamily* family)
  : BaseClass(family)
  {}
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief View of face information.
 */
class FaceInfoListView
: public ItemInfoListViewT<Face>
{
 public:

  using BaseClass = ItemInfoListViewT<Face>;

 public:

  FaceInfoListView() = default;

  //! Constructs a view associated with the family \a family.
  explicit FaceInfoListView(IItemFamily* family)
  : BaseClass(family)
  {}

 public:

  constexpr ARCCORE_HOST_DEVICE bool isSubDomainBoundary(FaceLocalId local_id) const
  {
    return ItemFlags::isSubDomainBoundary(m_flags[local_id]);
  }
  constexpr ARCCORE_HOST_DEVICE bool isSubDomainBoundaryOutside(FaceLocalId local_id) const
  {
    return ItemFlags::isSubDomainBoundaryOutside(m_flags[local_id]);
  }
  constexpr ARCCORE_HOST_DEVICE Int32 backCellIndex(FaceLocalId local_id) const
  {
    return ItemFlags::backCellIndex(m_flags[local_id]);
  }
  constexpr ARCCORE_HOST_DEVICE Int32 frontCellIndex(FaceLocalId local_id) const
  {
    return ItemFlags::frontCellIndex(m_flags[local_id]);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief View of cell information.
 */
class CellInfoListView
: public ItemInfoListViewT<Cell>
{
 public:

  using BaseClass = ItemInfoListViewT<Cell>;

 public:

  CellInfoListView() = default;

  //! Constructs a view associated with the family \a family.
  explicit CellInfoListView(IItemFamily* family)
  : BaseClass(family)
  {}
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief View of particle information.
 */
class ParticleInfoListView
: public ItemInfoListViewT<Particle>
{
 public:

  using BaseClass = ItemInfoListViewT<Particle>;

 public:

  ParticleInfoListView() = default;

  //! Constructs a view associated with the family \a family.
  explicit ParticleInfoListView(IItemFamily* family)
  : BaseClass(family)
  {}
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief View of DoF information.
 */
class DoFInfoListView
: public ItemInfoListViewT<DoF>
{
 public:

  using BaseClass = ItemInfoListViewT<DoF>;

 public:

  DoFInfoListView() = default;

  //! Constructs a view associated with the family \a family.
  explicit DoFInfoListView(IItemFamily* family)
  : BaseClass(family)
  {}
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
