// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemPairGroup.h                                             (C) 2000-2025 */
/*                                                                           */
/* Table of entity lists.                                                    */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ITEMPAIRGROUP_H
#define ARCANE_CORE_ITEMPAIRGROUP_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/AutoRef.h"
#include "arcane/utils/Iterator.h"
#include "arcane/utils/IFunctorWithArgument.h"

#include "arcane/core/ItemPairGroupImpl.h"
#include "arcane/core/ItemTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//NOTE: The complete documentation is in ItemPairGroup.cc

/*!
 * \brief Table of entity lists.
 */
class ARCANE_CORE_EXPORT ItemPairGroup
{
 public:

  /*!
   * \brief Functor for custom connectivity calculation.
   */
  typedef IFunctorWithArgumentT<ItemPairGroupBuilder&> CustomFunctor;
  class CustomFunctorWrapper;

 public:

  //! Constructs an empty table.
  ItemPairGroup();
  //! Constructs a group from the internal representation \a prv
  explicit ItemPairGroup(ItemPairGroupImpl* prv);
  /*!
   * \brief Constructs an instance by specifying the neighborhood via entities
   * of kind \a link_kind.
   */
  ItemPairGroup(const ItemGroup& group, const ItemGroup& sub_item_group,
                eItemKind link_kind);
  //! Constructs an instance with a specific functor.
  ItemPairGroup(const ItemGroup& group, const ItemGroup& sub_item_group,
                CustomFunctor* functor);
  //! Copy constructor.
  ItemPairGroup(const ItemPairGroup& from)
  : m_impl(from.m_impl)
  {}

  const ItemPairGroup& operator=(const ItemPairGroup& from)
  {
    m_impl = from.m_impl;
    return (*this);
  }
  virtual ~ItemPairGroup() = default;

 public:

  //! \a true means the group is the null group
  inline bool null() const { return m_impl->null(); }
  //! Type of entities in the group
  inline eItemKind itemKind() const { return m_impl->itemKind(); }
  //! Type of sub-entities in the group
  inline eItemKind subItemKind() const { return m_impl->subItemKind(); }

 public:

  /*!
   * \brief Returns the group implementation.
   *
   * \warning This method returns a pointer to the group's internal representation
   * and should not be used outside of Arcane.
   */
  ItemPairGroupImpl* internal() const { return m_impl.get(); }

  //! Entity family to which this group belongs (0 for a null list)
  IItemFamily* itemFamily() const { return m_impl->itemFamily(); }

  //! Entity family to which this group belongs (0 for a null list)
  IItemFamily* subItemFamily() const { return m_impl->subItemFamily(); }

  //! Mesh to which this list belongs (0 for a null list)
  IMesh* mesh() const { return m_impl->mesh(); }

  //! Initial item group
  const ItemGroup& itemGroup() const { return m_impl->itemGroup(); }

  //! Final item group (after bounce)
  const ItemGroup& subItemGroup() const { return m_impl->subItemGroup(); }

 public:

  /*! \brief Invalidates the list.
   */
  void invalidate(bool force_recompute = false)
  {
    m_impl->invalidate(force_recompute);
  }

  //! Internal check of group validity
  void checkValid() { m_impl->checkValid(); }

 public:

  ItemPairEnumerator enumerator() const;

 protected:

  //! Internal representation of the group.
  AutoRefT<ItemPairGroupImpl> m_impl;

 protected:

  //! Returns the group \a impl if it is of kind \a kt, the null group otherwise
  static ItemPairGroupImpl* _check(ItemPairGroupImpl* impl, eItemKind ik, eItemKind aik)
  {
    return (impl->itemKind() == ik && impl->subItemKind() == aik) ? impl : ItemPairGroupImpl::checkSharedNull();
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Compares the references of two groups.
 * \retval true if \a g1 and \a g2 refer to the same group,
 * \retval false otherwise.
 */
inline bool
operator==(const ItemPairGroup& g1, const ItemPairGroup& g2)
{
  return g1.internal() == g2.internal();
}

/*!
 * \brief Compares the references of two groups.
 * \retval true if \a g1 and \a g2 do not refer to the same group,
 * \retval false otherwise.
 */
inline bool
operator!=(const ItemPairGroup& g1, const ItemPairGroup& g2)
{
  return g1.internal() != g2.internal();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Reference to a group of a given kind.
 */
template <typename ItemKind, typename SubItemKind>
class ItemPairGroupT
: public ItemPairGroup
{
 public:

  //! Type of this class
  typedef ItemPairGroupT<ItemKind, SubItemKind> ThatClass;
  //! Type of the class containing the entity characteristics
  typedef ItemTraitsT<ItemKind> TraitsType;
  typedef ItemTraitsT<SubItemKind> SubTraitsType;

  typedef typename TraitsType::ItemType ItemType;
  typedef typename TraitsType::ItemGroupType ItemGroupType;
  typedef typename SubTraitsType::ItemType SubItemType;
  typedef typename SubTraitsType::ItemGroupType SubItemGroupType;

 public:

  ItemPairGroupT() {}
  ItemPairGroupT(const ItemPairGroup& from)
  : ItemPairGroup(_check(from.internal(), TraitsType::kind(), SubTraitsType::kind()))
  {}
  ItemPairGroupT(const ThatClass& from)
  : ItemPairGroup(from)
  {}
  ItemPairGroupT(const ItemGroupType& group, const SubItemGroupType& sub_group,
                 eItemKind link_kind)
  : ItemPairGroup(group, sub_group, link_kind)
  {}
  ItemPairGroupT(const ItemGroupType& group, const SubItemGroupType& sub_group,
                 CustomFunctor* functor)
  : ItemPairGroup(group, sub_group, functor)
  {}
  ~ItemPairGroupT() {}

 public:

  const ThatClass& operator=(const ThatClass& from)
  {
    m_impl = from.internal();
    return (*this);
  }
  const ThatClass& operator=(const ItemPairGroup& from)
  {
    _assign(from);
    return (*this);
  }

 protected:

  void _assign(const ItemPairGroup& from)
  {
    m_impl = _check(from.internal(), TraitsType::kind(), SubTraitsType::kind());
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
