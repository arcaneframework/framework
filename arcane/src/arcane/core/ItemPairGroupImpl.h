// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemPairGroupImpl.h                                         (C) 2000-2025 */
/*                                                                           */
/* Implementation of an array of lists of entities.                          */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ITEMPAIRGROUPIMPL_H
#define ARCANE_CORE_ITEMPAIRGROUPIMPL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ItemTypes.h"
#include "arcane/core/SharedReference.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class ItemPairGroupImplPrivate;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Implementation of an array of lists of entities.
 */
class ARCANE_CORE_EXPORT ItemPairGroupImpl
: public SharedReference
{
 public:

  ItemPairGroupImpl();
  ItemPairGroupImpl(const ItemGroup& group, const ItemGroup& sub_group);
  ~ItemPairGroupImpl() override; //!< Releases resources

 public:

  static ItemPairGroupImpl* shared_null;
  static ItemPairGroupImpl* checkSharedNull();

 public:

  virtual ISharedReference& sharedReference() { return *this; }

 public:

  //! Group name
  const String& name() const;

  //! Number of references on the group.
  virtual Integer nbRef() const { return refCount(); }

  //! Returns true if the group is null.
  bool null() const;

  //! Mesh to which the group belongs (0 for the null group).
  IMesh* mesh() const;

  //! Group kind. This is the kind of its elements.
  eItemKind itemKind() const;

  //! Group kind. This is the kind of its elements.
  eItemKind subItemKind() const;

  //! Family to which the group belongs (or 0 if none)
  IItemFamily* itemFamily() const;

  //! Family to which the group belongs (or 0 if none)
  IItemFamily* subItemFamily() const;

  //! Group of entities
  const ItemGroup& itemGroup() const;

  //! Group of sub-entities
  const ItemGroup& subItemGroup() const;

  //! Number of entities in the group
  Integer size() const;

  //! Invalidates the group
  void invalidate(bool force_recompute);

  //! Checks that the group is valid.
  void checkValid();

  /*!
   * \brief Updates the group if necessary.
   *
   * A group must be updated when it becomes invalid, for example
   * following a call to invalidate().
   * \retval true if the group was updated,
   * \retval false otherwise.
   */
  bool checkNeedUpdate();

  //! Changes the indices of the group's entities
  void changeIds(IntegerConstArrayView old_to_new_ids);

  /*!
   * \internal
   */
  Array<Int64>& unguardedIndexes() const;

  /*!
   * \internal
   */
  Array<Int32>& unguardedLocalIds() const;

  /*!
   * \internal
   */
  void setComputeFunctor(IFunctor* functor);

  Int64ArrayView indexes();

  Span<const Int32> subItemsLocalId();

 private:

  ItemPairGroupImplPrivate* m_p = nullptr; //!< Group implementation

 public:

  void addRef() override;
  void removeRef() override;

 private:

  void deleteMe() override;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
