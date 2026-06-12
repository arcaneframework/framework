// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemGroup.cc                                                (C) 2000-2025 */
/*                                                                           */
/* Mesh entity groups.                                                       */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ItemGroup.h"

#include "arcane/utils/String.h"
#include "arcane/utils/ArgumentException.h"

#include "arcane/core/IItemFamily.h"
#include "arcane/core/IMesh.h"
#include "arcane/core/ICaseMng.h"
#include "arcane/core/CaseOptionBase.h"
#include "arcane/core/ICaseOptionList.h"
#include "arcane/core/MeshHandle.h"
#include "arcane/core/internal/ItemGroupInternal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \class ItemGroup

 A mesh entity group is a list of entities of the same family (IItemFamily).

 An entity can only appear once in a group.

 An instance of this class holds a reference to any mesh entity group. From
 this reference, it is possible to know the type (itemKind(), the name (name()),
 and the number of elements (size()) of the group and to iterate generically
 over the elements that compose it. To iterate over derived elements
 (cells, nodes, ...), it must first be converted into a reference to a
 specific group (NodeGroup, FaceGroup, EdgeGroup, or CellGroup). For example:
 \code
 ItemGroup group = subDomain()->defaultMesh()->findGroup("Surface");
 FaceGroup surface(surface);
 if (surface.null())
   // Not a surface.
 if (surface.empty())
   // Surface exists but is empty
 \endcode

 It is possible to sort a group so that its elements are always classified
 in ascending order of the elements' uniqueId(), in order to ensure that
 sequential and parallel codes produce the same result.

 There is a special group, called the null group, which allows representing
 an unreferenced group, meaning one that does not exist. This group is the
 only one for which null() returns \c true. The null group has the following
 properties:
 \arg null() == \c true;
 \arg size() == \c 0;
 \arg name().null() == \c true;

 This class uses a reference counter and is therefore used by reference.
 For example:
 \code
 ItemGroup a = subDomain()->defaultMesh()->findGroup("Toto");
 ItemGroup b = a; // b and a refer to the same group.
 if (a.null())
   // Group not found...
   ;
 \endcode

 To iterate over the entities of a group, an enumerator must be used,
 via the ENUMERATE_* macros, for example ENUMERATE_CELL for a cell group:
 \code
 * CellGroup g;
 * ENUMERATE_CELL(icell,g){
 *   m_mass[icell] = m_volume[icell] * m_density[icell];
 * }
 \endcode

 It is possible to add (addItems()) or remove entities from a group
 (removeItems()).

 Groups that have no parents are persistent and can be recovered during
 a restart. The elements of these groups are automatically updated when
 the associated family is modified. For example, if an element of a family
 is deleted and belonged to a group, it is automatically deleted from that
 group. Similarly, groups are updated during a mesh repartitioning. However,
 there is a small restriction with the current implementation regarding
 this usage. To avoid updating the group with every family change, the group
 is marked as needing to be updated (via invalidate()) upon every change but
 is only actually recalculated when it is used. It is therefore theoretically
 possible that multiple additions and deletions between two uses of the group
 render its elements inconsistent (TODO: link to detailed explanation). To
 avoid this problem, it is possible to force the recalculation of the group
 by calling invalidate() with \a true as an argument.

 Derived groups (which have a parent), such as own() or cellGroup(), are
 invalidated and emptied of their elements when the associated family
 is modified.

 If a group is used as support for partial variables, then the entities
 belonging to the group must be consistent across subdomains. That is, if
 an entity \a x is present in several subdomains (whether as a local or
 ghost entity), it must be in this group for all subdomains or in none of
 the groups. For example, if the cell with uniqueId() 238 is present in
 subdomains 1, 4, and 8, and for subdomain 4 it is in the cell group
 'TOTO', then it must also be in this cell group 'TOTO' for subdomains 1 and 8.
*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemGroup::
ItemGroup(ItemGroupImpl* grp)
: m_impl(grp)
{
  // If \a grp is null, it is replaced by the null group.
  // This is done (version 2.3) for compatibility reasons.
  // Eventually, this constructor will be explicit, and in that case, you will have to do:
  //   ARCANE_CHECK_POINTER(grp);
  if (!grp) {
    std::cerr << "Creating group with null pointer is not allowed\n";
    m_impl = ItemGroupImpl::checkSharedNull();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemGroup::
ItemGroup()
: m_impl(ItemGroupImpl::checkSharedNull())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool ItemGroup::
isOwn() const
{
  if (null())
    return true;

  return m_impl->isOwn();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemGroup::
setOwn(bool v)
{
  if (!null())
    m_impl->setOwn(v);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Group equivalent to this one but containing only the local elements
 * of the subdomain.
 *
 * If this group is already a group containing only the local elements of the
 * subdomain, it is returned itself:
 * \code
 * group.own()==group; // For a local group
 * group.own().own()==group.own(); // Invariant
 * \endcode
 */
ItemGroup ItemGroup::
own() const
{
  if (null() || isOwn())
    return (*this);
  m_impl->checkNeedUpdate();
  return ItemGroup(m_impl->ownGroup());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// group of items owned by the subdomain
ItemGroup ItemGroup::
ghost() const
{
  if (null())
    return ItemGroup();
  m_impl->checkNeedUpdate();
  return ItemGroup(m_impl->ghostGroup());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Items in the group lying on the boundary between two subdomains
// Implemented for faces only
ItemGroup ItemGroup::
interface() const
{
  if (null())
    return ItemGroup();
  m_impl->checkNeedUpdate();
  return ItemGroup(m_impl->interfaceGroup());
}

// HANDLE CRASH SORTED
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

NodeGroup ItemGroup::
nodeGroup() const
{
  if (null())
    return ItemGroup();
  m_impl->checkNeedUpdate();
  return NodeGroup(m_impl->nodeGroup());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

EdgeGroup ItemGroup::
edgeGroup() const
{
  if (null())
    return ItemGroup();
  m_impl->checkNeedUpdate();
  return EdgeGroup(m_impl->edgeGroup());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

FaceGroup ItemGroup::
faceGroup() const
{
  if (null())
    return ItemGroup();
  m_impl->checkNeedUpdate();
  return FaceGroup(m_impl->faceGroup());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CellGroup ItemGroup::
cellGroup() const
{
  if (null())
    return ItemGroup();
  m_impl->checkNeedUpdate();
  return CellGroup(m_impl->cellGroup());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

FaceGroup ItemGroup::
innerFaceGroup() const
{
  if (null())
    return ItemGroup();
  m_impl->checkNeedUpdate();
  return FaceGroup(m_impl->innerFaceGroup());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

FaceGroup ItemGroup::
outerFaceGroup() const
{
  if (null())
    return ItemGroup();
  m_impl->checkNeedUpdate();
  return FaceGroup(m_impl->outerFaceGroup());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
//! AMR
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CellGroup ItemGroup::
activeCellGroup() const
{
  if (null())
    return ItemGroup();
  m_impl->checkNeedUpdate();
  return CellGroup(m_impl->activeCellGroup());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CellGroup ItemGroup::
ownActiveCellGroup() const
{
  if (null())
    return ItemGroup();
  m_impl->checkNeedUpdate();
  return CellGroup(m_impl->ownActiveCellGroup());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CellGroup ItemGroup::
levelCellGroup(const Integer& level) const
{
  if (null())
    return ItemGroup();
  m_impl->checkNeedUpdate();
  return CellGroup(m_impl->levelCellGroup(level));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CellGroup ItemGroup::
ownLevelCellGroup(const Integer& level) const
{
  if (null())
    return ItemGroup();
  m_impl->checkNeedUpdate();
  return CellGroup(m_impl->ownLevelCellGroup(level));
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

FaceGroup ItemGroup::
activeFaceGroup() const
{
  if (null())
    return ItemGroup();
  m_impl->checkNeedUpdate();
  return FaceGroup(m_impl->activeFaceGroup());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

FaceGroup ItemGroup::
ownActiveFaceGroup() const
{
  if (null())
    return ItemGroup();
  m_impl->checkNeedUpdate();
  return FaceGroup(m_impl->ownActiveFaceGroup());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

FaceGroup ItemGroup::
innerActiveFaceGroup() const
{
  if (null())
    return ItemGroup();
  m_impl->checkNeedUpdate();
  return FaceGroup(m_impl->innerActiveFaceGroup());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

FaceGroup ItemGroup::
outerActiveFaceGroup() const
{
  if (null())
    return ItemGroup();
  m_impl->checkNeedUpdate();
  return FaceGroup(m_impl->outerActiveFaceGroup());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemGroup ItemGroup::
createSubGroup(const String& suffix, IItemFamily* family, ItemGroupComputeFunctor* functor) const
{
  if (null())
    return ItemGroup();
  m_impl->checkNeedUpdate();
  return ItemGroup(m_impl->createSubGroup(suffix, family, functor));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemGroup ItemGroup::
findSubGroup(const String& suffix) const
{
  if (null())
    return ItemGroup();
  return ItemGroup(m_impl->findSubGroup(suffix));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemGroup::
clear()
{
  m_impl->clear();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * Adds the local ID entities \a items_local_id.
 *
 * The parameter \a check_if_present indicates whether to check if the entities
 * to be added are already present in the group; if so, they will not be
 * added. If the caller is certain that the entities to be added
 * are not currently in the group, they can set the
 * parameter \a check_if_present to \a false, which speeds up the addition.
 */
void ItemGroup::
addItems(Int32ConstArrayView items_local_id, bool check_if_present)
{
  if (null())
    throw ArgumentException(A_FUNCINFO, "Can not addItems() to null group");
  if (isAllItems())
    throw ArgumentException(A_FUNCINFO, "Can not addItems() to all-items group");
  m_impl->_checkNeedUpdateNoPadding();
  m_impl->addItems(items_local_id, check_if_present);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * Removes the local ID entities \a items_local_id.
 *
 * The parameter \a check_if_present indicates whether to check if the entities
 * to be removed are already present in the group; if so, they will not be
 * removed. If the caller is certain that the entities to be removed
 * are in the group, they can set the
 * parameter \a check_if_present to \a false, which speeds up the removal.
 */
void ItemGroup::
removeItems(Int32ConstArrayView items_local_id, bool check_if_present)
{
  if (null())
    throw ArgumentException(A_FUNCINFO, "Can not removeItems() to null group");
  if (isAllItems())
    throw ArgumentException(A_FUNCINFO, "Can not removeItems() to all-items group");
  m_impl->_checkNeedUpdateNoPadding();
  m_impl->removeItems(items_local_id, check_if_present);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Positions the group entities.
 *
 * Positions the entities whose local IDs are given by
 * \a items_local_id.
 * The caller guarantees that each entity is present only once in
 * this array
 */
void ItemGroup::
setItems(Int32ConstArrayView items_local_id)
{
  if (null())
    throw ArgumentException(A_FUNCINFO, "Can not setItems() to null group");
  m_impl->setItems(items_local_id);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Positions the group entities.
 *
 * Positions the entities whose local IDs are given by
 * \a items_local_id.
 * The caller guarantees that each entity is present only once in
 * this array
 * If \a do_sort is true, the entities are sorted by increasing uniqueId
 * before being added to the group.
 */
void ItemGroup::
setItems(Int32ConstArrayView items_local_id, bool do_sort)
{
  if (null())
    throw ArgumentException(A_FUNCINFO, "Can not setItems() to null group");
  m_impl->setItems(items_local_id, do_sort);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Internal check of group validity.
 */
void ItemGroup::
checkValid()
{
  m_impl->checkValid();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemGroup::
applyOperation(IItemOperationByBasicType* operation) const
{
  if (null())
    return;
  m_impl->checkNeedUpdate();
  m_impl->applyOperation(operation);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemEnumerator ItemGroup::
enumerator() const
{
  if (null())
    return ItemEnumerator();
  m_impl->_checkNeedUpdateNoPadding();
  return ItemEnumerator(m_impl->itemInfoListView(), m_impl->itemsLocalId(), m_impl.get());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemEnumerator ItemGroup::
_simdEnumerator() const
{
  if (null())
    return ItemEnumerator();
  m_impl->_checkNeedUpdateWithPadding();
  return ItemEnumerator(m_impl->itemInfoListView(), m_impl->itemsLocalId(), m_impl.get());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemVectorView ItemGroup::
_view(bool do_padding) const
{
  if (null())
    return ItemVectorView();
  m_impl->_checkNeedUpdate(do_padding);
  Int32 flags = 0;
  if (m_impl->isContiguousLocalIds())
    flags |= ItemIndexArrayView::F_Contigous;
  // TODO: gérer l'offset
  return ItemVectorView(m_impl->itemFamily(), ItemIndexArrayView(m_impl->itemsLocalId(), 0, flags));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemVectorView ItemGroup::
view() const
{
  return _view(true);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemVectorView ItemGroup::
_paddedView() const
{
  return _view(true);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemVectorView ItemGroup::
_unpaddedView() const
{
  return _view(false);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool ItemGroup::
isAllItems() const
{
  return m_impl->isAllItems();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IVariableSynchronizer* ItemGroup::
synchronizer() const
{
  return m_impl->synchronizer();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool ItemGroup::
isAutoComputed() const
{
  return m_impl->hasComputeFunctor();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool ItemGroup::
hasSynchronizer() const
{
  return m_impl->hasSynchronizer();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemGroupImplInternal* ItemGroup::
_internalApi() const
{
  return m_impl->_internalApi();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool ItemGroup::
checkIsSorted() const
{
  m_impl->_checkNeedUpdate(false);
  return m_impl->checkIsSorted();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemGroup::
incrementTimestamp() const
{
  m_impl->m_p->updateTimestamp();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern ARCANE_CORE_EXPORT bool
_caseOptionConvert(const CaseOptionBase& co, const String& name, ItemGroup& obj)
{
  IMesh* mesh = co.parentOptionList()->meshHandle().mesh();
  obj = mesh->findGroup(name);
  return obj.null();
}

extern ARCANE_CORE_EXPORT bool
_caseOptionConvert(const CaseOptionBase& co, const String& name, NodeGroup& obj)
{
  IMesh* mesh = co.parentOptionList()->meshHandle().mesh();
  obj = mesh->nodeFamily()->findGroup(name);
  return obj.null();
}

extern ARCANE_CORE_EXPORT bool
_caseOptionConvert(const CaseOptionBase& co, const String& name, EdgeGroup& obj)
{
  IMesh* mesh = co.parentOptionList()->meshHandle().mesh();
  obj = mesh->edgeFamily()->findGroup(name);
  return obj.null();
}

extern ARCANE_CORE_EXPORT bool
_caseOptionConvert(const CaseOptionBase& co, const String& name, FaceGroup& obj)
{
  IMesh* mesh = co.parentOptionList()->meshHandle().mesh();
  obj = mesh->faceFamily()->findGroup(name);
  return obj.null();
}

extern ARCANE_CORE_EXPORT bool
_caseOptionConvert(const CaseOptionBase& co, const String& name, CellGroup& obj)
{
  IMesh* mesh = co.parentOptionList()->meshHandle().mesh();
  obj = mesh->cellFamily()->findGroup(name);
  return obj.null();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
