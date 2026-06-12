// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemPairGroup.cc                                            (C) 2000-2025 */
/*                                                                           */
/* Table of entity lists.                                                    */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/IFunctor.h"
#include "arcane/utils/NotImplementedException.h"

#include "arcane/core/ItemPairGroup.h"
#include "arcane/core/IItemFamily.h"
#include "arcane/core/ItemGroup.h"
#include "arcane/core/ItemPairEnumerator.h"
#include "arcane/core/ItemPairGroupBuilder.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \class ItemPairGroup
 * \ingroup Mesh
 * \brief Table of entity lists.
 *
 * This class allows managing a list of entities associated with each entity
 * of an entity group (ItemGroup). For example, for every node in a group, the set
 * of cells connected to this node by faces.
 *
 * This class has a reference semantics in the same way as the
 * ItemGroup class.
 *
 * %Arcane provides a predefined set of methods to calculate the connectivities
 * of entities connected to other entities by a specific entity type. To use these
 * methods, you must use the following constructor:
 * ItemPairGroup(const ItemGroup& group,const ItemGroup& sub_item_group,
 * #eItemKind link_kind). \a link_kind then indicates the entity type
 * that links them. For example:
 *
 \code
 * CellGroup cells1;
 * CellGroup cells2;
 * // g1 contains for each cell in \a cells1 the cells that are
 * // connected to it by nodes and belong to the group \a cells2
 * CellCellGroup g1(cells1,cells2,IK_Node);
 * ENUMERATE_ITEMPAIR(Cell,Cell,iitem,ad_list){
 *   Cell cell = *iitem;
 *   // Iterates over cells connected to 'cell'
 *   ENUMERATE_SUB_ITEM(Cell,isubitem,iitem){
 *     Cell sub_cell = *iitem;
 *     ...
 *   }
 * }
 \endcode
 *
 * It is possible for the user to specify a particular way
 * of calculating connectivities by specifying a functor of type
 * ItemPairGroup::CustomFunctor as an argument to the constructor.
 *
 * \warning The functor passed as an argument must be allocated by
 * the new operator and will be destroyed at the same time as the associated ItemPairGroup.
 *
 * Here is a complete example that calculates the cells
 * connected to the cells via faces:
 *
 \code
 * auto f = [](ItemPairGroupBuilder& builder)
 *   {
 *     const ItemPairGroup& pair_group = builder.group();
 *     const ItemGroup& items = pair_group.itemGroup();
 *     const ItemGroup& sub_items = pair_group.subItemGroup();

 *     // Marks all entities that are not allowed to belong to
 *     // the connectivity list because they are not in \a sub_items;
 *     std::set<Int32> allowed_ids;
 *     ENUMERATE_CELL(iitem,sub_items) {
 *       allowed_ids.insert(iitem.itemLocalId());
 *     }

 *     Int32Array local_ids;
 *     local_ids.reserve(8);

 *     // List of entities already processed for the current cell
 *     std::set<Int32> already_in_list;
 *     ENUMERATE_CELL(icell,items){
 *       Cell cell = *icell;
 *       local_ids.clear();
 *       Int32 current_local_id = icell.itemLocalId();
 *       already_in_list.clear();

 *       // To avoid adding itself to its own connectivity list
 *       already_in_list.insert(current_local_id);

 *       for( FaceEnumerator iface(cell.faces()); iface.hasNext(); ++iface ){
 *         Face face = *iface;
 *         for( CellEnumerator isubcell(face.cells()); isubcell.hasNext(); ++isubcell ){
 *           const Int32 sub_local_id = isubcell.itemLocalId();
 *          // Checks if we are in the list of allowed cells and if we
 *           // have not yet been processed.
 *           if (allowed_ids.find(sub_local_id)==allowed_ids.end())
 *             continue;
 *           if (already_in_list.find(sub_local_id)!=already_in_list.end())
 *             continue;
 *           // This cell must be added. We mark it so as not to
 *           // iterate over it and we add it to the list.
 *           already_in_list.insert(sub_local_id);
 *           local_ids.add(sub_local_id);
 *         }
 *       }
 *       builder.addNextItem(local_ids);
 *     }
 *   };
 *
 * // Creates a group that calculates connectivities over all cells.
 * ItemPairGroupT<Cell,Cell> ad_list(allCells(),allCells(),functor::makePointer(f));
 \endcode
 */

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Wrapper for an ItemPairGroup::CustomFunctor.
 */
class ItemPairGroup::CustomFunctorWrapper
: public IFunctor
{
 public:

  CustomFunctorWrapper(ItemPairGroupImpl* g, ItemPairGroup::CustomFunctor* f)
  : m_group(g)
  , m_functor(f)
  {}
  ~CustomFunctorWrapper()
  {
    delete m_functor;
  }

 public:

  void executeFunctor() override
  {
    ItemPairGroup pair_group(m_group);
    ItemPairGroupBuilder builder(pair_group);
    m_functor->executeFunctor(builder);
  }

 public:

  ItemPairGroupImpl* m_group = nullptr;
  ItemPairGroup::CustomFunctor* m_functor = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemPairGroup::
ItemPairGroup(const ItemGroup& group, const ItemGroup& sub_item_group,
              eItemKind link_kind)
: m_impl(nullptr)
{
  IItemFamily* item_family = group.itemFamily();
  ItemPairGroup v = item_family->findAdjacencyItems(group, sub_item_group, link_kind, 1);
  m_impl = v.internal();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemPairGroup::
ItemPairGroup(const ItemGroup& group, const ItemGroup& sub_item_group,
              CustomFunctor* functor)
: m_impl(nullptr)
{
  ARCANE_CHECK_POINTER(functor);
  m_impl = new ItemPairGroupImpl(group, sub_item_group);
  IFunctor* f = new CustomFunctorWrapper(m_impl.get(), functor);
  m_impl->setComputeFunctor(f);
  m_impl->invalidate(true);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemPairGroup::
ItemPairGroup(ItemPairGroupImpl* p)
: m_impl(p)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemPairGroup::
ItemPairGroup()
: m_impl(ItemPairGroupImpl::checkSharedNull())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemPairEnumerator ItemPairGroup::
enumerator() const
{
  return ItemPairEnumerator(*this);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
