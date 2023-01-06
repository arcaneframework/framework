// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AbstractItemFamilyTopologyModifier.cc                       (C) 2000-2023 */
/*                                                                           */
/* Modification de la topologie des entités d'une famille.                   */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/NotSupportedException.h"
#include "arcane/utils/FatalErrorException.h"

#include "arcane/IItemFamily.h"
#include "arcane/Item.h"
#include "arcane/ItemInfoListView.h"

#include "arcane/mesh/AbstractItemFamilyTopologyModifier.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AbstractItemFamilyTopologyModifier::
AbstractItemFamilyTopologyModifier(IItemFamily* afamily)
: TraceAccessor(afamily->traceMng())
, m_family(afamily)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IItemFamily* AbstractItemFamilyTopologyModifier::
family() const
{
  return m_family;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Recherche l'index de l'entité de localid() \a local_id dans
 * la liste \a items
 */
inline Integer AbstractItemFamilyTopologyModifier::
_getItemIndex(const Int32* items,Integer nb_item,Int32 local_id)
{
  for( Integer i=0; i<nb_item; ++i )
    if (items[i] == local_id)
      return i;
  ARCANE_FATAL("Can not find item to replace local_id={0}",local_id);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline Integer AbstractItemFamilyTopologyModifier::
_getItemIndex(ItemInternalVectorView items,Int32 local_id)
{
  return _getItemIndex(items.localIds().data(),items.size(),local_id);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AbstractItemFamilyTopologyModifier::
_throwNotSupported()
{
  ARCANE_THROW(NotSupportedException,"Connectivity modification not supported for family name={0}",
               m_family->name());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AbstractItemFamilyTopologyModifier::
replaceNode(ItemLocalId item_lid,Integer index,ItemLocalId new_node_lid)
{
  ARCANE_UNUSED(item_lid);
  ARCANE_UNUSED(index);
  ARCANE_UNUSED(new_node_lid);
  _throwNotSupported();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AbstractItemFamilyTopologyModifier::
replaceEdge(ItemLocalId item_lid,Integer index,ItemLocalId new_edge_lid)
{
  ARCANE_UNUSED(item_lid);
  ARCANE_UNUSED(index);
  ARCANE_UNUSED(new_edge_lid);
  _throwNotSupported();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AbstractItemFamilyTopologyModifier::
replaceFace(ItemLocalId item_lid,Integer index,ItemLocalId new_face_lid)
{
  ARCANE_UNUSED(item_lid);
  ARCANE_UNUSED(index);
  ARCANE_UNUSED(new_face_lid);
  _throwNotSupported();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AbstractItemFamilyTopologyModifier::
replaceCell(ItemLocalId item_lid,Integer index,ItemLocalId new_cell_lid)
{
  ARCANE_UNUSED(item_lid);
  ARCANE_UNUSED(index);
  ARCANE_UNUSED(new_cell_lid);
  _throwNotSupported();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AbstractItemFamilyTopologyModifier::
replaceHParent(ItemLocalId item_lid,Integer index,ItemLocalId new_hparent_lid)
{
  ARCANE_UNUSED(item_lid);
  ARCANE_UNUSED(index);
  ARCANE_UNUSED(new_hparent_lid);
  _throwNotSupported();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AbstractItemFamilyTopologyModifier::
replaceHChild(ItemLocalId item_lid,Integer index,ItemLocalId new_hchild_lid)
{
  ARCANE_UNUSED(item_lid);
  ARCANE_UNUSED(index);
  ARCANE_UNUSED(new_hchild_lid);
  _throwNotSupported();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AbstractItemFamilyTopologyModifier::
findAndReplaceNode(ItemLocalId item_lid,ItemLocalId old_node_lid,
                  ItemLocalId new_node_lid)
{
  ItemInternal* ii = m_family->itemInfoListView()[item_lid].internal();
  Int32 index = _getItemIndex(ii->internalNodes(),old_node_lid);
  this->replaceNode(ItemLocalId(ii->localId()),index,new_node_lid);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AbstractItemFamilyTopologyModifier::
findAndReplaceEdge(ItemLocalId item_lid,ItemLocalId old_edge_lid,
                   ItemLocalId new_edge_lid)
{
  ItemInternal* ii = m_family->itemInfoListView()[item_lid].internal();
  Int32 index = _getItemIndex(ii->internalEdges(),old_edge_lid);
  this->replaceEdge(ItemLocalId(ii->localId()),index,new_edge_lid);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AbstractItemFamilyTopologyModifier::
findAndReplaceFace(ItemLocalId item_lid,ItemLocalId old_face_lid,
                   ItemLocalId new_face_lid)
{
  ItemInternal* ii = m_family->itemInfoListView()[item_lid].internal();
  Int32 index = _getItemIndex(ii->internalFaces(),old_face_lid);
  this->replaceFace(ItemLocalId(ii->localId()),index,new_face_lid);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AbstractItemFamilyTopologyModifier::
findAndReplaceCell(ItemLocalId item_lid,ItemLocalId old_cell_lid,
                   ItemLocalId new_cell_lid)
{
  ItemInternal* ii = m_family->itemInfoListView()[item_lid].internal();
  Int32 index = _getItemIndex(ii->internalCells(),old_cell_lid);
  this->replaceCell(ItemLocalId(ii->localId()),index,new_cell_lid);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
