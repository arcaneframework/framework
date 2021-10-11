// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemInternal.cc                                             (C) 2000-2020 */
/*                                                                           */
/* Partie interne d'une entité.                                              */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/ItemInternal.h"

#include "arcane/utils/NotImplementedException.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/Iostream.h"
#include "arcane/utils/TraceInfo.h"

#include "arcane/IItemFamily.h"
#include "arcane/IItemFamilyTopologyModifier.h"
#include "arcane/ItemPrinter.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemInternalConnectivityList ItemInternalConnectivityList::nullInstance;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Recherche l'index de l'entité de localid() \a local_id dans
 * la connectivité.
 */
inline Integer ItemInternal::
_getItemIndex(const Int32* items,Integer nb_item,Int32 local_id)
{
  for( Integer i=0; i<nb_item; ++i )
    if (items[i] == local_id)
      return i;
  ARCANE_FATAL("Can not find item to replace local_id={0}",local_id);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline void ItemInternal::
_replaceItem(Int32* items,Integer nb_item,
             Int32 old_local_id,Int32 new_local_id,eItemKind item_kind)
{
  for( Integer i=0; i<nb_item; ++i )
    if (items[i] == old_local_id){
      items[i] = new_local_id;
      return;
    }
  ARCANE_FATAL("Can not find item to replace kind={0} local_id={1}",
               itemKindName(item_kind),old_local_id);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemInternal::
_checkUniqueId(Int64 new_uid) const
{
  if (new_uid<0){
    ARCANE_FATAL("Bad unique id - new_uid={0} local_id={1} current={2}",
                 new_uid,m_local_id,uniqueId().asInt64());
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemInternal::
unsetUniqueId()
{
#ifdef ARCANE_CHECK
  _checkUniqueId((*m_shared_info->m_unique_ids)[m_local_id]);
#endif
  (*m_shared_info->m_unique_ids)[m_local_id] = NULL_ITEM_UNIQUE_ID;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 ItemUniqueId::
asInt32() const
{
  if (m_unique_id>2147483647)
    ARCANE_FATAL("Unique id is too big to be converted to a Int32");
  return (Int32)(m_unique_id);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

std::ostream&
operator<<(std::ostream& o,const ItemUniqueId& id)
{
  o << id.asInt64();
  return o;
}

//! AMR
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const ItemInternal* ItemInternal::
topHParent() const
{
  const ItemInternal* top_it = this;
  while (top_it->nbHParent())
    top_it = top_it->internalHParent(0);
  ARCANE_ASSERT((!top_it->null()),("topHParent Problem!"));
  ARCANE_ASSERT((top_it->level() == 0),("topHParent Problem"));
  return top_it;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemInternal* ItemInternal::
topHParent()
{
  ItemInternal* top_it = this;
  while (top_it->nbHParent())
    top_it = top_it->internalHParent(0);
  ARCANE_ASSERT((!top_it->null()),("topHParent Problem!"));
  ARCANE_ASSERT((top_it->level() == 0),("topHParent Problem"));
  return top_it;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 ItemInternal::
whichChildAmI(const ItemInternal *iitem) const
{
  ARCANE_ASSERT((this->hasHChildren()), ("item has non-child!"));
  for (Integer c=0; c<this->nbHChildren(); c++)
    if (this->internalHChild(c) == iitem)
      return c;
  return -1;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
ItemInternalVectorView ItemInternal::
activeCells(Int32Array& local_ids) const
{
	const Integer nbcell = this->nbCell();
	for(Integer icell = 0 ; icell < nbcell ; ++icell) {
    ItemInternal* cell   = this->internalCell(icell);
    if (cell->isActive()){
      const Int32 local_id= cell->localId();
      local_ids.add(local_id);
    }
	}
	return ItemInternalVectorView(m_shared_info->m_items->cells,local_ids);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemInternalVectorView ItemInternal::
activeFaces(Int32Array& local_ids) const
{
	const Integer nbface = this->nbFace();
	for(Integer iface = 0 ; iface < nbface ; ++iface) {
		ItemInternal* face   = this->internalFace(iface);
		if (!face->isBoundary()){
			ItemInternal* bcell = face->backCell();
      ItemInternal* fcell = face->frontCell();
			if ( (bcell && bcell->isActive()) && (fcell && fcell->isActive()) )
				local_ids.add(face->localId());
		}
		else {
			ItemInternal* bcell = face->boundaryCell();
			if ( (bcell && bcell->isActive()) )
				local_ids.add(face->localId());
		}
	}
	return ItemInternalVectorView(m_shared_info->m_items->faces,local_ids);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemInternalVectorView ItemInternal::
activeEdges() const
{
	throw NotImplementedException(A_FUNCINFO,
                                "Active edges group not yet implemented");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Indique si le genre \a item_kind utilise IItemFamilyTopologyModifier
inline bool ItemInternal::
_useTopologyModifier() const
{
  eItemKind item_kind = this->sharedInfo()->itemKind();
  return (item_kind==IK_Node || item_kind==IK_Edge || item_kind==IK_Face ||
          item_kind==IK_Cell);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemInternal::
_setNodeWithLocalId(Integer aindex,Int32 new_local_id)
{
  if (_useTopologyModifier()){
    auto x = m_shared_info->itemFamily()->_topologyModifier();
    x->replaceNode(ItemLocalId(m_local_id),aindex,ItemLocalId(new_local_id));
  }
  else
    _setNode(aindex,new_local_id);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemInternal::
_setEdgeWithLocalId(Integer aindex,Int32 new_local_id)
{
  if (_useTopologyModifier()){
    auto x = m_shared_info->itemFamily()->_topologyModifier();
    x->replaceEdge(ItemLocalId(m_local_id),aindex,ItemLocalId(new_local_id));
  }
  else
    _setEdge(aindex,new_local_id);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemInternal::
_setFaceWithLocalId(Integer aindex,Int32 new_local_id)
{
  if (_useTopologyModifier()){
    auto x = m_shared_info->itemFamily()->_topologyModifier();
    x->replaceFace(ItemLocalId(m_local_id),aindex,ItemLocalId(new_local_id));
  }
  else
    _setFace(aindex,new_local_id);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemInternal::
_setCellWithLocalId(Integer aindex,Int32 new_local_id)
{
  if (_useTopologyModifier()){
    auto x = m_shared_info->itemFamily()->_topologyModifier();
    x->replaceCell(ItemLocalId(m_local_id),aindex,ItemLocalId(new_local_id));
  }
  else{
    _setCell(aindex,new_local_id);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemInternal::
_setHParentWithLocalId(Integer aindex,Int32 new_local_id)
{
  if (_useTopologyModifier()){
    auto x = m_shared_info->itemFamily()->_topologyModifier();
    x->replaceHParent(ItemLocalId(m_local_id),aindex,ItemLocalId(new_local_id));
  }
  else{
    _setHParent(aindex,new_local_id);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemInternal::
_setHChildWithLocalId(Integer aindex,Int32 new_local_id)
{
  if (_useTopologyModifier()){
    auto x = m_shared_info->itemFamily()->_topologyModifier();
    x->replaceHChild(ItemLocalId(m_local_id),aindex,ItemLocalId(new_local_id));
  }
  else{
    _setHChild(aindex,new_local_id);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemInternal::
_setFaceInfos(Int32 cell0,Int32 cell1,Int32 mod_flags)
{
  Int32 face_flags = flags();
  face_flags &= ~II_InterfaceFlags;
  face_flags |= mod_flags;
  setFlags(face_flags);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemInternal::
_setFaceBackAndFrontCells(Int32 back_cell_lid,Int32 front_cell_lid)
{
  if (front_cell_lid==NULL_ITEM_LOCAL_ID){
    // Reste uniquement la back_cell ou aucune maille.
    Int32 mod_flags = (back_cell_lid!=NULL_ITEM_LOCAL_ID) ? (II_Boundary | II_HasBackCell | II_BackCellIsFirst) : 0;
    _setFaceInfos(back_cell_lid,NULL_ITEM_LOCAL_ID,mod_flags);
  }
  else if (back_cell_lid==NULL_ITEM_LOCAL_ID){
    // Reste uniquement la front cell
    _setFaceInfos(front_cell_lid,NULL_ITEM_LOCAL_ID,
                  II_Boundary | II_HasFrontCell | II_FrontCellIsFirst);
  }
  else{
    // Il y a deux mailles connectées.
    _setFaceInfos(back_cell_lid,front_cell_lid,
                  II_HasFrontCell | II_HasBackCell | II_BackCellIsFirst);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemInternal::
_internalCopyAndChangeSharedInfos(ItemSharedInfo* old_isi,ItemSharedInfo* new_isi,Integer new_data_index)
{
  ItemInternal* item = this;

  Integer old_flags = item->flags();
  Integer old_owner = item->owner();
  Int32* old_data = item->dataPtr();

  item->setSharedInfo(new_isi);
  item->setDataIndex(new_data_index);

  new_isi->setFlags(new_data_index,old_flags);
  new_isi->setOwner(new_data_index,old_owner);

  Int32* new_data = item->dataPtr();

  ::memcpy(new_data+new_isi->firstNode(),old_data+old_isi->firstNode(),old_isi->nbNode()*sizeof(Int32));
  ::memcpy(new_data+new_isi->firstEdge(),old_data+old_isi->firstEdge(),old_isi->nbEdge()*sizeof(Int32));
  ::memcpy(new_data+new_isi->firstFace(),old_data+old_isi->firstFace(),old_isi->nbFace()*sizeof(Int32));
  ::memcpy(new_data+new_isi->firstCell(),old_data+old_isi->firstCell(),old_isi->nbCell()*sizeof(Int32));
  ::memcpy(new_data+new_isi->firstParent(),old_data+old_isi->firstParent(),old_isi->nbParent()*sizeof(Int32));
  //! AMR
  ::memcpy(new_data+new_isi->firstHParent(),old_data+old_isi->firstHParent(),old_isi->nbHParent()*sizeof(Int32));
  ::memcpy(new_data+new_isi->firstHChild(),old_data+old_isi->firstHChild(),old_isi->nbHChildren()*sizeof(Int32));
  // OFF AMR
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemInternal::
_internalCopyAndSetDataIndex(Int32* data_ptr,Int32 data_index)
{
  Integer nb = neededMemory();
  ::memcpy(data_ptr + data_index,this->dataPtr(),nb*sizeof(Int32));
  setDataIndex(data_index);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemInternal::
_checkValidConnectivity(ItemInternal* item,Int32 nb_sub_item,
                        const Int32* ref_ptr,const Int32* new_ptr,
                        Int32 sub_item_kind)
{
  for( Integer z=0; z<nb_sub_item; ++z ){
    Int32 ref_lid = ref_ptr[z];
    Int32 new_lid = new_ptr[z];
    if (ref_lid!=new_lid)
      ARCANE_FATAL("Incoherent connected items index={0} ref_lid={1} new_lid={2}"
                   " item={3} sub_item_kind={4}",
                   z,ref_lid,new_lid,ItemPrinter(item),sub_item_kind);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemInternal::
_internalCheckValidConnectivityAccessor(ItemInternalConnectivityList* iicl)
{
  ItemInternal* item = this;
  ItemLocalId lid(item->localId());
  Integer nb_sub_node = item->nbNode();
  if (nb_sub_node!=0)
    _checkValidConnectivity(item,nb_sub_node,item->_nodesPtr(),
                            iicl->_nodeLocalIdsV2(lid),
                            ItemInternalConnectivityList::NODE_IDX);
  Integer nb_sub_edge = item->nbEdge();
  if (nb_sub_edge!=0)
    _checkValidConnectivity(item,nb_sub_edge,item->_edgesPtr(),
                            iicl->_edgeLocalIdsV2(lid),
                            ItemInternalConnectivityList::EDGE_IDX);
  Integer nb_sub_face = item->nbFace();
  if (nb_sub_face!=0)
    _checkValidConnectivity(item,nb_sub_face,item->_facesPtr(),
                            iicl->_faceLocalIdsV2(lid),
                            ItemInternalConnectivityList::FACE_IDX);
  Integer nb_sub_cell = item->nbCell();
  if (nb_sub_cell!=0){
    _checkValidConnectivity(item,nb_sub_cell,item->_cellsPtr(),
                            iicl->_cellLocalIdsV2(lid),
                            ItemInternalConnectivityList::CELL_IDX);
  }
  Integer nb_sub_hparent = item->nbHParent();
  if (nb_sub_hparent!=0){
    _checkValidConnectivity(item,nb_sub_hparent,item->_hParentPtr(),
                            iicl->_hParentLocalIdsV2(lid),
                            ItemInternalConnectivityList::HPARENT_IDX);
  }
  Integer nb_sub_hchild = item->nbHChildren();
  if (nb_sub_hchild!=0){
    _checkValidConnectivity(item,nb_sub_hchild,item->_hChildPtr(),
                            iicl->_hChildLocalIdsV2(lid),
                            ItemInternalConnectivityList::HCHILD_IDX);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
