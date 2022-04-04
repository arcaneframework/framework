﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
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

void ItemInternal::
_setFaceInfos(Int32 mod_flags)
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
    _setFaceInfos(mod_flags);
  }
  else if (back_cell_lid==NULL_ITEM_LOCAL_ID){
    // Reste uniquement la front cell
    _setFaceInfos(II_Boundary | II_HasFrontCell | II_FrontCellIsFirst);
  }
  else{
    // Il y a deux mailles connectées.
    _setFaceInfos(II_HasFrontCell | II_HasBackCell | II_BackCellIsFirst);
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

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
