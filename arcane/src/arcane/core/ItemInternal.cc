// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemInternal.cc                                             (C) 2000-2025 */
/*                                                                           */
/* Partie interne d'une entité.                                              */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ItemInternal.h"

#include "arcane/utils/NotImplementedException.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/Iostream.h"
#include "arcane/utils/TraceInfo.h"
#include "arcane/utils/CheckedConvert.h"

#include "arcane/core/IItemFamily.h"
#include "arcane/core/IItemFamilyTopologyModifier.h"
#include "arcane/core/ItemPrinter.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemInternalConnectivityList ItemInternalConnectivityList::nullInstance;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void impl::MutableItemBase::
_checkUniqueId(Int64 new_uid) const
{
  if (new_uid<0){
    ARCANE_FATAL("Bad unique id - new_uid={0} local_id={1} current={2}",
                 new_uid,m_local_id,uniqueId().asInt64());
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void impl::MutableItemBase::
unsetUniqueId()
{
#ifdef ARCANE_CHECK
  _checkUniqueId(m_shared_info->m_unique_ids[m_local_id]);
#endif
  m_shared_info->m_unique_ids[m_local_id] = NULL_ITEM_UNIQUE_ID;
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

ItemTypeId ItemTypeId::
fromInteger(Int64 v)
{
  Int16 x = CheckedConvert::toInt16(v);
  return ItemTypeId{x};
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 impl::ItemBase::
_nbLinearNode() const
{
  ItemTypeInfo* iti = typeInfo();
  Int16 linear_type_id = iti->linearTypeId();
  ItemTypeInfo* linear_iti = m_shared_info->typeInfoFromId(linear_type_id);
  Int32 nb_linear_node = linear_iti->nbLocalNode();
  return std::min(nbNode(), nb_linear_node);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

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
    top_it = top_it->_internalHParent(0);
  ARCANE_ASSERT((!top_it->null()),("topHParent Problem!"));
  ARCANE_ASSERT((top_it->level() == 0),("topHParent Problem"));
  return top_it;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

impl::ItemBase impl::ItemBase::
topHParentBase() const
{
  ItemBase top_it = *this;
  while (top_it.nbHParent())
    top_it = top_it.hParentBase(0);
  ARCANE_ASSERT((!top_it.null()),("topHParent Problem!"));
  ARCANE_ASSERT((top_it.level() == 0),("topHParent Problem"));
  return top_it;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemInternal* ItemInternal::
topHParent()
{
  ItemInternal* top_it = this;
  while (top_it->nbHParent())
    top_it = top_it->_internalHParent(0);
  ARCANE_ASSERT((!top_it->null()),("topHParent Problem!"));
  ARCANE_ASSERT((top_it->level() == 0),("topHParent Problem"));
  return top_it;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 impl::ItemBase::
whichChildAmI(Int32 local_id) const
{
  ARCANE_ASSERT((this->hasHChildren()), ("item has non-child!"));
  for (Integer c=0; c<this->nbHChildren(); c++)
    if (this->hChildId(c) == local_id)
      return c;
  return -1;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 ItemInternal::
whichChildAmI(const ItemInternal *iitem) const
{
  ARCANE_ASSERT((this->hasHChildren()), ("item has non-child!"));
  for (Integer c=0; c<this->nbHChildren(); c++)
    if (this->_internalHChild(c) == iitem)
      return c;
  return -1;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemInternalVectorView impl::ItemBase::
_internalActiveCells2(Int32Array& local_ids) const
{
  const Integer nbcell = this->nbCell();
  for(Integer icell = 0 ; icell < nbcell ; ++icell) {
    ItemBase cell = this->cellBase(icell);
    if (cell.isActive()){
      const Int32 local_id = cell.localId();
      local_ids.add(local_id);
    }
  }
  return ItemInternalVectorView(m_shared_info->m_items->m_cell_shared_info,local_ids,0);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void impl::MutableItemBase::
_setFaceInfos(Int32 mod_flags)
{
  Int32 face_flags = flags();
  face_flags &= ~II_InterfaceFlags;
  face_flags |= mod_flags;
  setFlags(face_flags);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Pour une face, positionne à la fois la back cell et la front cell.
 *
 * \a back_cell_lid et/ou \a front_cell_lid peuvent valoir NULL_ITEM_LOCAL_ID
 * ce qui signifie que l'entité n'a pas de back cell ou front cell. Si les
 * deux valeurs sont nulles, alors la face est considérée comme n'ayant
 * plus de mailles connectées.
 *
 * \note Cette méthode est utilisée uniquement par FaceFamily.
 */
void impl::MutableItemBase::
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
setDataIndex(Integer)
{
  ARCANE_FATAL("This method is no longer valid");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool ItemInternalVectorView::
_isValid()
{
  if (!m_shared_info)
    ARCANE_FATAL("Null ItemSharedInfo");
  if (!m_local_ids.empty()){
    auto* items_data = _items().data();
    if (!items_data)
      ARCANE_FATAL("Null ItemsInternal list");
  }
  return true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
