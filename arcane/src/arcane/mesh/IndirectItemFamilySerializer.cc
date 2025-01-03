// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IndirectItemFamilySerializer.cc                             (C) 2000-2024 */
/*                                                                           */
/* Sérialisation/Désérialisation indirecte des familles d'entités.           */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ISerializer.h"
#include "arcane/core/IItemFamily.h"
#include "arcane/core/ItemInfoListView.h"
#include "arcane/core/Item.h"

#include "arcane/mesh/IndirectItemFamilySerializer.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IndirectItemFamilySerializer::
IndirectItemFamilySerializer(IItemFamily* family)
: TraceAccessor(family->traceMng())
, m_family(family)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void IndirectItemFamilySerializer::
serializeItems(ISerializer* sbuf,Int32ConstArrayView local_ids)
{
  const Integer nb_item = local_ids.size();
  ItemInfoListView items_internal(m_family);

  switch(sbuf->mode()){
  case ISerializer::ModeReserve:
    sbuf->reserveInt64(1); // Pour le nombre d'entités
    sbuf->reserveSpan(eBasicDataType::Int64,nb_item); // Pour les uniqueId() des entités.
    break;
  case ISerializer::ModePut:
    sbuf->putInt64(nb_item);
    {
      Int64UniqueArray particle_unique_ids(nb_item);
      for( Integer z=0; z<nb_item; ++z ){
        Item item = items_internal[ local_ids[z] ];
        particle_unique_ids[z] = item.uniqueId().asInt64();
      }
      sbuf->putSpan(particle_unique_ids);
    }
    break;
  case ISerializer::ModeGet:
    deserializeItems(sbuf,nullptr);
    break;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void IndirectItemFamilySerializer::
deserializeItems(ISerializer* sbuf,Int32Array* local_ids)
{
  Int64UniqueArray unique_ids;

  Int64 nb_item = sbuf->getInt64();
  unique_ids.resize(nb_item);
  sbuf->getSpan(unique_ids);
  
  Int32UniqueArray temporary_local_ids;
  Int32Array* work_local_id = (local_ids) ? local_ids : &temporary_local_ids;
  work_local_id->resize(nb_item);

  m_family->itemsUniqueIdToLocalId(*work_local_id,unique_ids,true);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IItemFamily* IndirectItemFamilySerializer::
family() const
{
  return m_family;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
