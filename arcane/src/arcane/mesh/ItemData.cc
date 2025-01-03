// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemData.cc                                                 (C) 2000-2024 */
/*                                                                           */
/* Class gathering item data : ids and connectivities                        */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/CheckedConvert.h"

#include "arcane/core/IParallelMng.h"
#include "arcane/core/MeshPartInfo.h"

#include "arcane/mesh/ItemData.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemData::
serialize(ISerializer* buffer)
{
  switch(buffer->mode()){
  case ISerializer::ModeReserve:
    {
      buffer->reserve(m_item_family->name());
      buffer->reserve(itemKindName(m_item_family->itemKind()));
      buffer->reserveInt64(2); // nb_items + item_infos.size
      buffer->reserveSpan(eBasicDataType::Int32,m_nb_items);// m_owners
      buffer->reserveSpan(eBasicDataType::Int64,m_item_infos.size());
    }
    break;
  case ISerializer::ModePut:
    {
      buffer->put(m_item_family->name());
      buffer->put(itemKindName(m_item_family->itemKind()));
      buffer->putInt64(m_nb_items);
      buffer->putInt64(m_item_infos.size());
      buffer->putSpan(m_item_owners);
      buffer->putSpan(m_item_infos);
    }
    break;
  case ISerializer::ModeGet:
    {
      deserialize(buffer,m_item_family->mesh());
    }
    break;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemData::
deserialize(ISerializer* buffer, IMesh* mesh)
{
  _deserialize(buffer,mesh);
  // use internal lids array
  _internal_item_lids.resize(m_nb_items);
  m_item_lids = _internal_item_lids;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemData::
deserialize(ISerializer* buffer, IMesh* mesh, Int32Array& item_lids)
{
  _deserialize(buffer,mesh);
  item_lids.resize(m_nb_items);
  m_item_lids = item_lids;
}


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemData::
_deserialize(ISerializer* buffer, IMesh* mesh)
{
  ARCANE_ASSERT((buffer->mode() == ISerializer::ModeGet),
                ("Impossible to deserialize a buffer not in ModeGet. In ItemData::deserialize.Exiting"))
  String family_name;
  buffer->get(family_name);
  String item_kind_name;
  buffer->get(item_kind_name);
  std::istringstream iss(item_kind_name.localstr());
  eItemKind family_kind;
  iss >> family_kind;
  m_item_family = mesh->findItemFamily(family_kind,family_name,false);
  m_item_family_modifier = mesh->findItemFamilyModifier(family_kind,family_name);
  m_nb_items = CheckedConvert::toInt32(buffer->getInt64());
  m_item_owners.resize(m_nb_items);
  m_item_infos.resize(buffer->getInt64());
  buffer->getSpan(m_item_owners);
  buffer->getSpan(m_item_infos);
  m_subdomain_id = mesh->meshPartInfo().partRank();
//  mesh->traceMng()->debug() << " DESERIALIZE " << m_item_owners;
//  mesh->traceMng()->debug() << " DESERIALIZE " << m_item_infos;
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
