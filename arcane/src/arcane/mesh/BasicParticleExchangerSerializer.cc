// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* BasicParticleExchangerSerializer.cc                         (C) 2000-2021 */
/*                                                                           */
/* Echangeur de particules.                                                  */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/mesh/BasicParticleExchangerSerializer.h"

#include "arcane/utils/Array.h"

#include "arcane/IItemFamily.h"
#include "arcane/IMesh.h"
#include "arcane/IParticleFamily.h"
#include "arcane/ISerializeMessage.h"
#include "arcane/ISerializer.h"
#include "arcane/Item.h"
#include "arcane/ItemGroup.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

BasicParticleExchangerSerializer::
BasicParticleExchangerSerializer(IItemFamily* family,Int32 my_rank)
: TraceAccessor(family->traceMng())
, m_item_family(family)
, m_my_rank(my_rank)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

BasicParticleExchangerSerializer::
~BasicParticleExchangerSerializer()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicParticleExchangerSerializer::
beginNewExchange()
{
  // Récupère la liste des variables à transférer.
  // Il s'agit des variables qui ont la même famille que celle passée
  // en paramètre.
  // IMPORTANT: tous les sous-domaines doivent avoir ces mêmes variables
  m_variables_to_exchange.clear();
  m_item_family->usedVariables(m_variables_to_exchange);
  m_variables_to_exchange.sortByName(true);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void BasicParticleExchangerSerializer::
serializeMessage(WorkBuffer& work_buffer,
                 ISerializeMessage* sm,
                 Int32ConstArrayView acc_ids)
{
  Int64Array& items_to_send_uid = work_buffer.items_to_send_uid;
  Int64Array& items_to_send_cells_uid = work_buffer.items_to_send_cells_uid;

  ItemInternalList internal_items(m_item_family->itemsInternal());

  ISerializer* sbuf = sm->serializer();
  sbuf->setMode(ISerializer::ModeReserve);


  Integer nb_item = acc_ids.size();
  // Réserve pour l'id du message
  sbuf->reserve(DT_Int64,1);
  // Réserve pour le nombre de uniqueId()
  sbuf->reserve(DT_Int64,1);
  sbuf->reserveSpan(DT_Int64,nb_item);
  // Réserve pour les uniqueId() des mailles dans lesquelles se trouvent les particules
  sbuf->reserveSpan(DT_Int64,nb_item);

  for( VariableList::Enumerator i_var(m_variables_to_exchange); ++i_var; ){
    IVariable* var = *i_var;
    var->serialize(sbuf,acc_ids);
  }

  // Sérialise les données en écriture
  sbuf->allocateBuffer();

  if (m_debug_exchange_items_level>=1)
    info() << "BSE_SerializeMessage nb_item=" << nb_item
           << " id=" << m_serialize_id
           << " dest=" << sm->destination();

  sbuf->setMode(ISerializer::ModePut);
  
  sbuf->putInt64(m_serialize_id);
  ++m_serialize_id;

  sbuf->putInt64(nb_item);
  items_to_send_uid.resize(nb_item);
  items_to_send_cells_uid.resize(nb_item);
  for( Integer z=0; z<nb_item; ++z ){
    Particle item = internal_items[acc_ids[z]];
    items_to_send_uid[z] = item.uniqueId();
    bool has_cell = item.hasCell();
    items_to_send_cells_uid[z] = (has_cell) ? item.cell().uniqueId() : NULL_ITEM_UNIQUE_ID;
    if (m_debug_exchange_items_level>=2){
      info() << "Particle BufID=" << acc_ids[z]
             << " LID=" << item.localId()
             << " UID=" << items_to_send_uid[z]
             << " CellIUID=" << items_to_send_cells_uid[z]
             << " (owner=" << item.cell().owner() << ")";
    }
  }
  sbuf->putSpan(items_to_send_uid);
  sbuf->putSpan(items_to_send_cells_uid);

  for( VariableList::Enumerator i_var(m_variables_to_exchange); ++i_var; ){
    IVariable* var = *i_var;
    var->serialize(sbuf,acc_ids);
  }

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 BasicParticleExchangerSerializer::
deserializeMessage(WorkBuffer& work_buffer,ISerializeMessage* message,
                   ItemGroup item_group, Int32Array* new_particle_local_ids)
{
  Int64Array& items_to_create_unique_id = work_buffer.items_to_create_unique_id;
  Int64Array& items_to_create_cells_unique_id = work_buffer.items_to_create_cells_unique_id;
  Int32Array& items_to_create_local_id = work_buffer.items_to_create_local_id;
  Int32Array& items_to_create_cells_local_id = work_buffer.items_to_create_cells_local_id;

  IMesh* mesh = m_item_family->mesh();
  ISerializer* sbuf = message->serializer();
  IItemFamily* cell_family = mesh->cellFamily();

  // Indique qu'on souhaite sérialiser les données en lecture
  sbuf->setMode(ISerializer::ModeGet);
  sbuf->setReadMode(ISerializer::ReadReplace);

  {
    Int64 serialize_id = sbuf->getInt64();
    Int64 nb_item = sbuf->getInt64();
    if (m_debug_exchange_items_level>=1)
      info() << "BSE_DeserializeMessage id=" << serialize_id << " nb=" << nb_item
             << " orig=" << message->destination();

    items_to_create_local_id.resize(nb_item);
    items_to_create_unique_id.resize(nb_item);
    items_to_create_cells_unique_id.resize(nb_item);
    items_to_create_cells_local_id.resize(nb_item);
    sbuf->getSpan(items_to_create_unique_id);
    sbuf->getSpan(items_to_create_cells_unique_id);
    if (m_debug_exchange_items_level>=2){
      //info() << "Recv from SID " << sync_infos[i].subDomain() << " N=" << nb_item;
      for( Integer z=0; z<nb_item; ++z ){
        info() << "Particle UID=" << items_to_create_unique_id[z]
               << " CellIUID=" << items_to_create_cells_unique_id[z];
      }
    }

    items_to_create_cells_local_id.resize(nb_item);
    cell_family->itemsUniqueIdToLocalId(items_to_create_cells_local_id,items_to_create_cells_unique_id);

    m_item_family->toParticleFamily()->addParticles(items_to_create_unique_id,
                                                    items_to_create_cells_local_id,
                                                    items_to_create_local_id);

    // Notifie la famille qu'on a fini nos modifs.
    // Après appel à cette méthode, les variables sont à nouveau utilisables
    m_item_family->endUpdate();
    
    // Converti les uniqueId() récupérée en localId() et pour les particules
    // renseigne la maille correspondante
    ItemInternalList internal_items(m_item_family->itemsInternal());
      
    for( Integer z=0; z<nb_item; ++z ){
      Particle item = internal_items[items_to_create_local_id[z]];
      //item.setCell( internal_cells[items_to_create_cells_local_id[z]] );
      // Je suis le nouveau propriétaire (TODO: ne pas faire ici)
      item.internal()->setOwner(m_my_rank,m_my_rank);
    }
    if (!item_group.null())
      item_group.addItems(items_to_create_local_id,false);
    if (new_particle_local_ids)
      new_particle_local_ids->addRange(items_to_create_local_id);
    for( VariableCollection::Enumerator i_var(m_variables_to_exchange); ++i_var; ){
      IVariable* var = *i_var;
      var->serialize(sbuf,items_to_create_local_id);
    }
  }

  return items_to_create_unique_id.size();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
