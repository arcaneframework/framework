// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ParticleFamilySerializer.cc                                 (C) 2000-2024 */
/*                                                                           */
/* Sérialisation/Désérialisation des familles de particules.                 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ISerializer.h"
#include "arcane/core/ItemPrinter.h"
#include "arcane/core/IMesh.h"
#include "arcane/core/IParallelMng.h"

#include "arcane/mesh/ParticleFamilySerializer.h"
#include "arcane/mesh/ParticleFamily.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ParticleFamilySerializer::
ParticleFamilySerializer(ParticleFamily* family)
: TraceAccessor(family->traceMng())
, m_family(family)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParticleFamilySerializer::
serializeItems(ISerializer* sbuf,Int32ConstArrayView local_ids)
{
  const Integer nb_item = local_ids.size();
  ItemInfoListView items_internal(m_family);

  switch(sbuf->mode()){
  case ISerializer::ModeReserve:
    sbuf->reserveInt64(1); // Pour le nombre de particules
    sbuf->reserveSpan(eBasicDataType::Int64,nb_item); // Pour les uniqueId() des particules.
    sbuf->reserveSpan(eBasicDataType::Int64,nb_item); // Pour les uniqueId() des mailles dans lesquelles se trouve les particules
    break;
  case ISerializer::ModePut:
    sbuf->putInt64(nb_item);
    {
      Int64UniqueArray particle_unique_ids(nb_item);
      for( Integer z=0; z<nb_item; ++z ){
        particle_unique_ids[z] = items_internal.uniqueId(local_ids[z]).asInt64();
      }
      sbuf->putSpan(particle_unique_ids);
    }
    {
      Int64UniqueArray particles_cell_uid(nb_item);
      for( Integer z=0; z<nb_item; ++z ){
        Particle item(items_internal[local_ids[z]]);
        bool has_cell = item.hasCell();
        particles_cell_uid[z] = (has_cell) ? item.cell().uniqueId() : NULL_ITEM_UNIQUE_ID;
      }
      sbuf->putSpan(particles_cell_uid);
    }
    break;
  case ISerializer::ModeGet:
    deserializeItems(sbuf,nullptr);
    break;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParticleFamilySerializer::
deserializeItems(ISerializer* sbuf,Int32Array* local_ids)
{
  // NOTE: les mailles doivent avoir été désérialisées avant.
  Int64UniqueArray particles_uid;
  Int64UniqueArray cells_unique_id;
  Int32UniqueArray cells_local_id;
  Int32UniqueArray particles_owner;
  IMesh* mesh = m_family->mesh();
  IItemFamily* cell_family = mesh->cellFamily();
  CellInfoListView internal_cells(cell_family);

  Int64 nb_item = sbuf->getInt64();
  particles_uid.resize(nb_item);
  sbuf->getSpan(particles_uid);
  cells_unique_id.resize(nb_item);
  cells_local_id.resize(nb_item);
  sbuf->getSpan(cells_unique_id);
  Int32UniqueArray temporary_particles_local_id;
  Int32Array* particles_local_id = (local_ids) ? local_ids : &temporary_particles_local_id;
  particles_local_id->resize(nb_item);
  Int32ArrayView local_ids_view = particles_local_id->view();
  cell_family->itemsUniqueIdToLocalId(cells_local_id,cells_unique_id,true);

  // Si on gère les particules fantômes, alors les particules ont un propriétaire
  // et dans ce cas il faut créér les particules avec cette information.
  // On suppose alors que le propriétaire d'une particule est la maille dans laquelle
  // elle se trouve.
  // NOTE: dans la version actuelle, le support des particules fantômes implique
  // que ces dernières aient une table de hashage pour les uniqueId()
  // (c'est à dire particle_family->hasUniqueIdMap() == true).
  if (!m_family->getEnableGhostItems()){
    m_family->addParticles(particles_uid,local_ids_view);
  }
  else{
    particles_owner.resize(nb_item) ;
    for( Integer zz=0; zz<nb_item; ++zz ){
      Int32 cell_lid = cells_local_id[zz];
      if (cell_lid!=NULL_ITEM_LOCAL_ID){
        Cell c = internal_cells[ cell_lid ];
        particles_owner[zz] = c.owner() ;
      }
      else
        particles_owner[zz] = NULL_SUB_DOMAIN_ID;
    }

    m_family->addParticles2(particles_uid,particles_owner,local_ids_view);
  }

  // IMPORTANT: il faut le faire ici car cela peut changer via le endUpdate()
  ItemInternalList internal_particles(m_family->itemsInternal());
  for( Integer zz=0; zz<nb_item; ++zz ){
    Particle p = internal_particles[ local_ids_view[zz] ];
    Int32 cell_lid = cells_local_id[zz];
    if (cell_lid!=NULL_ITEM_LOCAL_ID){
      Cell c = internal_cells[ cell_lid ];
      m_family->setParticleCell(p,c);
    }
    else
      m_family->setParticleCell(p,Cell());
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IItemFamily* ParticleFamilySerializer::
family() const
{
  return m_family;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
