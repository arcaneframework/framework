// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ExtraGhostParticlesBuilder.cc                               (C) 2000-2024 */
/*                                                                           */
/* Construction des mailles fantômes supplémentaires.                        */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/mesh/ExtraGhostParticlesBuilder.h"
#include "arcane/mesh/DynamicMesh.h"

#include "arcane/utils/ScopedPtr.h"

#include "arcane/IExtraGhostParticlesBuilder.h"
#include "arcane/IParallelExchanger.h"
#include "arcane/IParallelMng.h"
#include "arcane/ISerializeMessage.h"
#include "arcane/SerializeBuffer.h"
#include "arcane/ParallelMngUtils.h"

#include "arcane/mesh/ParticleFamily.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ExtraGhostParticlesBuilder::
ExtraGhostParticlesBuilder(DynamicMesh* mesh)
: TraceAccessor(mesh->traceMng())
, m_mesh(mesh)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ExtraGhostParticlesBuilder::
addExtraGhostParticlesBuilder(IExtraGhostParticlesBuilder* builder)
{
  if (m_builders.contains(builder))
    ARCANE_FATAL("Instance {0} is already registered",builder);
  m_builders.add(builder);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ExtraGhostParticlesBuilder::
removeExtraGhostParticlesBuilder(IExtraGhostParticlesBuilder* builder)
{
  auto iter_begin = m_builders.begin();
  auto iter_end = m_builders.end();
  auto x = std::find(iter_begin,iter_end,builder);
  if (x==iter_end)
    ARCANE_FATAL("Instance {0} is not registered",builder);
  m_builders.remove(x - iter_begin);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool ExtraGhostParticlesBuilder::
hasBuilder() const
{
  return !m_builders.empty();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ExtraGhostParticlesBuilder::
computeExtraGhostParticles()
{
  const size_t nb_builder = m_builders.size();
  
  if (nb_builder == 0)
    return;
  
  info() << "Compute extra ghost particles";
  
  for( IExtraGhostParticlesBuilder* v : m_builders ){
    // Calcul de mailles extraordinaires à envoyer
    v->computeExtraParticlesToSend();
  }
  
  for( IItemFamilyCollection::Enumerator i(m_mesh->itemFamilies()); ++i; ) {
    IItemFamily* family = *i;
    if (family->itemKind()!=IK_Particle)
      continue;
    ParticleFamily* particle_family = ARCANE_CHECK_POINTER(dynamic_cast<ParticleFamily*>(family));
    if (particle_family && particle_family->getEnableGhostItems()==true){
      _computeForFamily(particle_family);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ExtraGhostParticlesBuilder::
_computeForFamily(ParticleFamily* particle_family)
{
  IParallelMng* pm = particle_family->itemFamily()->parallelMng();
  const Int32 nsd = pm->commRank();

  auto exchanger { ParallelMngUtils::createExchangerRef(pm) };

  // Construction des entités à envoyer
  UniqueArray<std::set<Integer> > to_sends(nsd);

  // Initialisation de l'échangeur de données
  for(Integer isd=0;isd<nsd;++isd){
    std::set<Integer>& particle_set = to_sends[isd];
    for( IExtraGhostParticlesBuilder* builder : m_builders ){
      Int32ConstArrayView extra_particles = builder->extraParticlesToSend(particle_family->name(),isd);
      // On trie les lids à envoyer pour éviter les doublons
      for(Integer j=0, size=extra_particles.size(); j<size; ++j)
        particle_set.insert(extra_particles[j]);
    }
    if (!particle_set.empty())
      exchanger->addSender(isd);
  }
  exchanger->initializeCommunicationsMessages();


  ItemInternalList items_internal(particle_family->itemsInternal());
  for( Integer i=0, ns=exchanger->nbSender(); i<ns; ++i){
    ISerializeMessage* comm = exchanger->messageToSend(i);
    const Int32 rank = comm->destination().value();
    //ISerializer* s = comm->serializer();
    const std::set<Integer>& particle_set = to_sends[rank];
    Int32UniqueArray dest_items_local_id(particle_set.size());
    std::copy(std::begin(particle_set), std::end(particle_set), std::begin(dest_items_local_id));
    //particle_family->serializeParticles(s, items_to_send);

    Int64 nb_item = particle_set.size();
    Int64UniqueArray dest_items_unique_id(nb_item);
    for( Integer z=0; z<nb_item; ++z ){
      ItemInternal* item = items_internal[ dest_items_local_id[z] ];
      dest_items_unique_id[z] = item->uniqueId().asInt64();
    }

    ISerializer* isbuf = comm->serializer();
    SerializeBuffer* sbuf = dynamic_cast<SerializeBuffer*>(isbuf);
    if (!sbuf)
      ARCANE_FATAL("buffer has to have type 'SerializeBuffer'");

    // Sauve les uid des mailles dans lesquelles se trouvent les particules
    // Il est possible qu'une particule n'appartienne pas à une maille.
    Int64UniqueArray particles_cell_uid(nb_item);
    for( Integer z=0; z<nb_item; ++z ){
      Particle item(items_internal[dest_items_local_id[z]]);
      bool has_cell = item.hasCell();
      particles_cell_uid[z] = (has_cell) ? item.cell().uniqueId() : NULL_ITEM_UNIQUE_ID;
    }

    // Réserve la mémoire pour la sérialisation
    sbuf->setMode(ISerializer::ModeReserve);
    sbuf->reserve(DT_Int64,1); // Pour le nombre de particules
    sbuf->reserveArray(dest_items_unique_id); // Pour les uniqueId() des particules. NOTE: A supprimer
    sbuf->reserveArray(particles_cell_uid); // Pour les uniqueId() des mailles dans lesquelles se trouve les particules

    sbuf->allocateBuffer();
    sbuf->setMode(ISerializer::ModePut);

    sbuf->putInt64(nb_item);
    sbuf->putArray(dest_items_unique_id);
    sbuf->putArray(particles_cell_uid);
  }
  exchanger->processExchange();

  Int64UniqueArray particles_uid;
  Int64UniqueArray cells_unique_id;
  Int32UniqueArray cells_local_id;
  Int32UniqueArray particles_owner;
  IItemFamily* cell_family = m_mesh->cellFamily();
  CellInfoListView internal_cells(cell_family);

  // Réception des particules
  for( Integer i=0, ns=exchanger->nbReceiver(); i<ns; ++i ){
    ISerializeMessage* sm = exchanger->messageToReceive(i);
    ISerializer* sbuf = sm->serializer();
    //particle_family->addParticles(s);

    Int64 nb_item = sbuf->getInt64();
    sbuf->getArray(particles_uid);
    sbuf->getArray(cells_unique_id);
    cells_local_id.resize(nb_item);
    Int32UniqueArray particles_local_id(nb_item);

    cell_family->itemsUniqueIdToLocalId(cells_local_id,cells_unique_id,true);
    particles_owner.resize(nb_item) ;
    for( Integer zz=0; zz<nb_item; ++zz ){
      Int32 cell_lid = cells_local_id[zz];
      Cell c = internal_cells[ cell_lid ];
      particles_owner[zz] = c.owner() ;
    }

    particle_family->addParticles2(particles_uid,particles_owner,particles_local_id);
    particle_family->endUpdate();
    // IMPORTANT: il faut le faire ici car cela peut changer via le endUpdate()
    ItemInternalList internal_particles(particle_family->itemsInternal());
    for( Integer zz=0; zz<nb_item; ++zz ){
      Particle p = internal_particles[ particles_local_id[zz] ];
      Int32 cell_lid = cells_local_id[zz];
      if (cell_lid!=NULL_ITEM_LOCAL_ID){
        Cell c = internal_cells[ cell_lid ];
        particle_family->setParticleCell(p,c);
      }
      else
        particle_family->setParticleCell(p,Cell());
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
