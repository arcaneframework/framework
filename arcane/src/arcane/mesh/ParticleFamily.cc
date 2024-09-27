// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ParticleFamily.cc                                           (C) 2000-2024 */
/*                                                                           */
/* Famille de particules.                                                    */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/mesh/ParticleFamily.h"

#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/PlatformUtils.h"

#include "arcane/core/ISubDomain.h"
#include "arcane/core/IMesh.h"
#include "arcane/core/IVariableMng.h"
#include "arcane/core/Properties.h"
#include "arcane/core/ItemPrinter.h"

#include "arcane/mesh/ItemsExchangeInfo2.h"
#include "arcane/mesh/DynamicMesh.h"
#include "arcane/mesh/ItemConnectivitySelector.h"
#include "arcane/mesh/DynamicMeshKindInfos.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ParticleFamily::
ParticleFamily(IMesh* mesh, const String& name)
: ItemFamily(mesh, IK_Particle, name)
, m_particle_type_info(nullptr)
, m_particle_shared_info(nullptr)
, m_sub_domain_id(NULL_SUB_DOMAIN_ID)
, m_enable_ghost_items(false)
, m_cell_connectivity(nullptr)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ParticleFamily::
~ParticleFamily()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParticleFamily::
build()
{
  ItemFamily::build();
  ItemTypeMng* itm = m_mesh->itemTypeMng();
  m_particle_type_info = itm->typeFromId(IT_NullType);
  m_sub_domain_id = subDomain()->subDomainId();

  // Temporaire: pour désactiver la table de hashage pour uniqueId
  if (!platform::getEnvironmentVariable("ARCANE_PARTICLE_NO_UNIQUE_ID_MAP").null()) {
    pwarning() << "TEMPORARY: suppress particule uniqueId map";
    setHasUniqueIdMap(false);
  }
  else {
    bool has_unique_id_map = !m_properties->getBool("no-unique-id-map");
    _setHasUniqueIdMap(has_unique_id_map);
  }

  m_cell_connectivity = new CellConnectivity(this, mesh()->cellFamily(), "ParticleCell");

  _addConnectivitySelector(m_cell_connectivity);

  _buildConnectivitySelectors();

  // Préalloue une maille par particule.
  m_cell_connectivity->setPreAllocatedSize(1);

  _setSharedInfo();

  auto network = m_mesh->itemFamilyNetwork();
  if (network) {
    network->addDependency(this, mesh()->cellFamily(), m_cell_connectivity->customConnectivity(), false);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Positionne les infos de connectivité pour une particule qui
 * vient d'être allouée en mémoire.
 */
inline void ParticleFamily::
_initializeNewlyAllocatedParticle(ItemInternal* particle, Int64 uid)
{
  _allocateInfos(particle, uid, m_particle_shared_info);
  m_cell_connectivity->addConnectedItem(ItemLocalId(particle), ItemLocalId(NULL_ITEM_LOCAL_ID));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline ItemInternal* ParticleFamily::
_allocParticle(Int64 uid, bool& need_alloc)
{
  ItemInternal* ii = _allocOne(uid, need_alloc);

  if (!need_alloc)
    ii->setUniqueId(uid);
  else
    _initializeNewlyAllocatedParticle(ii, uid);

  // Une particule appartient toujours au sous-domaine qui l'a créée
  ii->setOwner(m_sub_domain_id, m_sub_domain_id);
  return ii;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline ItemInternal* ParticleFamily::
_findOrAllocParticle(Int64 uid, bool& is_alloc)
{
  ItemInternal* ii = ItemFamily::_findOrAllocOne(uid, is_alloc);
  if (is_alloc)
    _initializeNewlyAllocatedParticle(ii, uid);
  return ii;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ParticleVectorView ParticleFamily::
addParticles(Int64ConstArrayView unique_ids, Int32ArrayView items)
{
  _addItems(unique_ids, items);
  return view(items);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ParticleVectorView ParticleFamily::
addParticles2(Int64ConstArrayView unique_ids,
              Int32ConstArrayView owners,
              Int32ArrayView items)
{
  addItems(unique_ids, owners, items);
  return view(items);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParticleFamily::
_setCell(ItemLocalId particle, ItemLocalId cell)
{
  m_cell_connectivity->replaceItem(particle, 0, cell);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ParticleVectorView ParticleFamily::
addParticles(Int64ConstArrayView unique_ids,
             Int32ConstArrayView cells_local_id,
             Int32ArrayView items)
{
  _addItems(unique_ids, items);
  Integer n = items.size();
  for (Integer i = 0; i < n; ++i) {
    _setCell(ItemLocalId(items[i]), ItemLocalId(cells_local_id[i]));
  }

  return view(items);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParticleFamily::
setParticleCell(Particle particle, Cell new_cell)
{
  _setCell(particle, new_cell);
  m_need_prepare_dump = true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParticleFamily::
setParticlesCell(ParticleVectorView particles, CellVectorView new_cells)
{
  Int32ConstArrayView cell_ids = new_cells.localIds();
  Int32ConstArrayView particle_ids = particles.localIds();
  for (Integer i = 0, n = particle_ids.size(); i < n; ++i)
    _setCell(ItemLocalId(particle_ids[i]), ItemLocalId(cell_ids[i]));
  m_need_prepare_dump = true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParticleFamily::
_addItems(Int64ConstArrayView unique_ids, Int32ArrayView items)
{
  Integer nb_item = unique_ids.size();
  if (nb_item == 0)
    return;
  preAllocate(nb_item);

  bool need_alloc = false;
  for (Integer i = 0; i < nb_item; ++i) {
    Int64 uid = unique_ids[i];
    ItemInternal* ii = _allocParticle(uid, need_alloc);
    items[i] = ii->localId();
  }

  m_need_prepare_dump = true;
  _printInfos(nb_item);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParticleFamily::
addItems(Int64ConstArrayView unique_ids, Int32ConstArrayView owners, Int32ArrayView items)
{
  Integer nb_item = unique_ids.size();
  if (nb_item == 0)
    return;
  preAllocate(nb_item);
  // La méthode _findOrAlloc nécessite la table de hashage des uniqueId().
  if (!hasUniqueIdMap())
    ARCANE_FATAL("Can not add particles with owners when hasUniqueIdMap()==false family={0}",
                 name());

  bool need_alloc = false;
  for (Integer i = 0; i < nb_item; ++i) {
    Int64 uid = unique_ids[i];
    ItemInternal* ii = _findOrAllocParticle(uid, need_alloc);
    ii->setOwner(owners[i], m_sub_domain_id);

    items[i] = ii->localId();
  }

  m_need_prepare_dump = true;
  _printInfos(nb_item);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParticleFamily::
exchangeParticles()
{
  ItemsExchangeInfo2 ex(this);
  ex.computeExchangeItems();
  ex.computeExchangeInfos();
  ex.prepareToSend();
  ex.processExchange();
  ex.removeSentItems();
  ex.readAndAllocItems();
  notifyItemsOwnerChanged();
  endUpdate();
  ex.readGroups();
  ex.readVariables();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParticleFamily::
_printInfos(Integer nb_added)
{
  ARCANE_UNUSED(nb_added);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParticleFamily::
internalRemoveItems(Int32ConstArrayView local_ids, bool keep_ghost)
{
  ARCANE_UNUSED(keep_ghost);

  InternalConnectivityPolicy policy = mesh()->_connectivityPolicy();

  // A noter que cette boucle n'est pas utile pour le moment si on
  // n'utilise pas les nouvelles connectivités.
  // Elle le sera si on souhaite avoir des connectivités inverses
  // maille->particule.
  bool want_nullify_cell = (policy != InternalConnectivityPolicy::Legacy);
  ItemLocalId null_item_lid(NULL_ITEM_LOCAL_ID);

  auto network = m_mesh->itemFamilyNetwork();
  if (network) {
    std::set<Int32> remove_lids;
    if (want_nullify_cell) {
      for (Integer i = 0, n = local_ids.size(); i < n; ++i) {
        ItemLocalId lid(local_ids[i]);
        m_cell_connectivity->replaceItem(lid, 0, null_item_lid);
        remove_lids.insert(local_ids[i]);
      }
    }
    for (auto parent_connectivity : m_mesh->itemFamilyNetwork()->getParentConnectivities(this)) {
      //for (auto parent_connectivity : m_mesh->itemFamilyNetwork()->getParentRelations(this)) { // Should be getParentConnectivities, but because legacy connectivity cannot remove a connectivity with a Node as target, we need to restrain to Relations...
      for (auto source_item : parent_connectivity->sourceFamily()->itemsInternal()) {
        if (source_item->isSuppressed())
          continue;
        ConnectivityItemVector connectivity_accessor(parent_connectivity);
        ENUMERATE_ITEM (connected_item, connectivity_accessor.connectedItems(ItemLocalId(source_item))) {
          if (remove_lids.find(connected_item->localId()) != remove_lids.end()) {
            parent_connectivity->removeConnectedItem(ItemLocalId(source_item), connected_item);
          }
        }
      }
    }
  }
  else {
    if (want_nullify_cell) {
      for (Integer i = 0, n = local_ids.size(); i < n; ++i) {
        ItemLocalId lid(local_ids[i]);
        m_cell_connectivity->replaceItem(lid, 0, null_item_lid);
      }
    }
  }
  _removeMany(local_ids);

  m_need_prepare_dump = true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParticleFamily::
removeParticles(Int32ConstArrayView items_local_id)
{
  internalRemoveItems(items_local_id, false);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParticleFamily::
prepareForDump()
{
  Integer nb_item = nbItem();
  info(4) << "ParticleFamily::prepareForDump: " << name()
          << " n=" << nb_item;
  ItemFamily::prepareForDump();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParticleFamily::
readFromDump()
{
  ItemFamily::readFromDump();
  // Actualise le shared_info car il peut changer suite à une relecture
  _setSharedInfo();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParticleFamily::
preAllocate(Integer nb_item)
{
  Integer nb_hash = itemsMap().nbBucket();
  Integer wanted_size = 2 * (nb_item + nbItem());
  if (nb_hash < wanted_size)
    itemsMap().resize(wanted_size, true);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParticleFamily::
setHasUniqueIdMap(bool v)
{
  _setHasUniqueIdMap(v);
  m_properties->setBool("no-unique-id-map", !v);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool ParticleFamily::
hasUniqueIdMap() const
{
  return _infos().hasUniqueIdMap();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParticleFamily::
_setSharedInfo()
{
  m_particle_shared_info = _findSharedInfo(m_particle_type_info);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParticleFamily::
checkValidConnectivity()
{
  ItemFamily::checkValidConnectivity();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParticleFamily::
removeNeedRemoveMarkedItems()
{
  if (getEnableGhostItems()) {
    UniqueArray<Integer> lids_to_remove;
    lids_to_remove.reserve(1000);

    ItemInternalMap& particle_map = itemsMap();
    particle_map.eachItem([&](Item item) {
      Integer f = item.itemBase().flags();
      if (f & ItemFlags::II_NeedRemove) {
        f &= ~ItemFlags::II_NeedRemove;
        item.mutableItemBase().setFlags(f);
        lids_to_remove.add(item.localId());
      }
    });

    info() << "Number of particles of family " << name() << " to remove: " << lids_to_remove.size();
    if (lids_to_remove.size() > 0)
      removeParticles(lids_to_remove);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
