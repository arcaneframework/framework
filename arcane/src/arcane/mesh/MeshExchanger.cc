// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshExchanger.cc                                            (C) 2000-2025 */
/*                                                                           */
/* Management of a mesh exchange between sub-domains.                        */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/TraceInfo.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/ValueConvert.h"

#include "arcane/core/IParallelMng.h"
#include "arcane/core/Timer.h"
#include "arcane/core/IItemFamilyPolicyMng.h"
#include "arcane/core/IItemFamilyExchanger.h"
#include "arcane/core/IParticleFamily.h"

#include "arcane/mesh/MeshExchanger.h"
#include "arcane/mesh/DynamicMesh.h"
#include "arcane/mesh/MeshExchange.h"
#include "arcane/core/internal/IMeshModifierInternal.h"
#include "arcane/core/internal/IItemFamilySerializerMngInternal.h"
#include "arcane/core/internal/IMeshInternal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshExchanger::
MeshExchanger(IMesh* mesh, ITimeStats* stats)
: TraceAccessor(mesh->traceMng())
, m_mesh(mesh)
, m_time_stats(stats)
, m_phase(ePhase::Init)
{
  // Temporarily uses an environment variable to specify the
  // maximum number of pending messages or if collective operations should be used
  String max_pending_str = platform::getEnvironmentVariable("ARCANE_MESH_EXCHANGE_MAX_PENDING_MESSAGE");
  if (!max_pending_str.null()) {
    Int32 max_pending = 0;
    if (!builtInGetValue(max_pending, max_pending_str))
      m_exchanger_option.setMaxPendingMessage(max_pending);
  }

  String use_collective_str = platform::getEnvironmentVariable("ARCANE_MESH_EXCHANGE_USE_COLLECTIVE");
  if (use_collective_str == "1" || use_collective_str == "TRUE")
    m_exchanger_option.setExchangeMode(ParallelExchangerOptions::EM_Collective);

  m_exchanger_option.setVerbosityLevel(1);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshExchanger::
~MeshExchanger()
{
  for (IItemFamilyExchanger* exchanger : m_family_exchangers)
    delete exchanger;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshExchanger::
build()
{
  if (!m_mesh->itemFamilyNetwork() || !IItemFamilyNetwork::plug_serializer) { // handle family order by hand
    // Sorted list of families specifically ordered to guarantee a certain order
    // during exchanges. For now, the order is determined as follows:
    // - first Cell, then Face, Edge, and Node
    // - then, Particle families must be handled before DualNode families.
    UniqueArray<IItemFamily*> sorted_families;
    IItemFamilyCollection families(m_mesh->itemFamilies());
    sorted_families.reserve(families.count());
    sorted_families.add(m_mesh->cellFamily());
    sorted_families.add(m_mesh->faceFamily());
    sorted_families.add(m_mesh->edgeFamily());
    sorted_families.add(m_mesh->nodeFamily());
    for (IItemFamily* family : families) {
      IParticleFamily* particle_family = family->toParticleFamily();
      if (particle_family)
        sorted_families.add(family);
    }

    // List of instances managing the exchange of a family.
    // WARNING: It is necessary to ensure the associated pointers are released.
    //m_family_exchangers.reserve(families.count());

    // Creation of each exchanger associated with a family.
    std::map<IItemFamily*, IItemFamilyExchanger*> family_exchanger_map;
    for (IItemFamily* family : sorted_families) {
      _addItemFamilyExchanger(family);
    }
  }
  else {
    if (m_mesh->useMeshItemFamilyDependencies()) {
      _buildWithItemFamilyNetwork();
    }
    else {
      std::set<String> family_set;
      UniqueArray<IItemFamily*> sorted_families;
      IItemFamilyCollection families(m_mesh->itemFamilies());
      sorted_families.reserve(families.count());
      sorted_families.add(m_mesh->cellFamily());
      family_set.insert(m_mesh->cellFamily()->name());
      sorted_families.add(m_mesh->faceFamily());
      family_set.insert(m_mesh->faceFamily()->name());
      sorted_families.add(m_mesh->edgeFamily());
      family_set.insert(m_mesh->edgeFamily()->name());
      sorted_families.add(m_mesh->nodeFamily());
      family_set.insert(m_mesh->nodeFamily()->name());
      for (IItemFamily* family : families) {
        IParticleFamily* particle_family = family->toParticleFamily();
        if (particle_family) {
          sorted_families.add(family);
          family_set.insert(family->name());
        }
      }

      for (auto family : m_mesh->itemFamilyNetwork()->getFamilies(IItemFamilyNetwork::InverseTopologicalOrder)) {
        auto value = family_set.insert(family->name());
        if (value.second) {
          sorted_families.add(family);
        }
      }

      // List of instances managing the exchange of a family.
      // WARNING: It is necessary to ensure the associated pointers are released.
      //m_family_exchangers.reserve(families.count());

      // Creation of each exchanger associated with a family.
      std::map<IItemFamily*, IItemFamilyExchanger*> family_exchanger_map;
      for (IItemFamily* family : sorted_families) {
        _addItemFamilyExchanger(family);
      }
    }
  }
  m_phase = ePhase::ComputeInfos;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshExchanger::
_buildWithItemFamilyNetwork()
{
  m_mesh->itemFamilyNetwork()->schedule([&](IItemFamily* family) {
    _addItemFamilyExchanger(family);
  },
                                        IItemFamilyNetwork::InverseTopologicalOrder);
  // Particle should be handled soon
  for (IItemFamily* family : m_mesh->itemFamilies()) {
    IParticleFamily* particle_family = family->toParticleFamily();
    if (particle_family)
      _addItemFamilyExchanger(family);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshExchanger::
_addItemFamilyExchanger(IItemFamily* family)
{
  IItemFamilyExchanger* exchanger = family->policyMng()->createExchanger();
  m_family_exchangers.add(exchanger);
  m_family_exchanger_map.insert(std::make_pair(family, exchanger));
  exchanger->setParallelExchangerOption(m_exchanger_option);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshExchanger::
_checkPhase(ePhase wanted_phase)
{
  if (m_phase != wanted_phase)
    ARCANE_FATAL("Invalid exchange phase wanted={0} current={1}",
                 (int)wanted_phase, (int)m_phase);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool MeshExchanger::
computeExchangeInfos()
{
  _checkPhase(ePhase::ComputeInfos);

  // TODO: make it possible to customize the exchange calculation
  // and do it by family if possible.
  MeshExchange mesh_exchange(m_mesh);
  info() << "MeshExchange begin date=" << platform::getCurrentDateTime();
  {
    Timer::Action ts_action1(m_time_stats, "MeshExchangeComputeInfos", true);
    mesh_exchange.computeInfos();
    // MeshExchange has set a NeedRemove mark on cells leaving this proc completely
  }

  IItemFamily* cell_family = m_mesh->cellFamily();
  IItemFamilyExchanger* cell_exchanger = findExchanger(cell_family);

  // First determine the info to exchange on the meshes because if there are no meshes
  // to exchange, the partitioning stops.
  // NOTE GG: we need to see if this remains true with links and dual nodes.
  cell_exchanger->setExchangeItems(mesh_exchange.getItemsToSend(cell_family));
  if (cell_exchanger->computeExchangeInfos()) {
    pwarning() << "No load balance is performed";
    return true;
  }

  // Determine the list of information to exchange for each family.
  for (IItemFamilyExchanger* exchanger : m_family_exchangers) {
    // The mesh exchange has already been done.
    if (exchanger == cell_exchanger)
      continue;
    IItemFamily* family = exchanger->itemFamily();
    info() << "ComputeExchange family=" << family->name()
           << " date=" << platform::getCurrentDateTime();
    // For particle families that do not support the notion
    // of ghost items, it is necessary to explicitly determine the list of entities to exchange
    // via the call to computeExchangeItems().
    // For other families where particle families do have the notion
    // of ghost items, this list has already been determined during the call to
    // mesh_exchange.computeInfos().
    IParticleFamily* particle_family = family->toParticleFamily();
    if (particle_family && particle_family->getEnableGhostItems() == false)
      exchanger->computeExchangeItems();
    else
      exchanger->setExchangeItems(mesh_exchange.getItemsToSend(family));
    exchanger->computeExchangeInfos();
  }

  // Copy the owner() field into the ItemInternal so that it
  // is consistent with the corresponding variable

  // WARNING: It is absolutely necessary that the owner() of the ItemInternal
  // are correct before sending the meshes that belonged to us
  // to the sub-domains to which they will then belong.

  // Note that we cannot merge this loop with the previous one because
  // the families need the info from other families to determine
  // the list of entities to send.
  Int32 rank = m_mesh->meshPartInfo().partRank();
  for (IItemFamilyExchanger* exchanger : m_family_exchangers) {
    IItemFamily* family = exchanger->itemFamily();
    VariableItemInt32& owners(family->itemsNewOwner());
    ENUMERATE_ITEM (i, family->allItems()) {
      Item item = *i;
      Integer new_owner = owners[item];
      item.mutableItemBase().setOwner(new_owner, rank);
    }
    family->notifyItemsOwnerChanged();
  }

  m_phase = ePhase::ProcessExchange;

  return false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshExchanger::
processExchange()
{
  _checkPhase(ePhase::ProcessExchange);

  info() << "ExchangeItems date=" << platform::getCurrentDateTime()
         << " MemUsed=" << platform::getMemoryUsed();

  Timer::Action ts_action1(m_time_stats, "MessagesExchange", true);
  for (IItemFamilyExchanger* e : m_family_exchangers) {
    // NOTE: To be able to send all messages at once and receive
    // them as well, it might be necessary to plan using MPI tags.
    e->prepareToSend(); // Preparation of all data to send then serialization
    e->processExchange(); // Actual sending
    e->releaseBuffer();
  }
  m_phase = ePhase::RemoveItems;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshExchanger::
removeNeededItems()
{
  _checkPhase(ePhase::RemoveItems);

  // Now that all messages with the mesh before modification
  // have been sent and received, we can modify this mesh by
  // removing the elements that no longer belong to it and adding
  // the new ones.

  // TODO: implement removal by family method.

  // For families other than particle families without ghosts, this
  // is done in the DynamicMeshIncrementalBuilder.
  // For particle families without ghosts, this is done here.
  info() << "RemoveItems date=" << platform::getCurrentDateTime();
  Timer::Action ts_action1(m_time_stats, "RemoveSendedItems", true);

  for (IItemFamilyExchanger* exchanger : m_family_exchangers) {
    IParticleFamily* particle_family = exchanger->itemFamily()->toParticleFamily();
    if (particle_family && particle_family->getEnableGhostItems() == false)
      exchanger->removeSentItems(); // integrates the treatment of sub-meshes (for particles)
  }

  // Remove entities that are no longer linked to the sub-domain
  m_mesh->modifier()->_modifierInternalApi()->removeNeedRemoveMarkedItems();

  m_phase = ePhase::AllocateItems;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshExchanger::
allocateReceivedItems()
{
  _checkPhase(ePhase::AllocateItems);
  {
    info() << "AllocItems date=" << platform::getCurrentDateTime();
    Timer::Action ts_action1(m_time_stats, "ReadAndAllocItems", true);
    // We must first perform the mesh exchange
    // This is guaranteed by the fact that the first element of family_exchangers
    // is the mesh family.
    for (IItemFamilyExchanger* e : m_family_exchangers) {
      e->readAndAllocItems(); // Caution, no longer proceeds on different sub-meshes
    }
    // If needed, finalize item allocations (for polyhedral meshes)
    auto* family_serializer_mng = m_mesh->_internalApi()->familySerializerMng();
    if (family_serializer_mng)
      family_serializer_mng->finalizeItemAllocation();

    // Build item relations (only dependencies are built in readAndAllocItems)
    // only for families registered in the graph
    if (m_mesh->itemFamilyNetwork() && m_mesh->itemFamilyNetwork()->isActivated()) {
      auto family_set = m_mesh->itemFamilyNetwork()->getFamilies();
      for (auto family : family_set) {
        m_family_exchanger_map[family]->readAndAllocItemRelations();
      }
    }

    // Separate mesh and submesh
    for (IItemFamilyExchanger* e : m_family_exchangers) {
      e->readAndAllocSubMeshItems(); // Proceeds on different sub-meshes
    }
  }

  // It is possible that the owners of the entities have changed
  // following readAndAllocItems() even if no entity was added.
  // Therefore, it must be indicated to the families.
  for (IItemFamilyExchanger* e : m_family_exchangers) {
    e->itemFamily()->notifyItemsOwnerChanged(); // applied up to a sub-mesh level
  }

  m_phase = ePhase::UpdateItemGroups;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshExchanger::
updateItemGroups()
{
  _checkPhase(ePhase::UpdateItemGroups);

  info() << "ReadGroups date=" << platform::getCurrentDateTime();
  // Now that the new mesh is created, we read the groups
  for (IItemFamilyExchanger* e : m_family_exchangers)
    e->readGroups();

  m_phase = ePhase::UpdateVariables;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshExchanger::
updateVariables()
{
  _checkPhase(ePhase::UpdateVariables);

  info() << "ReadVariables date=" << platform::getCurrentDateTime();
  Timer::Action ts(m_time_stats, "ReadVariables", true);
  // Now that the entities are created and the groups updated,
  // we can update the variables.
  for (IItemFamilyExchanger* e : m_family_exchangers)
    e->readVariables();

  m_phase = ePhase::Finalize;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshExchanger::
finalizeExchange()
{
  _checkPhase(ePhase::Finalize);

  // Finalize the exchanges
  // This must be done after compaction because in the case of linked interfaces,
  // the localId() numbering must not be changed once the
  // TiedInterface structures are updated.
  // TODO: this will need to be removed by doing this treatment before, but
  // for that, TiedInterfaceMng must be notified of the compaction to
  // update the localId() of its faces and nodes
  for (IItemFamilyExchanger* e : m_family_exchangers)
    e->finalizeExchange();

  m_phase = ePhase::Ended;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IItemFamilyExchanger* MeshExchanger::
findExchanger(IItemFamily* family)
{
  auto x = m_family_exchanger_map.find(family);
  if (x == m_family_exchanger_map.end())
    ARCANE_FATAL("No exchanger for family name={0}", family->name());
  return x->second;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IPrimaryMesh* MeshExchanger::
mesh() const
{
  return m_mesh->toPrimaryMesh();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshExchanger::
_setNextPhase(ePhase next_phase)
{
  m_phase = next_phase;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
