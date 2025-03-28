// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* GraphDoFs.cc                                                (C) 2000-2025 */
/*                                                                           */
/*---------------------------------------------------------------------------*/

#include "arcane/mesh/DualUniqueIdMng.h"
#include "arcane/mesh/GraphDoFs.h"

#include "arcane/utils/ArgumentException.h"

#include "arcane/IMesh.h"
#include "arcane/IItemConnectivity.h"
#include "arcane/IIncrementalItemConnectivity.h"
#include "arcane/MathUtils.h"

#include "arcane/mesh/ConnectivityNewWithDependenciesTypes.h"
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

GraphIncrementalConnectivity::
GraphIncrementalConnectivity(GraphDoFs* graph)
: m_dualnode_family(graph->dualNodeFamily())
, m_link_family(graph->linkFamily())
, m_dualnode_connectivity(graph->m_dualnodes_incremental_connectivity)
, m_link_connectivity(graph->m_links_incremental_connectivity)
, m_dualitem_connectivities(graph->m_incremental_connectivities)
, m_dualnode_to_connectivity_index(graph->m_dual_node_to_connectivity_index)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

GraphDoFs::
GraphDoFs(IMesh* mesh, String particle_family_name)
: TraceAccessor(mesh->traceMng())
, m_mesh(mesh)
, m_dof_mng(mesh)
, m_connectivity_mng(m_dof_mng.connectivityMng())
, m_dual_node_family(m_dof_mng.getFamily(GraphDoFs::dualNodeFamilyName(), true))
, m_link_family(m_dof_mng.getFamily(GraphDoFs::linkFamilyName(), true))
, m_update_sync_info(false)
, m_particle_family_name(particle_family_name)
{
}

void GraphDoFs::
_allocateGraph()
{
  if (m_graph_allocated)
    return;

  auto connectivity_index = 0;
  m_connectivity_indexes_per_type.resize(NB_BASIC_ITEM_TYPE, -1);
  m_connectivity_indexes_per_type[IT_DualNode] = connectivity_index++;
  m_connectivity_indexes_per_type[IT_DualEdge] = connectivity_index++;
  m_connectivity_indexes_per_type[IT_DualFace] = connectivity_index++;
  m_connectivity_indexes_per_type[IT_DualCell] = connectivity_index++;
  m_connectivity_indexes_per_type[IT_DualParticle] = connectivity_index++;

  m_item_family_network = m_mesh->itemFamilyNetwork();
  if (m_item_family_network == nullptr)
    ARCANE_FATAL("ARCANE_GRAPH_CONNECTIVITY_POLICY need to be activated");

  m_graph_id = m_item_family_network->registerConnectedGraph(this);

  m_dualnodes_incremental_connectivity =
  new IncrementalItemConnectivity(dualNodeFamily(),
                                  linkFamily(),
                                  mesh::connectivityName(dualNodeFamily(), linkFamily()));

  m_connectivity_mng->registerConnectivity(m_dualnodes_incremental_connectivity);

  if (m_item_family_network)
    m_item_family_network->addRelation(dualNodeFamily(), linkFamily(), m_dualnodes_incremental_connectivity);

  m_links_incremental_connectivity =
  new IncrementalItemConnectivity(linkFamily(),
                                  dualNodeFamily(),
                                  mesh::connectivityName(linkFamily(), dualNodeFamily()));
  m_connectivity_mng->registerConnectivity(m_links_incremental_connectivity);

  if (m_item_family_network)
    //m_item_family_network->addRelation(linkFamily(),dualNodeFamily(),m_links_incremental_connectivity);
    m_item_family_network->addDependency(linkFamily(), dualNodeFamily(), m_links_incremental_connectivity, false);

  m_incremental_connectivities.resize(NB_DUAL_ITEM_TYPE, nullptr);
  std::array<int, NB_DUAL_ITEM_TYPE> dual_node_kinds = { IT_DualCell, IT_DualFace, IT_DualEdge, IT_DualNode, IT_DualParticle };
  for (auto dual_node_kind : dual_node_kinds) {
    IItemFamily* dual_item_family = _dualItemFamily(dualItemKind(dual_node_kind));
    if (dual_item_family) {
      auto dof2dual_incremental_connectivity =
      new IncrementalItemConnectivity(dualNodeFamily(),
                                      dual_item_family,
                                      mesh::connectivityName(dualNodeFamily(), dual_item_family));

      m_connectivity_mng->registerConnectivity(dof2dual_incremental_connectivity);
      m_incremental_connectivities[_connectivityIndex(dual_node_kind)] = dof2dual_incremental_connectivity;
      if (m_item_family_network)
        m_item_family_network->addDependency(dualNodeFamily(), dual_item_family, dof2dual_incremental_connectivity, false);
    }
  }
  m_graph_connectivity.reset(new GraphIncrementalConnectivity(dualNodeFamily(),
                                                              linkFamily(),
                                                              m_dualnodes_incremental_connectivity,
                                                              m_links_incremental_connectivity,
                                                              m_incremental_connectivities,
                                                              m_dual_node_to_connectivity_index));

  for (auto& obs : m_connectivity_observer)
    if (obs.get())
      obs->notifyUpdateConnectivity();

  m_update_sync_info = false;
  m_graph_allocated = true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void GraphDoFs::
addLinks(Integer nb_link,
         Integer nb_dual_nodes_per_link,
         Int64ConstArrayView links_infos)
{
  //info() << "GRAPHDOF : ADD LINKS";
  Trace::Setter mci(traceMng(), _className());
  if (!m_graph_allocated)
    _allocateGraph();

  IDoFFamily* link_family = m_dof_mng.getFamily(GraphDoFs::linkFamilyName());
  IDoFFamily* dual_node_family = m_dof_mng.getFamily(GraphDoFs::dualNodeFamilyName());

  // Extract link infos
  Int64UniqueArray link_uids, connected_dual_node_uids;
  link_uids.reserve(nb_link);
  connected_dual_node_uids.reserve(nb_link * nb_dual_nodes_per_link);
  for (auto links_infos_index = 0; links_infos_index < links_infos.size();) {
    auto link_uid = links_infos[links_infos_index++];
    link_uids.add(link_uid);
    connected_dual_node_uids.addRange(
    links_infos.subConstView(links_infos_index, nb_dual_nodes_per_link));
    links_infos_index += nb_dual_nodes_per_link;
  }

  Int32UniqueArray link_lids(link_uids.size());
  link_family->addDoFs(link_uids, link_lids);
  link_family->endUpdate();

  // resize connectivity
  // fill connectivity
  Int32UniqueArray connected_dual_nodes_lids(nb_link * nb_dual_nodes_per_link);
  dual_node_family->itemFamily()->itemsUniqueIdToLocalId(
  connected_dual_nodes_lids.view(), connected_dual_node_uids.constView(), true);

  auto link_index = 0;
  ENUMERATE_DOF (inewlink, link_family->itemFamily()->view(link_lids)) {
    m_links_incremental_connectivity->notifySourceItemAdded(ItemLocalId(*inewlink));
    for (auto lid : connected_dual_nodes_lids.subConstView(link_index, nb_dual_nodes_per_link)) {
      m_links_incremental_connectivity->addConnectedItem(ItemLocalId(*inewlink), ItemLocalId(lid));
      m_dualnodes_incremental_connectivity->addConnectedItem(ItemLocalId(lid), ItemLocalId(*inewlink));
    }
    link_index += nb_dual_nodes_per_link;
  }
  m_connectivity_mng->setUpToDate(m_links_incremental_connectivity);
  m_connectivity_mng->setUpToDate(m_dualnodes_incremental_connectivity);

  m_update_sync_info = false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void GraphDoFs::
addDualNodes(Integer graph_nb_dual_node,
             Integer dual_node_kind,
             Int64ConstArrayView dual_nodes_infos)
{

  Trace::Setter mci(traceMng(), _className());
  if (!m_graph_allocated)
    _allocateGraph();

  // Size m_connecitivities if not yet done

  Int64UniqueArray dual_node_uids, dual_item_uids;
  dual_node_uids.reserve(graph_nb_dual_node);
  dual_item_uids.reserve(graph_nb_dual_node);
  for (auto infos_index = 0; infos_index < dual_nodes_infos.size();) {
    dual_node_uids.add(dual_nodes_infos[infos_index++]);
    dual_item_uids.add(dual_nodes_infos[infos_index++]);
  }

  Int32UniqueArray dual_node_lids(dual_node_uids.size());
  auto* dual_node_family = m_dof_mng.getFamily(GraphDoFs::dualNodeFamilyName());
  dual_node_family->addDoFs(dual_node_uids, dual_node_lids);
  dual_node_family->endUpdate();

  IItemFamily* dual_item_family = _dualItemFamily(dualItemKind(dual_node_kind));

  auto incremental_dual_item_connectivity = m_incremental_connectivities[_connectivityIndex(dual_node_kind)];
  if (incremental_dual_item_connectivity == nullptr) {
    incremental_dual_item_connectivity =
    new IncrementalItemConnectivity(dualNodeFamily(),
                                    dual_item_family,
                                    mesh::connectivityName(dualNodeFamily(), dual_item_family));

    m_connectivity_mng->registerConnectivity(incremental_dual_item_connectivity);
    m_incremental_connectivities[_connectivityIndex(dual_node_kind)] = incremental_dual_item_connectivity;
    if (m_item_family_network)
      m_item_family_network->addDependency(dualNodeFamily(), dual_item_family, incremental_dual_item_connectivity, false);
  }
  Int32UniqueArray dual_item_lids(dual_item_uids.size());
  dual_item_family->itemsUniqueIdToLocalId(dual_item_lids, dual_item_uids);

  ENUMERATE_DOF (idual_node, dual_node_family->itemFamily()->view(dual_node_lids)) {
    incremental_dual_item_connectivity->notifySourceItemAdded(ItemLocalId(*idual_node));
    incremental_dual_item_connectivity->addConnectedItem(ItemLocalId(*idual_node), ItemLocalId(dual_item_lids[idual_node.index()]));
  }
  m_connectivity_mng->setUpToDate(incremental_dual_item_connectivity);

  m_update_sync_info = false;
  //m_dual_node_to_connectivity_index.resize(&dual_node_family, _connectivityIndex(dual_node_kind));
}

void GraphDoFs::
addDualNodes(Integer graph_nb_dual_node,
             Int64ConstArrayView dual_nodes_infos)
{

  Trace::Setter mci(traceMng(), _className());
  if (!m_graph_allocated)
    _allocateGraph();

  // Size m_connecitivities if not yet done
  bool is_parallel = m_mesh->parallelMng()->isParallel();
  Integer domain_rank = m_mesh->parallelMng()->commRank();

  std::map<Int64, std::pair<Int64UniqueArray, Int64UniqueArray>> dual_info_per_kind;
  for (auto infos_index = 0; infos_index < dual_nodes_infos.size();) {
    Int64 dual_node_kind = dual_nodes_infos[infos_index++];
    auto& info = dual_info_per_kind[dual_node_kind];
    auto& dual_node_uids = info.first;
    auto& dual_item_uids = info.second;
    if (dual_node_uids.size() == 0) {
      dual_node_uids.reserve(graph_nb_dual_node);
      dual_item_uids.reserve(graph_nb_dual_node);
    }
    dual_node_uids.add(dual_nodes_infos[infos_index++]);
    dual_item_uids.add(dual_nodes_infos[infos_index++]);
  }

  for (Integer index = 0; index < NB_DUAL_ITEM_TYPE; ++index) {
    Integer dual_node_kind = m_dualnode_kinds[index];
    auto& info = dual_info_per_kind[dual_node_kind];
    auto& dual_node_uids = info.first;
    auto& dual_item_uids = info.second;

    auto* dual_node_family = m_dof_mng.getFamily(GraphDoFs::dualNodeFamilyName());
    Int32UniqueArray dual_node_lids(dual_node_uids.size());
    if (is_parallel) {
      IItemFamily* dual_item_family = _dualItemFamily(dualItemKind(dual_node_kind));
      if (dual_item_family) {
        Int32UniqueArray dual_item_lids(dual_item_uids.size());
        dual_item_family->itemsUniqueIdToLocalId(dual_item_lids, dual_item_uids);
        auto dual_item_view = dual_item_family->view(dual_item_lids);

        Int32UniqueArray local_dual_node_lids;
        Int64UniqueArray local_dual_node_uids;

        Integer local_size = 0;
        for (auto const& item : dual_item_view)
          if (item.owner() == domain_rank)
            ++local_size;
        local_dual_node_lids.resize(local_size);
        local_dual_node_uids.reserve(local_size);

        Integer ghost_size = dual_node_uids.size() - local_size;
        Int64UniqueArray ghost_dual_node_uids;
        Int32UniqueArray ghost_dual_node_lids;
        Int32UniqueArray ghost_dual_node_owner;
        ghost_dual_node_lids.resize(ghost_size);
        ghost_dual_node_uids.reserve(ghost_size);
        ghost_dual_node_owner.reserve(ghost_size);

        Integer icount = 0;
        for (auto const& item : dual_item_view) {
          if (item.owner() == domain_rank) {
            local_dual_node_uids.add(dual_node_uids[icount]);
          }
          else {
            ghost_dual_node_uids.add(dual_node_uids[icount]);
            ghost_dual_node_owner.add(item.owner());
          }
          ++icount;
        }

        dual_node_family->addDoFs(local_dual_node_uids, local_dual_node_lids);
        dual_node_family->addGhostDoFs(ghost_dual_node_uids, ghost_dual_node_lids, ghost_dual_node_owner);
        dual_node_family->endUpdate();

        icount = 0;
        Integer local_icount = 0;
        Integer ghost_icount = 0;
        for (auto const& item : dual_item_view) {
          if (item.owner() == domain_rank)
            dual_node_lids[icount] = local_dual_node_lids[local_icount++];
          else
            dual_node_lids[icount] = ghost_dual_node_lids[ghost_icount++];
          ++icount;
        }
      }
    }
    else {
      dual_node_family->addDoFs(dual_node_uids, dual_node_lids);
      dual_node_family->endUpdate();
    }

    IItemFamily* dual_item_family = _dualItemFamily(dualItemKind(dual_node_kind));
    if (dual_item_family) {

      auto incremental_dual_item_connectivity = m_incremental_connectivities[index];
      if (incremental_dual_item_connectivity == nullptr) {
        incremental_dual_item_connectivity =
        new IncrementalItemConnectivity(dualNodeFamily(),
                                        dual_item_family,
                                        mesh::connectivityName(dualNodeFamily(), dual_item_family));

        m_connectivity_mng->registerConnectivity(incremental_dual_item_connectivity);
        m_incremental_connectivities[index] = incremental_dual_item_connectivity;
        if (m_item_family_network)
          m_item_family_network->addDependency(dualNodeFamily(), dual_item_family, incremental_dual_item_connectivity, false);
      }
      Int32UniqueArray dual_item_lids(dual_item_uids.size());
      dual_item_family->itemsUniqueIdToLocalId(dual_item_lids, dual_item_uids);

      ENUMERATE_DOF (idual_node, dual_node_family->itemFamily()->view(dual_node_lids)) {
        incremental_dual_item_connectivity->notifySourceItemAdded(ItemLocalId(*idual_node));
        incremental_dual_item_connectivity->addConnectedItem(ItemLocalId(*idual_node), ItemLocalId(dual_item_lids[idual_node.index()]));
      }

      //m_dual_node_to_connectivity_index.resize(&dual_node_family, _connectivityIndex(dual_node_kind));

      m_connectivity_mng->setUpToDate(incremental_dual_item_connectivity);
    }
  }

  m_update_sync_info = false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void GraphDoFs::
removeDualNodes(Int32ConstArrayView dual_node_local_ids)
{
  Trace::Setter mci(traceMng(), _className());
  //m_dual_node_family.removeItems(dual_node_local_ids);
  m_dof_mng.getFamily(GraphDoFs::dualNodeFamilyName())->removeDoFs(dual_node_local_ids);
  if (dual_node_local_ids.size() > 0)
    m_update_sync_info = false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void GraphDoFs::
removeLinks(Int32ConstArrayView link_local_ids)
{
  Trace::Setter mci(traceMng(), _className());
  //m_link_family.removeItems(link_local_ids);
  m_dof_mng.getFamily(GraphDoFs::linkFamilyName())->removeDoFs(link_local_ids);
  if (link_local_ids.size() > 0)
    m_update_sync_info = false;
}

void GraphDoFs::
removeConnectedItemsFromCells(Int32ConstArrayView cell_local_ids)
{
  std::set<Int32> cell_to_remove_set;
  std::set<Int32> node_to_remove_set;
  std::set<Int32> face_to_remove_set;
  auto cell_family = m_mesh->cellFamily();
  auto remove_cells = CellVectorView(cell_family, cell_local_ids);
  ENUMERATE_CELL (icell, remove_cells) {
    cell_to_remove_set.insert(icell->localId());
    for (auto face : icell->faces()) {
      if (face.itemBase().isSuppressed())
        face_to_remove_set.insert(face.localId());
    }
    for (auto node : icell->nodes()) {
      if (node.itemBase().isSuppressed())
        node_to_remove_set.insert(node.localId());
    }
  }
  {
    std::set<Int32> link_to_remove_set;
    ConnectivityItemVector dual_nodes(m_links_incremental_connectivity);
    //ENUMERATE_DOF (ilink, linkFamily()->allItems())
    for (auto ilink : linkFamily()->itemsInternal()) {
      if (ilink->isSuppressed())
        continue;
      auto link_lid = ItemLocalId(ilink);
      bool to_remove = false;
      ENUMERATE_DOF (idual_node, dual_nodes.connectedItems(link_lid)) {
        auto dual_item = m_graph_connectivity->dualItem(*idual_node);
        switch (dual_item.kind()) {
        case IK_Cell:
          if (cell_to_remove_set.find(dual_item->localId()) != cell_to_remove_set.end())
            to_remove = true;
          break;
        case IK_Face:
          if (face_to_remove_set.find(dual_item->localId()) != face_to_remove_set.end())
            to_remove = true;
          break;
        case IK_Node:
          if (node_to_remove_set.find(dual_item->localId()) != node_to_remove_set.end())
            to_remove = true;
          break;
        case IK_Particle:
          if (dual_item.itemBase().isSuppressed())
            to_remove = true;
          break;
        }
        if (to_remove)
          break;
      }
      if (to_remove) {
        link_to_remove_set.insert(link_lid);
        ENUMERATE_DOF (idual_node, dual_nodes.connectedItems(link_lid)) {
          m_dualnodes_incremental_connectivity->removeConnectedItem(ItemLocalId(idual_node->localId()), link_lid);
        }
      }
    }
    m_detached_link_lids.resize(link_to_remove_set.size());
    std::copy(std::begin(link_to_remove_set), std::end(link_to_remove_set), std::begin(m_detached_link_lids));

    removeLinks(m_detached_link_lids);
    //link_family->endUpdate() ;
    m_detached_link_lids.clear();
  }
  {
    auto incremental_dof2cell_connectivity = m_incremental_connectivities[m_connectivity_indexes_per_type[IT_DualCell]];
    auto incremental_dof2face_connectivity = m_incremental_connectivities[m_connectivity_indexes_per_type[IT_DualFace]];
    auto incremental_dof2node_connectivity = m_incremental_connectivities[m_connectivity_indexes_per_type[IT_DualNode]];
    auto incremental_dof2part_connectivity = m_incremental_connectivities[m_connectivity_indexes_per_type[IT_DualParticle]];

    //auto source_family = incremental_dof2cell_connectivity->sourceFamily() ;
    //auto dof_family = dynamic_cast<DoFFamily*>(source_family) ;
    //auto accessor = incremental_dof2cell_connectivity->connectivityAccessor() ;

    //ENUMERATE_ITEM (idualnode, dualNodeFamily()->allItems())
    for (auto idualnode : dualNodeFamily()->itemsInternal()) {
      if (idualnode->isSuppressed())
        continue;
      auto dualnode_lid = ItemLocalId(idualnode);
      //auto lid = idualnode->localId() ;
      auto dual_item = m_graph_connectivity->dualItem(DoF(idualnode));
      switch (dual_item.kind()) {
      case IK_Cell:
        if (cell_to_remove_set.find(dual_item.localId()) != cell_to_remove_set.end()) {
          incremental_dof2cell_connectivity->replaceConnectedItem(dualnode_lid, 0, ItemLocalId(-1));
          m_detached_dualnode_lids.add(dualnode_lid);
          //info()<<"     REMOVE DUALCELL : "<<idualnode->uniqueId();
        }
        break;
      case IK_Face:
        if (face_to_remove_set.find(dual_item.localId()) != face_to_remove_set.end()) {
          incremental_dof2face_connectivity->replaceConnectedItem(dualnode_lid, 0, ItemLocalId(-1));
          m_detached_dualnode_lids.add(dualnode_lid);
          //info()<<"     REMOVE DUALFACE : "<<idualnode->uniqueId();
        }
        break;
      case IK_Node:
        if (node_to_remove_set.find(dual_item.localId()) != node_to_remove_set.end()) {
          incremental_dof2node_connectivity->replaceConnectedItem(dualnode_lid, 0, ItemLocalId(-1));
          m_detached_dualnode_lids.add(dualnode_lid);
          //info()<<"     REMOVE DUALNODE : "<<idualnode->uniqueId();
        }
        break;
      case IK_Particle:
        if (dual_item.itemBase().isSuppressed()) {
          incremental_dof2part_connectivity->replaceConnectedItem(dualnode_lid, 0, ItemLocalId(-1));
          m_detached_dualnode_lids.add(dualnode_lid);
          //info()<<"     REMOVE DUALPART : "<<idualnode->uniqueId();
        }
      }
    }

    removeDualNodes(m_detached_dualnode_lids);
    //dualNodeFamily()->endUpdate() ;
    m_detached_dualnode_lids.clear();
  }
  m_update_sync_info = false;
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool GraphDoFs::
isUpdated()
{
  if (!m_graph_allocated)
    return true;

  for (Integer index = 0; index < NB_DUAL_ITEM_TYPE; ++index) {
    Integer dual_node_kind = m_dualnode_kinds[index];
    auto dof2dual_incremental_connectivity = m_incremental_connectivities[_connectivityIndex(dual_node_kind)];

    if (dof2dual_incremental_connectivity && !m_connectivity_mng->isUpToDate(dof2dual_incremental_connectivity))
      m_update_sync_info = false;

    if (!m_connectivity_mng->isUpToDate(m_dualnodes_incremental_connectivity))
      m_update_sync_info = false;

    if (!m_connectivity_mng->isUpToDate(m_links_incremental_connectivity))
      m_update_sync_info = false;
  }

  Integer sync_info = (m_update_sync_info ? 0 : 1);
  sync_info = m_mesh->parallelMng()->reduce(Parallel::ReduceSum, sync_info);
  m_update_sync_info = sync_info == 0;

  return m_update_sync_info;
}

void GraphDoFs::
endUpdate()
{
  if (!m_graph_allocated)
    return;

  auto dualnode_family = dualNodeFamily();
  auto link_family = linkFamily();

  DualUniqueIdMng dual_uid_mng(traceMng());
  for (Integer index = 0; index < NB_DUAL_ITEM_TYPE; ++index) {
    Integer dual_node_kind = m_dualnode_kinds[index];
    Arcane::eItemKind dual_item_kind = dualItemKind(dual_node_kind);
    IItemFamily* dual_item_family = _dualItemFamily(dual_item_kind);

    if (dual_item_family == nullptr || dual_item_family->nbItem() == 0)
      continue;

    auto dof2dual_incremental_connectivity = m_incremental_connectivities[_connectivityIndex(dual_node_kind)];
    if (dof2dual_incremental_connectivity && !m_connectivity_mng->isUpToDate(dof2dual_incremental_connectivity)) {
      //info() << "UPDATE DUALNODE CONNECTIVITY KIND " << index << " " << dual_node_kind << " " << dual_item_kind;
      // Handle added nodes : create a dof for each own node added
      Int32ArrayView source_family_added_items_lids;
      Int32ArrayView source_family_removed_items_lids;
      m_connectivity_mng->getSourceFamilyModifiedItems(dof2dual_incremental_connectivity, source_family_added_items_lids, source_family_removed_items_lids);

      info() << "     NUMBER OF REMOVE ITEMS : " << source_family_removed_items_lids.size();
      info() << "     NUMBER OF ADDED ITEMS : " << source_family_added_items_lids.size();
      Int64UniqueArray dual_item_uids;
      dual_item_uids.reserve(source_family_added_items_lids.size());
      IntegerUniqueArray dual_node_lids;
      dual_node_lids.reserve(source_family_added_items_lids.size());

      ItemVector source_family_added_items(dualnode_family, source_family_added_items_lids);
      ENUMERATE_DOF (idof, source_family_added_items) {
        auto value = dual_uid_mng.uniqueIdOfDualItem(*idof);
        if (std::get<0>(value) == dual_item_kind) {
          dual_node_lids.add(idof->localId());
          dual_item_uids.add(std::get<1>(value));
        }
      }

      Int32SharedArray dual_item_lids(dual_item_uids.size());
      dual_item_family->itemsUniqueIdToLocalId(dual_item_lids, dual_item_uids);

      ENUMERATE_DOF (idual_node, dualnode_family->view(dual_node_lids)) {
        dof2dual_incremental_connectivity->notifySourceItemAdded(ItemLocalId(*idual_node));
        dof2dual_incremental_connectivity->addConnectedItem(ItemLocalId(*idual_node), ItemLocalId(dual_item_lids[idual_node.index()]));
        //info()<<"    ADD CONNECTED DUAL ITEM("<<idual_node->localId()<<" "<<idual_node->uniqueId()<<") "<<dual_item_lids[idual_node.index()]<<" "<<dual_item_uids[idual_node.index()] ;
      }

      // Update connectivity
      //dof2dual_incremental_connectivity->updateConnectivity(source_family_added_items_lids,dual_lids);

      // Update ghost
      //synchronizer->synchronize();

      // For test purpose only : try getSourceFamilyModifiedItem (must give back the new dofs created)
      //Int32ArrayView target_family_added_item_lids, target_family_removed_item_lids;
      //m_connectivity_mng->getTargetFamilyModifiedItems(dof2dual_incremental_connectivity,target_family_added_item_lids,target_family_removed_item_lids);
      //_checkTargetFamilyInfo(dualnode_family.view(target_family_added_item_lids),lids);

      //m_dual_node_to_connectivity_index.resize(dualnode_family, _connectivityIndex(dual_node_kind));

      // Finalize connectivity update
      m_connectivity_mng->setUpToDate(dof2dual_incremental_connectivity);
    }
  }

  if (!m_connectivity_mng->isUpToDate(m_links_incremental_connectivity)) {

    //info()<<"UPDATE LINK TO DUALNODE CONNECTIVITY" ;
    // Handle added nodes : create a dof for each own node added
    Int32ArrayView source_family_added_items_lids;
    Int32ArrayView source_family_removed_items_lids;
    m_connectivity_mng->getSourceFamilyModifiedItems(m_links_incremental_connectivity, source_family_added_items_lids, source_family_removed_items_lids);

    info() << "     NUMBER OF REMOVE ITEMS : " << source_family_removed_items_lids.size();
    info() << "     NUMBER OF ADDED ITEMS : " << source_family_added_items_lids.size();
    ItemVector source_family_added_items(link_family, source_family_added_items_lids);

    Int64UniqueArray link_uids;
    link_uids.reserve(source_family_added_items_lids.size());
    Int64UniqueArray dualnode_uids;
    dualnode_uids.reserve(2 * source_family_added_items_lids.size());
    ENUMERATE_DOF (ilink, source_family_added_items) {
      link_uids.add(ilink->uniqueId());
      auto value = dual_uid_mng.uniqueIdOfPairOfDualItems(*ilink);
      eItemKind dualitem_kind_1 = std::get<0>(value.first);
      Int64 dualitem_uid_1 = std::get<1>(value.first);
      Int64 dof_uid_1 = dual_uid_mng.uniqueIdOf(dualitem_kind_1, dualitem_uid_1);
      dualnode_uids.add(dof_uid_1);

      eItemKind dualitem_kind_2 = std::get<0>(value.second);
      Int64 dualitem_uid_2 = std::get<1>(value.second);
      Int64 dof_uid_2 = dual_uid_mng.uniqueIdOf(dualitem_kind_2, dualitem_uid_2);
      dualnode_uids.add(dof_uid_2);

      //info()<<"    ADDED DOF UID "<<ilink->uniqueId();
      //info()<<"      DUAL KIND 1="<<dualitem_kind_1<<" DUAL UID 1="<<dualitem_uid_1<<" DOF UID 1="<<dof_uid_1;
      //info()<<"      DUAL KIND 2="<<dualitem_kind_2<<" DUAL UID 2="<<dualitem_uid_2<<" DOF UID 2="<<dof_uid_2;
    }

    Integer nb_dual_nodes_per_link = 2;
    Int32UniqueArray connected_dual_nodes_lids(dualnode_uids.size());

    dualnode_family->itemsUniqueIdToLocalId(connected_dual_nodes_lids.view(), dualnode_uids);

    Integer link_index = 0;
    ENUMERATE_DOF (inewlink, source_family_added_items) {
      m_links_incremental_connectivity->notifySourceItemAdded(ItemLocalId(*inewlink));
      for (auto lid : connected_dual_nodes_lids.subConstView(link_index, nb_dual_nodes_per_link)) {
        m_links_incremental_connectivity->addConnectedItem(ItemLocalId(*inewlink), ItemLocalId(lid));
        m_dualnodes_incremental_connectivity->addConnectedItem(ItemLocalId(lid), ItemLocalId(*inewlink));
      }
      link_index += nb_dual_nodes_per_link;
    }

    // Finalize connectivity update
    m_connectivity_mng->setUpToDate(m_links_incremental_connectivity);
    m_connectivity_mng->setUpToDate(m_dualnodes_incremental_connectivity);
  }

  dualnode_family->computeSynchronizeInfos();
  link_family->computeSynchronizeInfos();
  m_update_sync_info = true;

  updateAfterMeshChanged();

  auto* x = new GraphIncrementalConnectivity(dualNodeFamily(),
                                             linkFamily(),
                                             m_dualnodes_incremental_connectivity,
                                             m_links_incremental_connectivity,
                                             m_incremental_connectivities,
                                             m_dual_node_to_connectivity_index);
  m_graph_connectivity.reset(x);

  for (auto& obs : m_connectivity_observer) {
    if (obs.get())
      obs->notifyUpdateConnectivity();
  }

  for (auto& obs : m_graph_observer) {
    if (obs.get())
      obs->notifyUpdate();
  }

  //printDualNodes();
  //printLinks();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void GraphDoFs::
updateAfterMeshChanged()
{
  if (!m_graph_allocated)
    return;

  auto* dual_node_family = m_dof_mng.getFamily(GraphDoFs::dualNodeFamilyName());
  m_dual_node_to_connectivity_index.resize(dual_node_family->itemFamily(), -1);
  ENUMERATE_DOF (idof, dual_node_family->allItems()) {
    for (Integer index = 0; index < m_incremental_connectivities.size(); ++index) {
      auto connectivity = m_incremental_connectivities[index];
      if (connectivity && (connectivity->maxNbConnectedItem() > 0)) {
        ConnectivityItemVector accessor(connectivity);
        if (accessor.connectedItems(*idof).size() > 0) {
          m_dual_node_to_connectivity_index[*idof] = index;
        }
      }
    }
  }
  //endUpdate();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void GraphDoFs::
printDualNodes() const
{
  ENUMERATE_DOF (idualnode, dualNodeFamily()->allItems()) {
    info() << "DualNode : lid = " << idualnode->localId();
    info() << "           uid = " << idualnode->uniqueId();
    info() << "         owner = " << idualnode->owner();
    auto dual_item = m_graph_connectivity->dualItem(*idualnode);
    info() << "           DualItem : lid = " << dual_item.localId();
    info() << "                      uid = " << dual_item.uniqueId();
    info() << "                     kind = " << dual_item.kind();
    info() << "                    owner = " << dual_item.owner();
    auto links = m_graph_connectivity->links(*idualnode);
    for (auto const& link : links) {
      info() << "           Connected link : lid = " << link.localId();
      info() << "                            uid = " << link.uniqueId();
      info() << "                          owner = " << link.owner();
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void GraphDoFs::
printLinks() const
{
  ENUMERATE_DOF (ilink, linkFamily()->allItems()) {
    info() << "Link       :         LID = " << ilink.localId();
    info() << "                     UID = " << ilink->uniqueId();
    info() << "                   OWNER = " << ilink->owner();
    ENUMERATE_DOF (idual_node, m_graph_connectivity->dualNodes(*ilink)) {
      info() << "     Dof :       index = " << idual_node.index();
      info() << "     Dof :         LID = " << idual_node->localId();
      info() << "                   UID = " << idual_node->uniqueId();
      info() << "                 OWNER = " << idual_node->owner();
      auto dual_item = m_graph_connectivity->dualItem(*idual_node);
      info() << "         dual item UID = " << dual_item.uniqueId();
      info() << "                  KIND = " << dual_item.kind();
      info() << "                 OWNER = " << dual_item.owner();
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
