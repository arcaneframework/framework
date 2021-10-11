// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* GraphDoFs.cc                                                (C) 2000-2021 */
/*                                                                           */
/*---------------------------------------------------------------------------*/

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
, m_connectivity_mng(mesh->traceMng())
, m_dof_mng(mesh, &m_connectivity_mng)
, m_dual_node_family(m_dof_mng.family(GraphDoFs::dualNodeFamilyName(), true))
, m_link_family(m_dof_mng.family(GraphDoFs::linkFamilyName(), true))
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
    traceMng()->fatal() << "ARCANE_GRAPH_CONNECTIVITY_POLICY need to be activated";

  m_links_incremental_connectivity =
  new IncrementalItemConnectivity(linkFamily(),
                                  dualNodeFamily(),
                                  mesh::connectivityName(linkFamily(), dualNodeFamily()));
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
      m_incremental_connectivities[_connectivityIndex(dual_node_kind)] = dof2dual_incremental_connectivity;
      if (m_item_family_network)
        m_item_family_network->addDependency(dualNodeFamily(), dual_item_family, dof2dual_incremental_connectivity, false);
    }
  }
  m_graph_connectivity.reset(new GraphIncrementalConnectivity(dualNodeFamily(),
                                                              linkFamily(),
                                                              m_links_incremental_connectivity,
                                                              m_incremental_connectivities,
                                                              m_dual_node_to_connectivity_index));
  m_graph_allocated = true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void GraphDoFs::
addLinks(Integer nb_link,
         Integer nb_dual_nodes_per_link,
         Int64ConstArrayView links_infos)
{
  Trace::Setter mci(traceMng(), _className());
  if (!m_graph_allocated)
    _allocateGraph();

  mesh::DoFFamily& link_family = m_dof_mng.family(GraphDoFs::linkFamilyName());
  mesh::DoFFamily& dual_node_family = m_dof_mng.family(GraphDoFs::dualNodeFamilyName());

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
  link_family.addDoFs(link_uids, link_lids);
  link_family.endUpdate();

  // resize connectivity
  // fill connectivity
  Int32UniqueArray connected_dual_nodes_lids(nb_link * nb_dual_nodes_per_link);
  dual_node_family.itemsUniqueIdToLocalId(
  connected_dual_nodes_lids.view(), connected_dual_node_uids.constView(), true);

  auto link_index = 0;
  ENUMERATE_DOF (inewlink, link_family.view(link_lids)) {
    m_links_incremental_connectivity->notifySourceItemAdded(ItemLocalId(*inewlink));
    for (auto lid : connected_dual_nodes_lids.subConstView(link_index, nb_dual_nodes_per_link))
      m_links_incremental_connectivity->addConnectedItem(ItemLocalId(*inewlink), ItemLocalId(lid));
    link_index += nb_dual_nodes_per_link;
  }
  m_update_sync_info = true;
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
  auto& dual_node_family = m_dof_mng.family(GraphDoFs::dualNodeFamilyName());
  dual_node_family.addDoFs(dual_node_uids, dual_node_lids);
  dual_node_family.endUpdate();

  IItemFamily* dual_item_family = _dualItemFamily(dualItemKind(dual_node_kind));

  auto incremental_dual_item_connectivity = m_incremental_connectivities[_connectivityIndex(dual_node_kind)];
  Int32UniqueArray dual_item_lids(dual_item_uids.size());
  dual_item_family->itemsUniqueIdToLocalId(dual_item_lids, dual_item_uids);

  ENUMERATE_DOF (idual_node, dual_node_family.view(dual_node_lids)) {
    incremental_dual_item_connectivity->notifySourceItemAdded(ItemLocalId(*idual_node));
    incremental_dual_item_connectivity->addConnectedItem(ItemLocalId(*idual_node), ItemLocalId(dual_item_lids[idual_node.index()]));
  }

  m_dual_node_to_connectivity_index.resize(&dual_node_family, _connectivityIndex(dual_node_kind));

  m_update_sync_info = true;
}

void GraphDoFs::
addDualNodes(Integer graph_nb_dual_node,
             Int64ConstArrayView dual_nodes_infos)
{

  Trace::Setter mci(traceMng(), _className());
  if (!m_graph_allocated)
    _allocateGraph();

  // Size m_connecitivities if not yet done

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

    Int32UniqueArray dual_node_lids(dual_node_uids.size());
    auto& dual_node_family = m_dof_mng.family(GraphDoFs::dualNodeFamilyName());
    dual_node_family.addDoFs(dual_node_uids, dual_node_lids);
    dual_node_family.endUpdate();

    auto incremental_dual_item_connectivity = m_incremental_connectivities[index];
    IItemFamily* dual_item_family = _dualItemFamily(dualItemKind(dual_node_kind));
    if (dual_item_family) {
      Int32UniqueArray dual_item_lids(dual_item_uids.size());
      dual_item_family->itemsUniqueIdToLocalId(dual_item_lids, dual_item_uids);

      ENUMERATE_DOF (idual_node, dual_node_family.view(dual_node_lids)) {
        incremental_dual_item_connectivity->notifySourceItemAdded(ItemLocalId(*idual_node));
        incremental_dual_item_connectivity->addConnectedItem(ItemLocalId(*idual_node), ItemLocalId(dual_item_lids[idual_node.index()]));
      }

      m_dual_node_to_connectivity_index.resize(&dual_node_family, _connectivityIndex(dual_node_kind));
    }
  }
  m_update_sync_info = true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void GraphDoFs::
removeDualNodes(Int32ConstArrayView dual_node_local_ids)
{
  Trace::Setter mci(traceMng(), _className());
  //m_dual_node_family.removeItems(dual_node_local_ids);
  m_dof_mng.family(GraphDoFs::dualNodeFamilyName()).removeDoFs(dual_node_local_ids);
  if (dual_node_local_ids.size() > 0)
    m_update_sync_info = true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void GraphDoFs::
removeLinks(Int32ConstArrayView link_local_ids)
{
  Trace::Setter mci(traceMng(), _className());
  //m_link_family.removeItems(link_local_ids);
  m_dof_mng.family(GraphDoFs::linkFamilyName()).removeDoFs(link_local_ids);
  if (link_local_ids.size() > 0)
    m_update_sync_info = true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void GraphDoFs::
endUpdate()
{
  auto* x = new GraphIncrementalConnectivity(dualNodeFamily(),
                                             linkFamily(),
                                             m_links_incremental_connectivity,
                                             m_incremental_connectivities,
                                             m_dual_node_to_connectivity_index);
  m_graph_connectivity.reset(x);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void GraphDoFs::
updateAfterMeshChanged()
{
  if (!m_graph_allocated)
    return;

  auto& dual_node_family = m_dof_mng.family(GraphDoFs::dualNodeFamilyName());
  m_dual_node_to_connectivity_index.resize(&dual_node_family, -1);
  ENUMERATE_DOF (idof, dual_node_family.allItems()) {
    for (Integer index = 0; index < m_incremental_connectivities.size(); ++index) {
      auto connectivity = m_incremental_connectivities[index];
      if (connectivity) {
        ConnectivityItemVector accessor(connectivity);
        if (accessor.connectedItems(*idof).size() > 0) {
          m_dual_node_to_connectivity_index[*idof] = index;
        }
      }
    }
  }

  {
    // TODO: GG: appeler endUpdate() à la place pour éviter une duplication de code
    auto* x = new GraphIncrementalConnectivity(dualNodeFamily(),
                                               linkFamily(),
                                               m_links_incremental_connectivity,
                                               m_incremental_connectivities,
                                               m_dual_node_to_connectivity_index);
    m_graph_connectivity.reset(x);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void GraphDoFs::
printDualNodes() const
{
  auto graph_connectivity = GraphIncrementalConnectivity(dualNodeFamily(),
                                                         linkFamily(),
                                                         m_links_incremental_connectivity,
                                                         m_incremental_connectivities,
                                                         m_dual_node_to_connectivity_index);
  ENUMERATE_DOF (idualnode, dualNodeFamily()->allItems()) {
    info() << "DualNode : lid = " << idualnode->localId();
    info() << "           uid = " << idualnode->uniqueId();
    auto dual_item = graph_connectivity.dualItem(*idualnode);
    info() << "           DualItem : lid = " << dual_item.localId();
    info() << "                      uid = " << dual_item.uniqueId();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void GraphDoFs::
printLinks() const
{
  ConnectivityItemVector dual_nodes(m_links_incremental_connectivity);
  auto graph_connectivity = GraphIncrementalConnectivity(dualNodeFamily(),
                                                         linkFamily(),
                                                         m_links_incremental_connectivity,
                                                         m_incremental_connectivities,
                                                         m_dual_node_to_connectivity_index);
  ENUMERATE_DOF (ilink, linkFamily()->allItems()) {
    info() << "Link       :         LID = " << ilink.localId();
    info() << "                     UID = " << ilink->uniqueId();
    ENUMERATE_DOF (idual_node, dual_nodes.connectedItems(ilink)) {
      info() << "     Dof :       index = " << idual_node.index();
      info() << "     Dof :         LID = " << idual_node->localId();
      info() << "                   UID = " << idual_node->uniqueId();
      auto dual_item = graph_connectivity.dualItem(*idual_node);
      info() << "         dual item UID = " << dual_item.uniqueId();
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
