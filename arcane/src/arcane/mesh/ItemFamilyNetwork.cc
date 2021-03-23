// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemFamilyNetwork.cc                                        (C) 2000-2017 */
/*                                                                           */
/* ItemFamily relations through their connectivities.                        */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "ItemFamilyNetwork.h"

/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_MESH_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamilyNetwork::
addDependency(IItemFamily* master_family, IItemFamily* slave_family, IIncrementalItemConnectivity* master_to_slave_connectivity)
{
  m_dependency_graph.addEdge(master_family,slave_family,master_to_slave_connectivity);
  m_connectivity_list.add(master_to_slave_connectivity);
  m_connectivity_status[master_to_slave_connectivity] = false; // connectivity not stored by default
  m_families.insert(master_family);
  m_families.insert(slave_family);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamilyNetwork::
addRelation(IItemFamily* source_family, IItemFamily* target_family, IIncrementalItemConnectivity* source_to_target_connectivity)
{
  m_relation_graph.addEdge(source_family,target_family,source_to_target_connectivity);
  m_connectivity_list.add(source_to_target_connectivity);
  m_connectivity_status[source_to_target_connectivity] = false; // connectivity not stored by default
  m_families.insert(source_family);
  m_families.insert(target_family);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IIncrementalItemConnectivity* ItemFamilyNetwork::
getConnectivity(IItemFamily* source_family, IItemFamily* target_family, const String& name)
{
  bool is_dependency;
  return getConnectivity(source_family,target_family,name,is_dependency);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IIncrementalItemConnectivity* ItemFamilyNetwork::
getConnectivity(IItemFamily* source_family, IItemFamily* target_family, const String& name, bool& is_dependency)
{
  // Up to now DirectedGraph and DirectedAcyclicGraph are not hypergraph, we cannot have multiple edges between two same nodes.
  // => we just check the name is OK
  auto* connectivity = m_dependency_graph.getEdge(source_family,target_family);
  is_dependency = false;
  if (connectivity)
    {
      _checkConnectivityName(*connectivity,name);
      is_dependency = true;
      return *connectivity;
    }
  connectivity = m_relation_graph.getEdge(source_family,target_family);
  if (connectivity)
    {
      _checkConnectivityName(*connectivity,name);
      return *connectivity;
    }
  else return nullptr;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IIncrementalItemConnectivity* ItemFamilyNetwork::
getStoredConnectivity(IItemFamily* source_family, IItemFamily* target_family, const String& name)
{
  bool is_dependency;
  return getStoredConnectivity(source_family,target_family,name,is_dependency);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IIncrementalItemConnectivity* ItemFamilyNetwork::
getStoredConnectivity(IItemFamily* source_family, IItemFamily* target_family, const String& name, bool& is_dependency)
{
  IIncrementalItemConnectivity* con = getConnectivity(source_family,target_family,name,is_dependency);
  if (con) {
      if (!isStored(con)) con = nullptr;
  }
  return con;
}
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

List<IIncrementalItemConnectivity*> ItemFamilyNetwork::
getConnectivities()
{
  return m_connectivity_list;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

SharedArray<IIncrementalItemConnectivity*> ItemFamilyNetwork::
getChildDependencies(IItemFamily* source_family)
{
  return _getConnectivitiesFromGraph(m_dependency_graph.outEdges(source_family));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

SharedArray<IIncrementalItemConnectivity*> ItemFamilyNetwork::
getParentDependencies(IItemFamily* target_family)
{
  return _getConnectivitiesFromGraph(m_dependency_graph.inEdges(target_family));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

SharedArray<IIncrementalItemConnectivity*> ItemFamilyNetwork::
getChildRelations(IItemFamily* source_family)
{
  return _getConnectivitiesFromGraph(m_relation_graph.outEdges(source_family));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

SharedArray<IIncrementalItemConnectivity*> ItemFamilyNetwork::
getParentRelations(IItemFamily* target_family)
{
  return _getConnectivitiesFromGraph(m_relation_graph.inEdges(target_family));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

SharedArray<IIncrementalItemConnectivity*> ItemFamilyNetwork::
getChildConnectivities(IItemFamily* source_family)
{
  return _getConnectivitiesFromGraph(m_relation_graph.outEdges(source_family), m_dependency_graph.outEdges(source_family));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

SharedArray<IIncrementalItemConnectivity*> ItemFamilyNetwork::
getParentConnectivities(IItemFamily* target_family)
{
  return _getConnectivitiesFromGraph(m_relation_graph.inEdges(target_family), m_dependency_graph.inEdges(target_family));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

SharedArray<IIncrementalItemConnectivity*> ItemFamilyNetwork::
_getConnectivitiesFromGraph(const ConnectivityGraph::ConnectedEdgeSet& connectivity_edges)
{
  SharedArray<IIncrementalItemConnectivity*> connectivities(connectivity_edges.size());
  Integer index(0);
  for (auto connectivity_edge : connectivity_edges){
    connectivities[index++] = connectivity_edge;
  }
  return connectivities;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

SharedArray<IIncrementalItemConnectivity*> ItemFamilyNetwork::
_getConnectivitiesFromGraph(const ConnectivityGraph::ConnectedEdgeSet& connectivity_edges1, const ConnectivityGraph::ConnectedEdgeSet& connectivity_edges2)
{
  SharedArray<IIncrementalItemConnectivity*> connectivities(connectivity_edges1.size()+connectivity_edges2.size());
  Integer index(0);
  for (auto connectivity_edge : connectivity_edges1){
    connectivities[index++] = connectivity_edge;
  }
  for (auto connectivity_edge : connectivity_edges2){
    connectivities[index++] = connectivity_edge;
  }
  return connectivities;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamilyNetwork::
setIsStored(IIncrementalItemConnectivity* connectivity)
{
  _getConnectivityStatus(connectivity).second = true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool ItemFamilyNetwork::
isStored(IIncrementalItemConnectivity* connectivity)
{
  return _getConnectivityStatus(connectivity).second;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamilyNetwork::
_checkConnectivityName(IIncrementalItemConnectivity* connectivity, const String& name)
{
  if (connectivity->name() != name) throw FatalErrorException(String::format("Found connectivity ({0}) has not the expected name ({1}) ",connectivity->name(),name));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ItemFamilyNetwork::
schedule(IItemFamilyNetworkTask task, eSchedulingOrder scheduling_order)
{
  switch (scheduling_order) {
    case TopologicalOrder:
      for (auto family : m_dependency_graph.topologicalSort())
        {
          task(family);
        }
      break;
    case InverseTopologicalOrder:
      for (auto family : m_dependency_graph.topologicalSort().reverseOrder())
        {
          task(family);
        }
      break;
    case Unknown:
      throw m_trace_mng->fatal() << "Cannot schedule task, scheduling order is unkwnown. Set Scheduling order";
      break;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

std::pair<IIncrementalItemConnectivity* const,bool>& ItemFamilyNetwork::
_getConnectivityStatus(IIncrementalItemConnectivity* connectivity)
{
  auto connectivity_iterator = m_connectivity_status.find(connectivity);
  if (connectivity_iterator == m_connectivity_status.end()) throw FatalErrorException(String::format("Cannot find connectivity {0} between families {1} and {2}",
                                                                                                      connectivity->name(),connectivity->sourceFamily(),connectivity->targetFamily()));
  return *(connectivity_iterator);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_MESH_END_NAMESPACE
ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/


