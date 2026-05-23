// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemFamilyNetwork.h                                         (C) 2000-2024 */
/*                                                                           */
/* ItemFamily relations through their connectivities.                        */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MESH_ITEMFAMILYNETWORK_H
#define ARCANE_MESH_ITEMFAMILYNETWORK_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include <map>
#include <set>

#include "arcane/utils/ArcaneGlobal.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/DirectedGraphT.h"
#include "arcane/utils/DirectedAcyclicGraphT.h"
#include "arcane/utils/List.h"
#include "arcane/IItemFamily.h"
#include "arcane/IIncrementalItemConnectivity.h"
#include "arcane/mesh/MeshGlobal.h"

#include "arcane/IGraph2.h"
#include "arcane/IGraphModifier2.h"
#include "arcane/IItemFamilyNetwork.h"
#include "arcane/utils/NotImplementedException.h" //tmp !

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_MESH_EXPORT ItemFamilyNetwork
: public IItemFamilyNetwork
{
 public:

  /** Constructor of the class */
  ItemFamilyNetwork(ITraceMng* trace_mng)
  : m_relation_graph(trace_mng)
  , m_dependency_graph(trace_mng)
{}

  /** Destructor of the class */
  virtual ~ItemFamilyNetwork() {
    for (auto connectivity : m_connectivity_list)
      {
        delete connectivity;
      }
  }

public:

  bool isActivated() const override {
    return m_is_activated ;
  }

  /*! Adds a dependency between two families; an element of \a master_family is composed of elements of \a slave_family.
   *  ItemFamilyNetwork takes responsibility for the memory of \a master_to_slave_connectivity
   */
  void addDependency(IItemFamily* master_family, IItemFamily* slave_family, IIncrementalItemConnectivity* master_to_slave_connectivity, bool is_deep_connectivity) override;

  /*! Adds a relation between two families; an element of \a source_family is connected to one or more elements of \a target_family
   *  ItemFamilyNetwork takes responsibility for the memory of \a source_to_target_connectivity
   */
  void addRelation(IItemFamily* source_family, IItemFamily* target_family, IIncrementalItemConnectivity* source_to_target_connectivity) override;

  IIncrementalItemConnectivity* getDependency(IItemFamily* source_family, IItemFamily* target_family) override;
  IIncrementalItemConnectivity* getRelation(IItemFamily* source_family, IItemFamily* target_family) override;

  //! Get a connectivity between the families \a source_family and \a target_family named \a name, whether it is a relation or a dependency
  IIncrementalItemConnectivity* getConnectivity(IItemFamily* source_family, IItemFamily* target_family, const String& name) override;
  IIncrementalItemConnectivity* getConnectivity(IItemFamily* source_family, IItemFamily* target_family, const String& name, bool& is_dependency) override;

  //! Returns, if associated with storage, the connectivity between the families \a source_family and \a target_family named \a name, whether it is a relation or a dependency
  IIncrementalItemConnectivity* getStoredConnectivity(IItemFamily* source_family, IItemFamily* target_family, const String& name) override;
  IIncrementalItemConnectivity* getStoredConnectivity(IItemFamily* source_family, IItemFamily* target_family, const String& name, bool& is_dependency) override;


  //! Get the list of all connectivities, whether they are relations or dependencies
  List<IIncrementalItemConnectivity*> getConnectivities() override;

  //! Get the list of all connectivities (dependencies or relations), children of a family \a source_family or parents of a family \a target_family
  SharedArray<IIncrementalItemConnectivity*> getChildConnectivities(IItemFamily* source_family) override;
  SharedArray<IIncrementalItemConnectivity*> getParentConnectivities(IItemFamily* target_family) override;

  //! Get the list of all dependencies, children of a family \a source_family or parents of a family \a target_family
  SharedArray<IIncrementalItemConnectivity*> getChildDependencies(IItemFamily* source_family) override;
  SharedArray<IIncrementalItemConnectivity*> getParentDependencies(IItemFamily* target_family) override;

  //! Get the list of all relations, children of a family \a source_family or parents of a family \a target_family
  SharedArray<IIncrementalItemConnectivity*> getChildRelations(IItemFamily* source_family) override;
  SharedArray<IIncrementalItemConnectivity*> getParentRelations(IItemFamily* target_family) override;

  //! Get the list of all families
  const std::set<IItemFamily*>& getFamilies() const override {return m_families;}

  SharedArray<IItemFamily*> getFamilies(eSchedulingOrder order) const override;

  //! Schedules the execution of a task, in topological or reverse topological order of the family dependency graph
  void schedule(IItemFamilyNetworkTask task, eSchedulingOrder order = TopologicalOrder) override;

  //! Marks a connectivity as stored. When added, connectivities are described as not stored.
  void setIsStored(IIncrementalItemConnectivity* connectivity) override;

  //! Retrieves information regarding the storage of the connectivity
  bool isStored(IIncrementalItemConnectivity* connectivity) override;

  bool isDeep(IIncrementalItemConnectivity* connectivity) override;


  Integer registerConnectedGraph(IGraph2* graph) override;

  void releaseConnectedGraph(Integer graph_id) override;

  void removeConnectedDoFsFromCells(Int32ConstArrayView local_ids) override;
 private:

  bool m_is_activated = false ;
  using ConnectivityGraph = GraphBaseT<IItemFamily*, IIncrementalItemConnectivity*>;
  mutable DirectedGraphT<IItemFamily*, IIncrementalItemConnectivity*> m_relation_graph;
  mutable DirectedAcyclicGraphT<IItemFamily*, IIncrementalItemConnectivity*> m_dependency_graph;

  // NOTE GG: It would be necessary to use List<Ref<IIncrementalItemConnectivity>>
  // but this changes the interface of this class because of the method
  // getConnectivities(). To check with Stéphane regarding the modification
  List<IIncrementalItemConnectivity*> m_connectivity_list;

  std::map<IIncrementalItemConnectivity*,std::pair<bool,bool>> m_connectivity_status; // bool = is_stored
  std::set<IItemFamily*> m_families;

  UniqueArray<IGraph2*> m_registred_graphs ;

 private:

  void _checkConnectivityName(IIncrementalItemConnectivity* connectivity, const String& name);
  std::pair<IIncrementalItemConnectivity* const, std::pair<bool,bool>>& _getConnectivityStatus(IIncrementalItemConnectivity* connectivity);
  SharedArray<IIncrementalItemConnectivity*> _getConnectivitiesFromGraph(const ConnectivityGraph::ConnectedEdgeSet& connectivity_edges);
  SharedArray<IIncrementalItemConnectivity*> _getConnectivitiesFromGraph(const ConnectivityGraph::ConnectedEdgeSet& connectivity_edges1, const ConnectivityGraph::ConnectedEdgeSet& connectivity_edges2);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif /* ITEMFAMILYNETWORK_H_ */
