﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemFamilyNetwork.h                                         (C) 2000-2017 */
/*                                                                           */
/* ItemFamily relations through their connectivities.                        */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ITEMFAMILYNETWORK_H_ 
#define ARCANE_ITEMFAMILYNETWORK_H_ 
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

#include "arcane/IItemFamilyNetwork.h"
#include "arcane/utils/NotImplementedException.h" //tmp !

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE
ARCANE_MESH_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_MESH_EXPORT ItemFamilyNetwork : public IItemFamilyNetwork
{
public:

  /** Constructeur de la classe */
  ItemFamilyNetwork(ITraceMng* trace_mng)
  : m_trace_mng(trace_mng)
  , m_relation_graph(trace_mng)
  , m_dependency_graph(trace_mng)
{}

  /** Destructeur de la classe */
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

  /*! Ajoute une dépendance entre deux familles ; un élément de \a master_family est constitué d'éléments de \a slave_family.
   *  La responsabilité de la mémoire de \a master_to_slave_connectivity est prise en charge par ItemFamilyNetwork
   */
  void addDependency(IItemFamily* master_family, IItemFamily* slave_family, IIncrementalItemConnectivity* master_to_slave_connectivity, bool is_deep_connectivity) override;

  /*! Ajoute une relation entre deux familles ; un élément de \a source_family est connecté à un ou plusieurs éléments de \a target_family
   *  La responsabilité de la mémoire de \a source_to_target_connectivity est prise en charge par ItemFamilyNetwork
   */
  void addRelation(IItemFamily* source_family, IItemFamily* target_family, IIncrementalItemConnectivity* source_to_target_connectivity) override;

  //! Obtenir une connectivité entre les familles \a source_family et \a target_family de nom \a name, qu'elle soit une relation ou une dépendance
  IIncrementalItemConnectivity* getConnectivity(IItemFamily* source_family, IItemFamily* target_family, const String& name) override;
  IIncrementalItemConnectivity* getConnectivity(IItemFamily* source_family, IItemFamily* target_family, const String& name, bool& is_dependency) override;

  //! Retourne, si elle es associée à un stockage, la connectivité entre les familles \a source_family et \a target_family de nom \a name, qu'elle soit une relation ou une dépendance
  IIncrementalItemConnectivity* getStoredConnectivity(IItemFamily* source_family, IItemFamily* target_family, const String& name) override;
  IIncrementalItemConnectivity* getStoredConnectivity(IItemFamily* source_family, IItemFamily* target_family, const String& name, bool& is_dependency) override;


  //! Obtenir la liste de toutes les connectivités, qu'elles soient relation ou dépendance
  List<IIncrementalItemConnectivity*> getConnectivities() override;

  //! Obtenir la liste de toutes les connectivités (dépendances ou relations), filles d'une famille \a source_family ou parentes d'une famille \a target_family
  SharedArray<IIncrementalItemConnectivity*> getChildConnectivities(IItemFamily* source_family) override;
  SharedArray<IIncrementalItemConnectivity*> getParentConnectivities(IItemFamily* target_family) override;

  //! Obtenir la liste de toutes les dépendances, filles d'une famille \a source_family ou parentes d'une famille \a target_family
  SharedArray<IIncrementalItemConnectivity*> getChildDependencies(IItemFamily* source_family) override;
  SharedArray<IIncrementalItemConnectivity*> getParentDependencies(IItemFamily* target_family) override;

  //! Obtenir la liste de toutes les relations, filles d'une famille \a source_family ou parentes d'une famille \a target_family
  SharedArray<IIncrementalItemConnectivity*> getChildRelations(IItemFamily* source_family) override;
  SharedArray<IIncrementalItemConnectivity*> getParentRelations(IItemFamily* target_family) override;

  //! Obtenir la liste de toutes les familles
  const std::set<IItemFamily*>& getFamilies() const override {return m_families;}

  SharedArray<IItemFamily*> getFamilies(eSchedulingOrder order) const override;

  //! Ordonnance l'exécution d'une tâche, dans l'ordre topologique ou topologique inverse du graphe de dépendance des familles
  void schedule(IItemFamilyNetworkTask task, eSchedulingOrder order = TopologicalOrder) override;

  //! Positionne une connectivité comme étant stockée. A l'ajout les connectivités sont décrites comme non stockée.
  void setIsStored(IIncrementalItemConnectivity* connectivity) override;

  //! Récupère l'information relative au stockage de la connectivité
  bool isStored(IIncrementalItemConnectivity* connectivity) override;

  bool isDeep(IIncrementalItemConnectivity* connectivity) override;

private:
  ITraceMng* m_trace_mng = nullptr;
  bool m_is_activated = false ;
  using ConnectivityGraph = GraphBaseT<IItemFamily*, IIncrementalItemConnectivity*>;
  mutable DirectedGraphT<IItemFamily*, IIncrementalItemConnectivity*> m_relation_graph;
  mutable DirectedAcyclicGraphT<IItemFamily*, IIncrementalItemConnectivity*> m_dependency_graph;
  List<IIncrementalItemConnectivity*> m_connectivity_list;
  std::map<IIncrementalItemConnectivity*,std::pair<bool,bool>> m_connectivity_status; // bool = is_stored
  std::set<IItemFamily*> m_families;

private:
  void _checkConnectivityName(IIncrementalItemConnectivity* connectivity, const String& name);
  std::pair<IIncrementalItemConnectivity* const, std::pair<bool,bool>>& _getConnectivityStatus(IIncrementalItemConnectivity* connectivity);
  SharedArray<IIncrementalItemConnectivity*> _getConnectivitiesFromGraph(const ConnectivityGraph::ConnectedEdgeSet& connectivity_edges);
  SharedArray<IIncrementalItemConnectivity*> _getConnectivitiesFromGraph(const ConnectivityGraph::ConnectedEdgeSet& connectivity_edges1, const ConnectivityGraph::ConnectedEdgeSet& connectivity_edges2);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_MESH_END_NAMESPACE
ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif /* ITEMFAMILYNETWORK_H_ */
