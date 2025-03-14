// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* GraphDoFs.h                                                 (C) 2000-2025 */
/*                                                                           */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MESH_GRAPHDOFS_H
#define ARCANE_MESH_GRAPHDOFS_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/IGraph2.h"
#include "arcane/core/IGraphModifier2.h"
#include "arcane/core/IItemConnectivityMng.h"

#include "arcane/utils/TraceAccessor.h"
#include "arcane/utils/Convert.h"

#include "arcane/mesh/ItemFamily.h"
#include "arcane/mesh/DoFManager.h"
#include "arcane/mesh/DoFFamily.h"
#include "arcane/mesh/ItemConnectivity.h"
#include "arcane/mesh/IncrementalItemConnectivity.h"
#include "arcane/mesh/GhostLayerFromConnectivityComputer.h"
#include "arcane/mesh/DualUniqueIdMng.h"
#include "arcane/mesh/ParticleFamily.h"
#include "arcane/mesh/IndexedItemConnectivityAccessor.h"

#include <memory>
#include <array>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename T>
class GraphConnectivityObserverT
: public IGraphConnectivityObserver
{
 public:

  GraphConnectivityObserverT(T* parent)
  : m_parent(parent)
  {}

  virtual ~GraphConnectivityObserverT() {}

  void notifyUpdateConnectivity()
  {
    m_parent->updateGraphConnectivity();
  }

 private:

  T* m_parent = nullptr;
};

template <typename T>
class GraphObserverT
: public IGraphObserver
{
 public:

  explicit GraphObserverT(T* parent)
  : m_parent(parent)
  {}

  virtual ~GraphObserverT() {}

  void notifyUpdate()
  {
    m_parent->notifyGraphUpdate();
  }

 private:

  T* m_parent = nullptr;
};

class GraphDoFs;

class ARCANE_MESH_EXPORT GraphIncrementalConnectivity
: public IGraphConnectivity
{
 public:

  GraphIncrementalConnectivity(IItemFamily const* dualnode_family,
                               IItemFamily const* link_family,
                               Arcane::mesh::IncrementalItemConnectivity* dualnode_connectivity,
                               Arcane::mesh::IncrementalItemConnectivity* link_connectivity,
                               UniqueArray<Arcane::mesh::IncrementalItemConnectivity*> const& dualitem_connectivities,
                               ItemScalarProperty<Integer> const& dualnode_to_connectivity)
  : m_dualnode_family(dualnode_family)
  , m_link_family(link_family)
  , m_dualnode_connectivity(dualnode_connectivity)
  , m_link_connectivity(link_connectivity)
  , m_dualitem_connectivities(dualitem_connectivities)
  , m_dualnode_to_connectivity_index(dualnode_to_connectivity)
  {}

  GraphIncrementalConnectivity(GraphIncrementalConnectivity const& rhs)
  : m_dualnode_family(rhs.m_dualnode_family)
  , m_link_family(rhs.m_link_family)
  , m_dualnode_connectivity(rhs.m_dualnode_connectivity)
  , m_link_connectivity(rhs.m_link_connectivity)
  , m_dualitem_connectivities(rhs.m_dualitem_connectivities)
  , m_dualnode_to_connectivity_index(rhs.m_dualnode_to_connectivity_index)
  {}

  GraphIncrementalConnectivity(GraphDoFs* graph);

  virtual ~GraphIncrementalConnectivity() {}

  inline Item dualItem(const DoF& dualNode) const
  {
    auto dualitem_connectivity_accessor = m_dualitem_connectivities[m_dualnode_to_connectivity_index[dualNode]]->connectivityAccessor();
    return dualitem_connectivity_accessor(ItemLocalId(dualNode))[0];
  }

  inline DoFVectorView links(const DoF& dualNode) const
  {
    return m_dualnode_connectivity->connectivityAccessor()(ItemLocalId(dualNode));
  }

  inline DoFVectorView dualNodes(const DoF& link) const
  {
    return m_link_connectivity->connectivityAccessor()(ItemLocalId(link));
  }

 private:

  IItemFamily const* m_dualnode_family = nullptr;
  IItemFamily const* m_link_family = nullptr;
  Arcane::mesh::IncrementalItemConnectivity* m_dualnode_connectivity = nullptr;
  Arcane::mesh::IncrementalItemConnectivity* m_link_connectivity = nullptr;
  UniqueArray<Arcane::mesh::IncrementalItemConnectivity*> const& m_dualitem_connectivities;
  ItemScalarProperty<Integer> const& m_dualnode_to_connectivity_index;
};

class ARCANE_MESH_EXPORT GraphDoFs
: public TraceAccessor
, public IGraph2
, public IGraphModifier2
{
 public:

  friend class GraphIncrementalConnectivity;

  typedef DynamicMeshKindInfos::ItemInternalMap ItemInternalMap;

  static const String dualNodeFamilyName() { return "DualNodes"; }
  static const String linkFamilyName() { return "Links"; }

 public:

  GraphDoFs(IMesh* mesh, String particle_family_name = ParticleFamily::defaultFamilyName());

  virtual ~GraphDoFs() {}

 public:

  IGraphModifier2* modifier() override { return this; }

  IGraphConnectivity const* connectivity() const override
  {
    return m_graph_connectivity.get();
  }

  Integer registerNewGraphConnectivityObserver(IGraphConnectivityObserver* observer) override
  {
    Integer id = CheckedConvert::toInteger(m_connectivity_observer.size());
    m_connectivity_observer.push_back(std::unique_ptr<IGraphConnectivityObserver>(observer));
    return id;
  }

  void releaseGraphConnectivityObserver(Integer observer_id) override
  {
    if ((observer_id >= 0) && (observer_id < (Integer)m_connectivity_observer.size()))
      m_connectivity_observer[observer_id].reset();
  }

  Integer registerNewGraphObserver(IGraphObserver* observer) override
  {
    Integer id = CheckedConvert::toInteger(m_graph_observer.size());
    m_graph_observer.push_back(std::unique_ptr<IGraphObserver>(observer));
    return id;
  }

  void releaseGraphObserver(Integer observer_id) override
  {
    if ((observer_id >= 0) && (observer_id < (Integer)m_graph_observer.size()))
      m_graph_observer[observer_id].reset();
  }

  IItemFamily* dualNodeFamily() override { return m_dual_node_family->itemFamily(); }
  const IItemFamily* dualNodeFamily() const override { return m_dual_node_family->itemFamily(); }

  IItemFamily* linkFamily() override { return m_link_family->itemFamily(); }
  const IItemFamily* linkFamily() const override { return m_link_family->itemFamily(); }

  inline Integer nbLink() const override { return linkFamily()->nbItem(); }
  inline Integer nbDualNode() const override { return dualNodeFamily()->nbItem(); }

  //! Ajout de liaisons dans le graphe avec un nombre fixe de noeuds dual par liaison
  void addLinks(Integer nb_link,
                Integer nb_dual_nodes_per_link,
                Int64ConstArrayView links_infos) override;

  //! Ajout de noeuds duaux dans le graphe avec un type fixe d'item dual par noeud
  void addDualNodes(Integer graph_nb_dual_node,
                    Integer dual_node_kind,
                    Int64ConstArrayView dual_nodes_infos) override;
  void addDualNodes(Integer graph_nb_dual_node,
                    Int64ConstArrayView dual_nodes_infos) override;
  void removeDualNodes(Int32ConstArrayView dual_node_local_ids) override;
  void removeLinks(Int32ConstArrayView link_local_ids) override;

  void removeConnectedItemsFromCells(Int32ConstArrayView cell_local_ids) override;

  bool isUpdated() override;

  void endUpdate() override;

  void updateAfterMeshChanged() override;

  void printDualNodes() const override;
  void printLinks() const override;

 private:

  String _className() const { return "GraphDoFs"; }
  inline Integer _connectivityIndex(Integer dual_node_IT) const
  {
    ARCANE_ASSERT((dual_node_IT < NB_BASIC_ITEM_TYPE),
                  ("dual node item type must be IT_DualNode, IT_DualEdge, IT_DualFace, IT_DualCell or IT_DualParticle"));
    return m_connectivity_indexes_per_type[dual_node_IT];
  }

 private:

  IItemFamily* _dualItemFamily(Arcane::eItemKind kind)
  {
    if (kind == IK_Particle) {
      return m_mesh->findItemFamily(kind, m_particle_family_name, false);
    }
    else {
      return m_mesh->itemFamily(kind);
    }
  }

  Int64 _doFUid(Integer dual_item_kind, Item const& item)
  {
    switch (dual_item_kind) {
    case IT_DualNode:
      return Arcane::DualUniqueIdMng::uniqueIdOf<Node>(item.toNode());
    case IT_DualEdge:
      return Arcane::DualUniqueIdMng::uniqueIdOf<Edge>(item.toEdge());
    case IT_DualFace:
      return Arcane::DualUniqueIdMng::uniqueIdOf<Face>(item.toFace());
    case IT_DualCell:
      return Arcane::DualUniqueIdMng::uniqueIdOf<Cell>(item.toCell());
    case IT_DualParticle:
      return Arcane::DualUniqueIdMng::uniqueIdOf<Particle>(item.toParticle());
    default:
      return -1;
    }
  }

  void _allocateGraph();

  IMesh* m_mesh = nullptr;
  IItemFamilyNetwork* m_item_family_network = nullptr;
  bool m_graph_allocated = false;
  Integer m_graph_id = -1;

  DoFManager m_dof_mng;
  IItemConnectivityMng* m_connectivity_mng;
  IDoFFamily* m_dual_node_family = nullptr;
  IDoFFamily* m_link_family = nullptr;

  UniqueArray<Arcane::mesh::IncrementalItemConnectivity*> m_incremental_connectivities;
  UniqueArray<Arcane::mesh::IncrementalItemConnectivity*> m_dual2dof_incremental_connectivities;
  Arcane::mesh::IncrementalItemConnectivity* m_dualnodes_incremental_connectivity = nullptr;
  Arcane::mesh::IncrementalItemConnectivity* m_links_incremental_connectivity = nullptr;
  std::unique_ptr<GraphIncrementalConnectivity> m_graph_connectivity;
  std::vector<std::unique_ptr<Arcane::IGraphConnectivityObserver>> m_connectivity_observer;

  std::vector<std::unique_ptr<Arcane::IGraphObserver>> m_graph_observer;

  std::vector<std::unique_ptr<Arcane::GhostLayerFromConnectivityComputer>> m_ghost_layer_computers;
  Int32UniqueArray m_connectivity_indexes_per_type;
  std::array<Integer, NB_BASIC_ITEM_TYPE> m_dualnode_kinds = { IT_DualNode, IT_DualEdge, IT_DualFace, IT_DualCell, IT_DualParticle };
  ItemScalarProperty<Integer> m_dual_node_to_connectivity_index;

  UniqueArray<Int32> m_detached_dualnode_lids;
  UniqueArray<Int32> m_detached_link_lids;

  bool m_update_sync_info = false;

  String m_particle_family_name;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
