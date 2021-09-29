/*
 * GraphDofs.h
 *
 *  Created on: 7 mai 2020
 *      Author: pajon
 */

#ifndef ARCANE_SRC_ARCANE_MESH_GRAPHDOFS_H_
#define ARCANE_SRC_ARCANE_MESH_GRAPHDOFS_H_

#include <memory>
#include <array>

#include "arcane/IGraph2.h"
#include "arcane/IGraphModifier2.h"

#include "arcane/utils/TraceAccessor.h"

#include "arcane/mesh/ItemFamily.h"

#include "arcane/mesh/DoFManager.h"
#include "arcane/mesh/DoFFamily.h"
#include "arcane/mesh/ItemConnectivity.h"
#include "arcane/mesh/ItemConnectivityMng.h"
#include "arcane/mesh/IncrementalItemConnectivity.h"
#include "arcane/mesh/GhostLayerFromConnectivityComputer.h"
#include "arcane/utils/DualUniqueIdMng.h"
#include "arcane/mesh/ParticleFamily.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_MESH_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_MESH_EXPORT GraphDofs
  : public TraceAccessor
  , public IGraph2
  , public IGraphModifier2
{
public:

  typedef DynamicMeshKindInfos::ItemInternalMap ItemInternalMap;

  static const String dualNodeFamilyName() {return "DualNodes";}
  static const String linkFamilyName() {return "Links";}

public:

  GraphDofs(IMesh* mesh, String particle_family_name=ParticleFamily::defaultFamilyName());

public:

  IGraphModifier2* modifier() override { return this; }

#ifdef GRAPH_USE_LEGACY_CONNECTIVITY
  GraphConnectivity const* legacyConnectivity() const override {
    return m_graph_connectivity.get() ;
    //return GraphConnectivity{ &m_links_connectivity,m_connectivities,m_dual_node_to_connectivity_index };
  }
#endif

#ifdef GRAPH_USE_INCREMENTAL_CONNECTIVITY
  GraphIncrementalConnectivity const* incrementalConnectivity() const override {
    return m_graph_connectivity.get() ;
    //return GraphIncrementalConnectivity{ m_links_incremental_connectivity,m_incremental_connectivities,m_dual_node_to_connectivity_index };
  }
#endif
  IItemFamily* dualNodeFamily() override { return &m_dual_node_family;}
  const IItemFamily* dualNodeFamily() const override { return &m_dual_node_family;}

  IItemFamily* linkFamily() override     { return &m_link_family; }
  const IItemFamily* linkFamily() const override { return &m_link_family;}


  inline Integer nbLink() const override{ return linkFamily()->nbItem(); }
  inline Integer nbDualNode() const override{ return dualNodeFamily()->nbItem(); }

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

  void endUpdate() override;

  void updateAfterMeshChanged() override ;

  void printDualNodes() const override ;
  void printLinks() const override;

private:

  String _className() const { return "GraphDofs"; }
  inline Integer _connectivityIndex(Integer dual_node_IT) const {
    ARCANE_ASSERT((dual_node_IT < NB_BASIC_ITEM_TYPE),
                  ("dual node item type must be IT_DualNode, IT_DualEdge, IT_DualFace, IT_DualCell or IT_DualParticle"));
    return m_connectivity_indexes_per_type[dual_node_IT];
  }

  private:
  IItemFamily* _dualItemFamily(Arcane::eItemKind kind)
  {
    if(kind==IK_Particle)
    {
      return m_mesh->findItemFamily(kind, m_particle_family_name, false);
    }
    else
    {
       return m_mesh->itemFamily(kind);
    }
  }

  Int64 _doFUid(Integer dual_item_kind, Item const& item)
  {
    switch(dual_item_kind)
    {
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
        return -1 ;
    }
  }

  void _allocateGraph() ;

  IMesh* m_mesh = nullptr;
  IItemFamilyNetwork* m_item_family_network = nullptr ;
  bool m_graph_allocated = false ;

  ItemConnectivityMng m_connectivity_mng;
  DoFManager m_dof_mng;
  DoFFamily& m_dual_node_family;
  DoFFamily& m_link_family;

#ifdef GRAPH_USE_LEGACY_CONNECTIVITY
  UniqueArray<ItemConnectivity> m_connectivities;
  mutable ItemMultiArrayConnectivity m_links_connectivity;
#endif
#ifdef GRAPH_USE_INCREMENTAL_CONNECTIVITY
  UniqueArray<Arcane::mesh::IncrementalItemConnectivity*> m_incremental_connectivities;
  UniqueArray<Arcane::mesh::IncrementalItemConnectivity*> m_dual2dof_incremental_connectivities;
  Arcane::mesh::IncrementalItemConnectivity* m_links_incremental_connectivity = nullptr;
  std::unique_ptr<GraphIncrementalConnectivity> m_graph_connectivity ;
#endif
  std::vector<std::unique_ptr<Arcane::GhostLayerFromConnectivityComputer>> m_ghost_layer_computers ;
  Int32UniqueArray m_connectivity_indexes_per_type;
  std::array<Integer,NB_BASIC_ITEM_TYPE> m_dualnode_kinds = {IT_DualNode, IT_DualEdge, IT_DualFace, IT_DualCell, IT_DualParticle } ;
  ItemScalarProperty<Integer> m_dual_node_to_connectivity_index;


  bool m_update_sync_info = false;

  String m_particle_family_name;

};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_MESH_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------r------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#endif /* ARCANE_SRC_ARCANE_MESH_GRAPHDOFS_H_ */
