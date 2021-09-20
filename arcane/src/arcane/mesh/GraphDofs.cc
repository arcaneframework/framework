/*
 * GraphDofs.cc
 *
 *  Created on: 7 mai 2020
 *      Author: pajon
 */
#include "GraphDofs.h"

#include "arcane/IMesh.h"
#include "arcane/IItemConnectivity.h"
#include "arcane/IIncrementalItemConnectivity.h"
#include "arcane/mesh/ConnectivityNewWithDependenciesTypes.h"

#include "arcane/MeshUtils.h"
#include "arcane/MathUtils.h"


#include "arcane/utils/ArgumentException.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_MESH_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

GraphDofs::
GraphDofs(IMesh* mesh, String particle_family_name)
  : TraceAccessor(mesh->traceMng())
  , m_mesh(mesh)
  , m_connectivity_mng(mesh->traceMng())
  , m_dof_mng(mesh,&m_connectivity_mng)
  , m_dual_node_family(m_dof_mng.family(GraphDofs::dualNodeFamilyName(),true))
  , m_link_family(m_dof_mng.family(GraphDofs::linkFamilyName(),true))
#ifdef GRAPH_USE_LEGACY_CONNECTIVITY
  , m_links_connectivity{&m_link_family, &m_dual_node_family,
                       ItemMultiArrayProperty<Int32>{},
                       "links_to_dual_nodes"}
   ,
#endif
  , m_update_sync_info(false)
  , m_particle_family_name(particle_family_name)
{
}

void GraphDofs::
_allocateGraph()
{
  if(m_graph_allocated) return ;

  auto connectivity_index = 0;
  m_connectivity_indexes_per_type.resize(NB_BASIC_ITEM_TYPE,-1);
  m_connectivity_indexes_per_type[IT_DualNode] = connectivity_index++;
  m_connectivity_indexes_per_type[IT_DualEdge] = connectivity_index++;
  m_connectivity_indexes_per_type[IT_DualFace] = connectivity_index++;
  m_connectivity_indexes_per_type[IT_DualCell] = connectivity_index++;
  m_connectivity_indexes_per_type[IT_DualParticle] = connectivity_index++;

#ifdef GRAPH_USE_INCREMENTAL_CONNECTIVITY

  m_item_family_network = m_mesh->itemFamilyNetwork() ;
  if(m_item_family_network==nullptr)
    traceMng()->fatal()<<"ARCANE_GRAPH_CONNECTIVITY_POLICY need to be activated" ;

  /*
  auto dual2links_incremental_connectivity =
      new IncrementalItemConnectivity ( dualNodeFamily(),
                                        linkFamily(),
                                        mesh::connectivityName(dualNodeFamily(),linkFamily())) ;
  if(m_item_family_network)
  m_item_family_network->addDependency(dualNodeFamily(),linkFamily(),dual2links_incremental_connectivity);
  //m_item_family_network->addDependency(dualNodeFamily(),linkFamily(),m_links_incremental_connectivity,false);
   */
  m_links_incremental_connectivity =
      new IncrementalItemConnectivity ( linkFamily(),
                                        dualNodeFamily(),
                                        mesh::connectivityName(linkFamily(),dualNodeFamily())) ;
  if(m_item_family_network)
    //m_item_family_network->addRelation(linkFamily(),dualNodeFamily(),m_links_incremental_connectivity);
    m_item_family_network->addDependency(linkFamily(),dualNodeFamily(),m_links_incremental_connectivity,false);


  m_incremental_connectivities.resize(NB_DUAL_ITEM_TYPE,nullptr) ;
  std::array<int,NB_DUAL_ITEM_TYPE> dual_node_kinds = {IT_DualCell,IT_DualFace,IT_DualEdge,IT_DualNode,IT_DualParticle} ;
  for(auto dual_node_kind : dual_node_kinds)
  {
    IItemFamily* dual_item_family = _dualItemFamily(dualItemKind(dual_node_kind)) ;
    if(dual_item_family)
    {
      /*
      auto dual2dof_incremental_connectivity =
              new IncrementalItemConnectivity ( dual_item_family,
                                                dualNodeFamily (),
                                                mesh::connectivityName(dual_item_family,dualNodeFamily()));
      if(m_item_family_network)
      m_item_family_network->addDependency(dual_item_family,dualNodeFamily(),dual2dof_incremental_connectivity);
      //m_item_family_network->addDependency(dual_item_family,dualNodeFamily(),dual_item_incremental_connectivity,false);
      */
      auto dof2dual_incremental_connectivity =
              new IncrementalItemConnectivity ( dualNodeFamily (),
                                                dual_item_family,
                                                mesh::connectivityName(dualNodeFamily(),dual_item_family));
      m_incremental_connectivities[_connectivityIndex(dual_node_kind)] = dof2dual_incremental_connectivity ;
      if(m_item_family_network)
        //m_item_family_network->addRelation(dualNodeFamily(),dual_item_family,dof2dual_incremental_connectivity);
        m_item_family_network->addDependency(dualNodeFamily(),dual_item_family,dof2dual_incremental_connectivity,false);
    }
  }
  m_graph_connectivity.reset( new GraphIncrementalConnectivity(m_links_incremental_connectivity,
                                                               m_incremental_connectivities,
                                                               m_dual_node_to_connectivity_index)) ;
#endif
  m_graph_allocated = true ;
}



/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void GraphDofs::
addLinks(Integer nb_link,
         Integer nb_dual_nodes_per_link,
         Int64ConstArrayView links_infos)
{
  Trace::Setter mci(traceMng(),_className());
  if(!m_graph_allocated)
    _allocateGraph() ;

  mesh::DoFFamily& link_family = m_dof_mng.family(GraphDofs::linkFamilyName());
  mesh::DoFFamily& dual_node_family = m_dof_mng.family(GraphDofs::dualNodeFamilyName());

  // Extract link infos
  Int64UniqueArray link_uids, connected_dual_node_uids;
  link_uids.reserve(nb_link);
  connected_dual_node_uids.reserve(nb_link * nb_dual_nodes_per_link);
  for (auto links_infos_index = 0; links_infos_index < links_infos.size() ;) {
    auto link_uid = links_infos[links_infos_index++];
    link_uids.add(link_uid);
    connected_dual_node_uids.addRange(
        links_infos.subConstView(links_infos_index, nb_dual_nodes_per_link));
    links_infos_index += nb_dual_nodes_per_link;
  }

  Int32UniqueArray link_lids(link_uids.size());
  link_family.addDoFs(link_uids,link_lids);
  link_family.endUpdate();

  // resize connectivity
#ifdef GRAPH_USE_LEGACY_CONNECTIVITY
  IntegerUniqueArray nb_connected_dual_nodes_per_link{
    m_links_connectivity.itemProperty().dim2Sizes()
  };
  m_nb_connected_dual_nodes_per_link.resize(link_family.maxLocalId());
  ENUMERATE_DOF(inewlink, link_family.view(link_lids))
  {
    m_nb_connected_dual_nodes_per_link[inewlink->localId()] = nb_dual_nodes_per_link;
    //m_links_incremental_connectivity->addConnectedItems(ItemLocalId(*inewlink),nb_dual_nodes_per_link) ;
  }
  m_links_connectivity.itemProperty().resize(
      &link_family, m_nb_connected_dual_nodes_per_link, NULL_ITEM_LOCAL_ID);
#endif
  // fill connectivity
  Int32UniqueArray connected_dual_nodes_lids(nb_link * nb_dual_nodes_per_link);
  dual_node_family.itemsUniqueIdToLocalId(
      connected_dual_nodes_lids.view(), connected_dual_node_uids.constView(), true);

#ifdef GRAPH_USE_LEGACY_CONNECTIVITY
  auto& link_connectivity_property = m_links_connectivity.itemProperty();
#endif
  auto link_index = 0;
  ENUMERATE_DOF(inewlink, link_family.view(link_lids))
  {
#ifdef GRAPH_USE_LEGACY_CONNECTIVITY
    link_connectivity_property[inewlink].copy(
        connected_dual_nodes_lids.subConstView(link_index, nb_dual_nodes_per_link));
#endif
#ifdef GRAPH_USE_INCREMENTAL_CONNECTIVITY
    m_links_incremental_connectivity->notifySourceItemAdded(ItemLocalId(*inewlink)) ;
    for ( auto lid : connected_dual_nodes_lids.subConstView(link_index, nb_dual_nodes_per_link))
      m_links_incremental_connectivity->addConnectedItem(ItemLocalId(*inewlink),ItemLocalId(lid)) ;
#endif
    link_index += nb_dual_nodes_per_link;
  }
  m_update_sync_info = true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void GraphDofs::
addDualNodes(Integer graph_nb_dual_node,
             Integer dual_node_kind,
             Int64ConstArrayView dual_nodes_infos)
{

  Trace::Setter mci(traceMng(),_className());
  if(!m_graph_allocated)
    _allocateGraph() ;

  // Size m_connecitivities if not yet done
#ifdef GRAPH_USE_LEGACY_CONNECTIVITY
  if (m_connectivities.empty()) {
    m_connectivities.resize(NB_DUAL_ITEM_TYPE,
        ItemConnectivity{
            nullptr, nullptr, ItemScalarProperty<Int32>{}, "empty_connectivity" });
  }
#endif

  Int64UniqueArray dual_node_uids, dual_item_uids;
  dual_node_uids.reserve(graph_nb_dual_node);
  dual_item_uids.reserve(graph_nb_dual_node);
  for (auto infos_index = 0; infos_index < dual_nodes_infos.size();) {
    dual_node_uids.add(dual_nodes_infos[infos_index++]);
    dual_item_uids.add(dual_nodes_infos[infos_index++]);
  }

  Int32UniqueArray dual_node_lids(dual_node_uids.size());
  auto& dual_node_family = m_dof_mng.family(GraphDofs::dualNodeFamilyName());
  dual_node_family.addDoFs(dual_node_uids, dual_node_lids);
  dual_node_family.endUpdate();

  IItemFamily* dual_item_family = _dualItemFamily(dualItemKind(dual_node_kind)) ;
#ifdef GRAPH_USE_LEGACY_CONNECTIVITY
  auto& dual_item_connectivity = m_connectivities[_connectivityIndex(dual_node_kind)];
  if (dual_item_connectivity.name() == "empty_connectivity")
  {
    dual_item_connectivity = ItemConnectivity(dualNodeFamily(), dual_item_family,
        ItemScalarProperty<Int32>{}, "dual_node_to_" + dual_item_family->name());
  }
  auto& connectivity_property = dual_item_connectivity.itemProperty();
  connectivity_property.resize(&dual_node_family, NULL_ITEM_LOCAL_ID);
#endif


#ifdef GRAPH_USE_INCREMENTAL_CONNECTIVITY
  auto incremental_dual_item_connectivity = m_incremental_connectivities[_connectivityIndex(dual_node_kind)] ;
#endif
  Int32UniqueArray dual_item_lids(dual_item_uids.size());
  dual_item_family->itemsUniqueIdToLocalId(dual_item_lids, dual_item_uids);

  ENUMERATE_DOF(idual_node,dual_node_family.view(dual_node_lids))
  {
#ifdef GRAPH_USE_LEGACY_CONNECTIVITY
    connectivity_property[idual_node] = dual_item_lids[idual_node.index()];
#endif
#ifdef GRAPH_USE_INCREMENTAL_CONNECTIVITY
    incremental_dual_item_connectivity->notifySourceItemAdded(ItemLocalId(*idual_node)) ;
    incremental_dual_item_connectivity->addConnectedItem(ItemLocalId(*idual_node),ItemLocalId(dual_item_lids[idual_node.index()])) ;
#endif
  }

  m_dual_node_to_connectivity_index.resize(&dual_node_family, _connectivityIndex(dual_node_kind));

  m_update_sync_info = true;
}

void GraphDofs::
addDualNodes(Integer graph_nb_dual_node,
             Int64ConstArrayView dual_nodes_infos)
{

  Trace::Setter mci(traceMng(),_className());
  if(!m_graph_allocated)
    _allocateGraph() ;

  // Size m_connecitivities if not yet done
#ifdef GRAPH_USE_LEGACY_CONNECTIVITY
  if (m_connectivities.empty())
  {

    m_connectivities.resize(NB_DUAL_ITEM_TYPE,
        ItemConnectivity{
            nullptr, nullptr, ItemScalarProperty<Int32>{}, "empty_connectivity" });
    Integer index = 0 ;
    for(auto& connectivity : m_connectivities)
    {
        Integer dual_node_kind = m_dualnode_kinds[index] ;

        IItemFamily* dual_item_family = _dualItemFamily(dualItemKind(dual_node_kind)) ;
        if(dual_item_family)
        {
          connectivity = ItemConnectivity(dualNodeFamily(), dual_item_family,
                                          ItemScalarProperty<Int32>{}, "dual_node_to_" + dual_item_family->name());

        }
        ++index ;
    }
  }
#endif

  std::map<Int64,std::pair<Int64UniqueArray,Int64UniqueArray>> dual_info_per_kind;
  for (auto infos_index = 0; infos_index < dual_nodes_infos.size();)
  {
    Int64 dual_node_kind = dual_nodes_infos[infos_index++] ;
    auto& info = dual_info_per_kind[dual_node_kind] ;
    auto& dual_node_uids = info.first ;
    auto& dual_item_uids = info.second ;
    if(dual_node_uids.size()==0)
    {
        dual_node_uids.reserve(graph_nb_dual_node) ;
        dual_item_uids.reserve(graph_nb_dual_node) ;
    }
    dual_node_uids.add(dual_nodes_infos[infos_index++]);
    dual_item_uids.add(dual_nodes_infos[infos_index++]);
  }


  for(Integer index = 0 ;index< NB_DUAL_ITEM_TYPE;++index)
  {
    Integer dual_node_kind = m_dualnode_kinds[index] ;
    auto& info = dual_info_per_kind[dual_node_kind] ;
    auto& dual_node_uids = info.first ;
    auto& dual_item_uids = info.second ;

    Int32UniqueArray dual_node_lids(dual_node_uids.size());
    auto& dual_node_family = m_dof_mng.family(GraphDofs::dualNodeFamilyName());
    dual_node_family.addDoFs(dual_node_uids, dual_node_lids);
    dual_node_family.endUpdate();

#ifdef GRAPH_USE_LEGACY_CONNECTIVITY
    auto& dual_item_connectivity = m_connectivities[index];
    auto& connectivity_property = dual_item_connectivity.itemProperty();
    connectivity_property.resize(&dual_node_family, NULL_ITEM_LOCAL_ID);
#endif
#ifdef GRAPH_USE_INCREMENTAL_CONNECTIVITY
    auto incremental_dual_item_connectivity = m_incremental_connectivities[index] ;
#endif
    IItemFamily* dual_item_family = _dualItemFamily(dualItemKind(dual_node_kind)) ;
    if(dual_item_family)
    {
      Int32UniqueArray dual_item_lids(dual_item_uids.size());
      dual_item_family->itemsUniqueIdToLocalId(dual_item_lids, dual_item_uids);

      ENUMERATE_DOF(idual_node,dual_node_family.view(dual_node_lids))
      {
#ifdef GRAPH_USE_LEGACY_CONNECTIVITY
        connectivity_property[idual_node] = dual_item_lids[idual_node.index()];
#endif
#ifdef GRAPH_USE_INCREMENTAL_CONNECTIVITY
        incremental_dual_item_connectivity->notifySourceItemAdded(ItemLocalId(*idual_node)) ;
        incremental_dual_item_connectivity->addConnectedItem(ItemLocalId(*idual_node),ItemLocalId(dual_item_lids[idual_node.index()])) ;
#endif
      }

      m_dual_node_to_connectivity_index.resize(&dual_node_family, _connectivityIndex(dual_node_kind));
    }
  }
  m_update_sync_info = true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void GraphDofs::
removeDualNodes(Int32ConstArrayView dual_node_local_ids)
{
  Trace::Setter mci(traceMng(),_className());
  //m_dual_node_family.removeItems(dual_node_local_ids);
  m_dof_mng.family(GraphDofs::dualNodeFamilyName()).removeDoFs(dual_node_local_ids);
  if(dual_node_local_ids.size()>0)
      m_update_sync_info = true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void GraphDofs::
removeLinks(Int32ConstArrayView link_local_ids)
{
  Trace::Setter mci(traceMng(),_className());
  //m_link_family.removeItems(link_local_ids);
  m_dof_mng.family(GraphDofs::linkFamilyName()).removeDoFs(link_local_ids);
  if(link_local_ids.size()>0)
      m_update_sync_info = true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void GraphDofs::
endUpdate()
{
#ifdef GRAPH_USE_LEGACY_CONNECTIVITY
  traceMng()->info()<<"EndUpdate()"<<m_connectivities.size();
  m_ghost_layer_computers.reserve(m_connectivities.size()) ;
  for( auto& connectivity : m_connectivities)
  {
      if(connectivity.sourceFamily())
      {
        auto ghost_builder = new Arcane::GhostLayerFromConnectivityComputer(&connectivity) ;
        m_ghost_layer_computers.push_back(std::unique_ptr<GhostLayerFromConnectivityComputer>(ghost_builder)) ;
        IItemConnectivitySynchronizer* synchronizer = m_dof_mng.connectivityMng()->createSynchronizer(&connectivity, ghost_builder);
        synchronizer->synchronize();

        // Save your connectivity
        m_connectivity_mng.registerConnectivity(&connectivity);
      }
  }
  if(m_links_connectivity.sourceFamily())
  {
    auto ghost_builder = new Arcane::GhostLayerFromConnectivityComputer(&m_links_connectivity);
    m_ghost_layer_computers.push_back(std::unique_ptr<GhostLayerFromConnectivityComputer>(ghost_builder)) ;
    IItemConnectivitySynchronizer* synchronizer = m_dof_mng.connectivityMng()->createSynchronizer(&m_links_connectivity, ghost_builder);
    synchronizer->synchronize();

    // Save your connectivity
    m_connectivity_mng.registerConnectivity(&m_links_connectivity);
  }
#endif
}

void GraphDofs::updateAfterMeshChanged()
{

  if(!m_graph_allocated) return ;

#ifdef GRAPH_USE_LEGACY_CONNECTIVITY
  Integer index = 0 ;
  for( auto& connectivity : m_connectivities)
  {
      if (connectivity.sourceFamily() && !m_connectivity_mng.isUpToDate(&connectivity))
      {
        // Handle added nodes : create a dof for each own node added
        Arcane::Int32ArrayView target_family_added_items_lids;
        Arcane::Int32ArrayView target_family_removed_items_lids;
        m_connectivity_mng.getTargetFamilyModifiedItems(&connectivity, target_family_added_items_lids,
                                                       target_family_removed_items_lids);
        auto item_family = connectivity.targetFamily() ;
        Arcane::ItemVector target_family_added_items_own(item_family);
        {
          Integer i=0 ;
          ENUMERATE_ITEM(iitem,item_family->view(target_family_added_items_lids))
          {
             if (target_family_added_items_lids[i++]!= -1 && iitem->isOwn())
               target_family_added_items_own.add(iitem->localId());
          }
        }
        Arcane::Int32ConstArrayView target_family_added_items_own_lids =
          target_family_added_items_own.viewAsArray();

        // Create new dofs on these new nodes : on the owned node only
        Arcane::Int64UniqueArray added_uids;
        added_uids.reserve(target_family_added_items_own.size());
        ENUMERATE_ITEM(iitem,target_family_added_items_own)
        {
          added_uids.add(_doFUid(m_dualnode_kinds[index],*iitem));
        }
        Arcane::Int32SharedArray added_lids(added_uids.size());
        m_dual_node_family.addDoFs(added_uids, added_lids);

        // Update connectivity
        connectivity.updateConnectivity(added_lids,target_family_added_items_own_lids);

        Arcane::ItemVector target_family_removed_items_own(item_family);
        ENUMERATE_ITEM(iitem,item_family->view(target_family_removed_items_lids))
        {
           if (iitem->isOwn()) target_family_removed_items_own.add(iitem->localId());
        }
        Arcane::Int32ConstArrayView target_family_removed_items_own_lids =
          target_family_removed_items_own.viewAsArray();
        // Create new dofs on these new nodes : on the owned node only
        Arcane::Int64UniqueArray removed_uids;
        removed_uids.reserve(target_family_removed_items_own.size());
        ENUMERATE_ITEM(iitem,target_family_removed_items_own)
        {
          removed_uids.add(_doFUid(m_dualnode_kinds[index],*iitem));
        }
        Arcane::Int32UniqueArray remoded_lids(removed_uids.size());
        m_dual_node_family.itemsUniqueIdToLocalId(remoded_lids,removed_uids,true) ;
        m_dual_node_family.removeDoFs(remoded_lids);

        m_dual_node_family.endUpdate();

        // Update ghost
        //m_connectivity_mng.getSynchronizer(&connectivity)->synchronize();

        // Finalize connectivity update
        m_connectivity_mng.setUpToDate(&connectivity);

        ++index ;
      }
  }

  {
      if (m_links_connectivity.sourceFamily() && !m_connectivity_mng.isUpToDate(&m_links_connectivity))
      {
        // Handle added nodes : create a dof for each own node added
        Arcane::Int32ArrayView source_family_added_items_lids;
        Arcane::Int32ArrayView source_family_removed_items_lids;
        m_connectivity_mng.getSourceFamilyModifiedItems(&m_links_connectivity, source_family_added_items_lids,
                                                       source_family_removed_items_lids);
        auto item_family = m_links_connectivity.sourceFamily() ;
        Arcane::ItemVector source_family_added_items_own(item_family);
        ENUMERATE_ITEM(iitem,item_family->view(source_family_added_items_lids))
        {
           if (iitem->isOwn()) source_family_added_items_own.add(iitem->localId());
        }
        Arcane::Int32ConstArrayView source_family_added_items_own_lids =
          source_family_added_items_own.viewAsArray();
        // Create new dofs on these new nodes : on the owned node only
        Arcane::Int64UniqueArray uids(source_family_added_items_own.size());
        Arcane::Integer i = 0;
        ENUMERATE_ITEM(iitem,source_family_added_items_own)
        {
          uids[i++] = Arcane::mesh::DoFUids::uid(iitem->uniqueId().asInt64());
        }
        Arcane::Int32SharedArray lids(uids.size());
        m_link_family.addDoFs(uids, lids);
        m_link_family.endUpdate();
        // Update connectivity
        m_links_connectivity.updateConnectivity(source_family_added_items_own_lids, lids);

        Arcane::Int32SharedArray null_item_lids(source_family_removed_items_lids.size(),
            Arcane::NULL_ITEM_LOCAL_ID);
        m_links_connectivity.updateConnectivity(source_family_removed_items_lids, null_item_lids);

        // Update ghost
        //m_connectivity_mng.getSynchronizer(&m_links_connectivity)->synchronize();

        // Finalize connectivity update
        m_connectivity_mng.setUpToDate(&m_links_connectivity);
      }
  }
#endif

#ifdef GRAPH_USE_INCREMENTAL_CONNECTIVITY
  auto& dual_node_family = m_dof_mng.family(GraphDofs::dualNodeFamilyName());
  m_dual_node_to_connectivity_index.resize(&dual_node_family, -1);
  ENUMERATE_DOF(idof,dual_node_family.allItems())
  {
    debug()<<" DUALNODE LID : "<<idof->localId();
    debug()<<"          UID : "<<idof->uniqueId();
    for(Integer index =0;index<m_incremental_connectivities.size();++index)
    {
      auto connectivity = m_incremental_connectivities[index] ;
      if(connectivity)
      {
        debug()<<"UPDATE CONNECTIVITY : "<<connectivity->name();
        ConnectivityItemVector accessor(connectivity) ;
        if(accessor.connectedItems(*idof).size()>0)
        {
          m_dual_node_to_connectivity_index[*idof] = index ;
          debug()<<"        INDEX : "<<index<<" "<<m_dualnode_kinds[index] ;
        }
      }
    }
  }

  {

    ConnectivityItemVector accessor(m_links_incremental_connectivity) ;
    ENUMERATE_DOF(ilink,m_link_family.allItems())
    {
      debug()<<"LINK : "<<ilink->uniqueId();
      debug()<<"        NB DUALNODES : "<<accessor.connectedItems(*ilink).size() ;
      auto dof_view = accessor.connectedItems(*ilink) ;
      for(Integer i=0;i<dof_view.size();++i)
      {
        debug()<<"                         DOF["<<i<<"] LID : "<<dof_view[i].localId();
        debug()<<"                                      UID : "<<dof_view[i].uniqueId();
        auto index = m_dual_node_to_connectivity_index[dof_view[i]] ;
        debug()<<"                                    INDEX : "<<index<<" "<<m_dualnode_kinds[index] ;
      }
    }
  }

  m_graph_connectivity.reset( new GraphIncrementalConnectivity(m_links_incremental_connectivity,
                                                               m_incremental_connectivities,
                                                               m_dual_node_to_connectivity_index)) ;
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void GraphDofs::
printDualNodes() const
{
#ifdef GRAPH_USE_INCREMENTAL_CONNECTIVITY
    ENUMERATE_DOF(idualnode,dualNodeFamily()->allItems())
    {
      debug() << "DualNode : lid = " << idualnode->localId();
      debug() << "           uid = " << idualnode->uniqueId();
        auto dual_item = m_graph_connectivity->dualItem(*idualnode);
        debug() << "           DualItem : lid = "<<dual_item.localId();
        debug() << "                      uid = "<<dual_item.uniqueId();
    }
#endif

}
void GraphDofs::
printLinks() const
{
#ifdef GRAPH_USE_LEGACY_CONNECTIVITY
    ConnectivityItemVector dual_nodes(&m_links_connectivity);
    ENUMERATE_DOF(ilink,linkFamily()->allItems())
    {
        info() << "Link = "  <<  ilink.localId();
        ENUMERATE_DOF(idual_node,dual_nodes.connectedItems(ilink) ) {
          info() << "     DoF : index = "<< idual_node.index();
          info() << "           lid   = " <<idual_node->localId();
          info() << "           uid   = " <<idual_node->uniqueId();
          auto dual_item = m_graph_connectivity.dualItem(*idual_node);
          info() << "           DualItem : lid = "<<dual_item.localId();
          info() << "                      uid = "<<dual_item.uniqueId();
        }
    }
#endif

#ifdef GRAPH_USE_INCREMENTAL_CONNECTIVITY
    ConnectivityItemVector dual_nodes(m_links_incremental_connectivity);
    ENUMERATE_DOF(ilink,linkFamily()->allItems())
    {
        debug() << "Link       : LID   = "  <<  ilink.localId() << "UID = "<<ilink->uniqueId();
        ENUMERATE_DOF(idual_node,dual_nodes.connectedItems(ilink) ) {
          debug() << "     Dof : index = "<< idual_node.index();
          debug() << "     Dof : lid   = "<< idual_node->localId();
          debug() << "           uid   = " <<idual_node->uniqueId();
          //info() << "           dual uid = " << dualItem(*idual_node).uniqueId();
        }
    }
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_MESH_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

