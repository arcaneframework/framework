// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SubMeshTools.cc                                             (C) 2000-2024 */
/*                                                                           */
/* Algorithmes spécifiques aux sous-maillages.                               */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/mesh/SubMeshTools.h"

#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/ScopedPtr.h"

// Includes génériques non spécifiques à l'implémentation DynamicMesh
#include "arcane/core/Variable.h"
#include "arcane/core/SharedVariable.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/MeshToMeshTransposer.h"
#include "arcane/core/IVariableSynchronizer.h"
#include "arcane/core/IParallelExchanger.h"
#include "arcane/core/ISerializer.h"
#include "arcane/core/ISerializeMessage.h"
#include "arcane/core/ItemPrinter.h"
#include "arcane/core/TemporaryVariableBuildInfo.h"
#include "arcane/core/ParallelMngUtils.h"

// Includes spécifiques à l'implémentation à base de DynamicMesh
#include "arcane/mesh/DynamicMesh.h"
#include "arcane/mesh/DynamicMeshIncrementalBuilder.h"
#include "arcane/mesh/ItemFamily.h"
#include "arcane/mesh/CellFamily.h"
#include "arcane/mesh/FaceFamily.h"
#include "arcane/mesh/EdgeFamily.h"
#include "arcane/mesh/NodeFamily.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

SubMeshTools::
SubMeshTools(DynamicMesh * mesh, DynamicMeshIncrementalBuilder * mesh_builder)
: TraceAccessor(mesh->traceMng())
, m_mesh(mesh)
, m_mesh_builder(mesh_builder)
, m_parallel_mng(mesh->parallelMng())
{
  ;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

SubMeshTools::
~SubMeshTools()
{
  ;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SubMeshTools::
_updateGroups()
{
  // Réajuste les groupes en supprimant les entités qui ne sont plus dans le maillage
  for( IItemFamilyCollection::Enumerator i_family(m_mesh->itemFamilies()); ++i_family; ){
    IItemFamily* family = *i_family;
    for( ItemGroupCollection::Enumerator i_group((*i_family)->groups()); ++i_group; ){
      ItemGroup group = *i_group;
      // GG: la méthode suivante est équivalente à ce qui est dans le define OLD.
      family->partialEndUpdateGroup(group);
#ifdef OLD
      // GG: il ne faut pas modifier le groupe de toutes les entités car
      // il sera modifié dans DynamicMeshKindInfos::finalizeMeshChanged()
      // et donc si on le modifie ici, il le sera 2 fois.
      if (group.isAllItems())
        continue;
      // Anciennement: if (group.isLocalToSubDomain() || group.isOwn())
      if (group.internal()->hasComputeFunctor())
        group.invalidate();
      else
        group.internal()->removeSuppressedItems();
#endif
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
//! Remplit \a items_Local_id avec les entités fantomes de la famille \a family
void SubMeshTools::
_fillGhostItems(ItemFamily* family, Array<Int32>& items_local_id)
{
  items_local_id.clear();
  ItemInternalMap& items_map = family->itemsMap();
  Int32 rank = m_parallel_mng->commRank();

  items_map.eachItem([&](Item item) {
    if (item.owner() != rank) {
      items_local_id.add(item.localId());
    }
  });
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

SharedArray<ItemInternal*> SubMeshTools::
_floatingItems(ItemFamily * family)
{
  SharedArray<ItemInternal*> items_to_remove;
  items_to_remove.reserve(1000);
  DynamicMeshKindInfos::ItemInternalMap& items_map = family->itemsMap();
  items_map.eachItem([&](impl::ItemBase item) {
    if (!item.isSuppressed() && item.nbCell() == 0 && !item.isOwn()) {
      debug(Trace::High) << "Floating item to remove " << ItemPrinter(Item(item));
      items_to_remove.add(item.itemInternal());
    }
  });
  return items_to_remove;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SubMeshTools::
_removeCell(Cell cell)
{
  CellFamily& cell_family = m_mesh->trueCellFamily();
  FaceFamily& face_family = m_mesh->trueFaceFamily();
  EdgeFamily& edge_family = m_mesh->trueEdgeFamily();
  NodeFamily& node_family = m_mesh->trueNodeFamily();

  ItemLocalId cell_lid(cell);
  for( Face face : cell.faces() )
    face_family.removeCellFromFace(face,cell_lid);
  for( Edge edge : cell.edges() )
    edge_family.removeCellFromEdge(edge,cell_lid);
  for( Node node : cell.nodes() )
    node_family.removeCellFromNode(node,cell_lid);
  cell_family.removeItem(cell);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SubMeshTools::
removeDeadGhostCells()
{
  // TODO
  CellFamily& cell_family = m_mesh->trueCellFamily();
  FaceFamily& face_family = m_mesh->trueFaceFamily();
  EdgeFamily& edge_family = m_mesh->trueEdgeFamily();
  NodeFamily& node_family = m_mesh->trueNodeFamily();

  // On cherche les items fantômes dont les parents sont supprimés
  UniqueArray<Int32> items_to_remove;
  _fillGhostItems(&cell_family, items_to_remove);
  ENUMERATE_ (Cell, icell, cell_family.view(items_to_remove)) {
    impl::ItemBase item(icell->itemBase());
    ARCANE_ASSERT((!item.parentBase(0).isSuppressed()),("SubMesh cell not synchronized with its support group"));

    // on met no_destroy=true => ce sont les sous-items qui s'occuperont de la destruction
    if (item.parentBase(0).isSuppressed())
      _removeCell(item);
  }

  _fillGhostItems(&face_family, items_to_remove);
  ENUMERATE_ (Face, iface, face_family.view(items_to_remove)) {
    impl::ItemBase item(iface->itemBase());
    if (item.parentBase(0).isSuppressed()) {
      ARCANE_ASSERT((item.nbCell() == 0),("Cannot remove connected sub-item"));
      face_family.removeFaceIfNotConnected(item);
    }
  }

  _fillGhostItems(&edge_family, items_to_remove);
  EdgeInfoListView edges(&edge_family);
  ENUMERATE_ (Edge, iedge, edge_family.view(items_to_remove)) {
    impl::ItemBase item(iedge->itemBase());
    if (item.parentBase(0).isSuppressed()) {
      ARCANE_ASSERT((item.nbCell() == 0 && item.nbFace() == 0),("Cannot remove connected sub-item"));
      edge_family.removeEdgeIfNotConnected(item);
    }
  }

  _fillGhostItems(&node_family, items_to_remove);
  NodeInfoListView nodes(&node_family);
  ENUMERATE_ (Node, inode, node_family.view(items_to_remove)) {
    impl::ItemBase item(inode->itemBase());
    if (item.parentBase(0).isSuppressed()) {
      ARCANE_ASSERT((item.nbCell()==0 && item.nbFace()==0 && item.nbEdge()==0),("Cannot remove connected sub-item"));
      node_family.removeItem(item);
    }
  }

  _updateGroups();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SubMeshTools::
removeGhostMesh()
{
  CellFamily& cell_family = m_mesh->trueCellFamily();
  FaceFamily& face_family = m_mesh->trueFaceFamily();
  EdgeFamily& edge_family = m_mesh->trueEdgeFamily();
  NodeFamily& node_family = m_mesh->trueNodeFamily();

  // NOTE GG: normalement on devrait pouvoir remplacer tout le
  // code de destruction par:
  // for( ItemInternal* item : items_to_remove ){
  //   cell_family.removeCell(item);
  // }
  // A priori cela marche sur les tests comme je n'ai pas tous les tests IFPEN
  // je préfère laisser comme cela.

  
  // L'ordre est important pour correctement déconnecter les connectivités
  // Les méthodes sont ici écrites directement avec les primitives des *Family 
  // car les méthodes toutes prêtes (dont CellFamily::removeCell)
  // ne tiennent pas compte du mécanisme fantome particulier des sous-maillages
  // où un sous-item peuvent vivre sans sur-item rattaché à celui-ic
  // (ex: Face 'own' sans Cell autour)
  UniqueArray<Int32> items_to_remove;

  _fillGhostItems(&cell_family, items_to_remove);
  ENUMERATE_ (Cell, icell, cell_family.view(items_to_remove)) {
    _removeCell(*icell);
  }

  _fillGhostItems(&face_family, items_to_remove);
  ENUMERATE_ (Face, iface, face_family.view(items_to_remove)) {
    face_family.removeFaceIfNotConnected(*iface);
  }

  _fillGhostItems(&edge_family, items_to_remove);
  ENUMERATE_ (Edge, iedge, edge_family.view(items_to_remove)) {
    Edge edge(*iedge);
    if (edge.nbCell() == 0 && edge.nbFace() == 0)
      edge_family.removeEdgeIfNotConnected(edge);
  }

  _fillGhostItems(&node_family, items_to_remove);
  ENUMERATE_ (Node, inode, node_family.view(items_to_remove)) {
    Node node(*inode);
    if (node.nbCell() == 0 && node.nbFace() == 0 && node.nbEdge() == 0)
      node_family.removeItem(node);
  }

  _updateGroups();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SubMeshTools::
removeFloatingItems()
{
  FaceFamily& face_family = m_mesh->trueFaceFamily();
  EdgeFamily& edge_family = m_mesh->trueEdgeFamily();
  NodeFamily& node_family = m_mesh->trueNodeFamily();

  // L'ordre est important pour correctement déconnecter les connectivités
  // Les méthodes sont ici écrites directement avec les primitives des *Family 
  // car les méthodes toutes prêtes (dont CellFamily::removeCell)
  // ne tiennent pas compte du mécanisme fantome particulier des sous-maillages
  // où un sous-item peuvent vivre sans sur-item rattaché à celui-ic
  // (ex: Face 'own' sans Cell autour)
  SharedArray<ItemInternal*> items_to_remove;
  items_to_remove = _floatingItems(&face_family);
  for( ItemInternal* item : items_to_remove ){
    if (item->nbCell()==0)
      face_family.removeFaceIfNotConnected(item);
  }
  items_to_remove = _floatingItems(&edge_family);
  for( Integer i=0, is=items_to_remove.size(); i<is; ++i ){
    ItemInternal * item = items_to_remove[i];
    if (item->nbCell()==0 && item->nbFace()==0)
      edge_family.removeEdgeIfNotConnected(item);
  }
  items_to_remove = _floatingItems(&node_family);
  for( Integer i=0, is=items_to_remove.size(); i<is; ++i ){
    ItemInternal * item = items_to_remove[i];
    if (item->nbCell()==0 && item->nbFace()==0 && item->nbEdge()==0)
      node_family.removeItem(item);
  }

  _updateGroups();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SubMeshTools::
updateGhostMesh()
{
  removeGhostMesh();

  _checkValidItemOwner();

  eItemKind kinds[] = { IK_Cell, IK_Face, IK_Edge, IK_Node  };
  Integer nb_kind = sizeof(kinds)/sizeof(eItemKind);
  for(Integer i_kind=0;i_kind<nb_kind;++i_kind){
    IItemFamily * family = m_mesh->itemFamily(kinds[i_kind]);
    updateGhostFamily(family);
  }

  removeFloatingItems();
  _checkFloatingItems();
  _checkValidItemOwner();

  // La finalisation des familles et la construction 
  // de leurs synchronizers est délégué à l'extérieur
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SubMeshTools::
updateGhostFamily(IItemFamily * family)
{
  const eItemKind kind = family->itemKind();
  // IMesh * parent_mesh = m_mesh->parentMesh();
  IItemFamily * parent_family = family->parentFamily();

  debug(Trace::High) << "Process ghost on submesh " << m_mesh->name() << " with kind=" << kind;

  auto exchanger { ParallelMngUtils::createExchangerRef(m_parallel_mng) };
  IVariableSynchronizer * synchronizer = parent_family->allItemsSynchronizer();
  Int32ConstArrayView ranks = synchronizer->communicatingRanks();
  std::map<Integer, SharedArray<Int64> > to_send_items;
  for(Integer i=0;i<ranks.size();++i){
    const Integer rank = ranks[i];
    debug(Trace::High) << "Has " << kind << " comm with " << rank << " : " << i << " / " << ranks.size() << " ranks";

    // Les shared sont forcément own => consistence des requêtes
    ItemVectorView shared_items(parent_family->view(synchronizer->sharedItems(i)));
    ItemVector shared_submesh_items = MeshToMeshTransposer::transpose(parent_family, family, shared_items);
    SharedArray<Int64> current_to_send_items;

    ENUMERATE_ITEM(iitem, shared_submesh_items){
      if (iitem.localId() != NULL_ITEM_LOCAL_ID){
        const Item & item = *iitem;
        ARCANE_ASSERT((item.uniqueId() == item.parent().uniqueId()),("Inconsistent item/parent uid"));
        debug(Trace::Highest) << "Send shared submesh item to " << rank << " " << ItemPrinter(item);
        current_to_send_items.add(item.parent().uniqueId());
      }
    }

    debug(Trace::High) << "SubMesh ghost comm " << kind << " with " << rank << " : "
                       << shared_items.size() << " / " << current_to_send_items.size();

    // Pour les cas de sous-maillages localisés, on ne considère réellement que les 
    // destinataires où il y a quelque chose à envoyer
    if (!current_to_send_items.empty()){
      exchanger->addSender(rank);
      to_send_items[rank] = current_to_send_items;
    }
  }

  exchanger->initializeCommunicationsMessages();
      
  for(Integer i=0, ns=exchanger->nbSender(); i<ns; ++i){
    ISerializeMessage* sm = exchanger->messageToSend(i);
    const Int32 rank = sm->destination().value();
    ISerializer* s = sm->serializer();
    const Int64Array & current_to_send_items = to_send_items[rank];
    s->setMode(ISerializer::ModeReserve);

    s->reserveArray(current_to_send_items);

    s->allocateBuffer();
    s->setMode(ISerializer::ModePut);

    s->putArray(current_to_send_items);
  }
      
  to_send_items.clear(); // destruction des données temporaires avant envoie
  exchanger->processExchange();

  for( Integer i=0, ns=exchanger->nbReceiver(); i<ns; ++i ){
    ISerializeMessage* sm = exchanger->messageToReceive(i);
    const Int32 rank = sm->destination().value();
    ISerializer* s = sm->serializer();
    //Integer new_submesh_ghost_size = s->getInteger();
    Int64UniqueArray uids;
    s->getArray(uids);
    Int32UniqueArray lids(uids.size());
    parent_family->itemsUniqueIdToLocalId(lids, uids);
    ItemVectorView new_submesh_ghosts = parent_family->view(lids);
    ENUMERATE_ITEM(iitem, new_submesh_ghosts){
      if (iitem->owner() != rank)
        fatal() << "Bad ghost owner " << ItemPrinter(*iitem) << " : expected owner=" << rank;
      debug(Trace::Highest) << "Add ghost submesh item from " << rank << " : " << FullItemPrinter(*iitem);
    }
    m_mesh_builder->addParentItems(new_submesh_ghosts, kind);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SubMeshTools::
display(IMesh* mesh, const String msg)
{
  ITraceMng * traceMng = mesh->traceMng();
  traceMng->info() << "Display mesh " << mesh->name() << " : " << msg;
  eItemKind kinds[] = { IK_Cell, IK_Face, IK_Edge, IK_Node  };
  Integer nb_kind = sizeof(kinds)/sizeof(eItemKind);
  for(Integer i_kind=0;i_kind<nb_kind;++i_kind){      
    IItemFamily* family = mesh->itemFamily(kinds[i_kind]);
    ItemInfoListView items(family);
    Integer count = 0;
    for( Integer z=0, zs=family->maxLocalId(); z<zs; ++z )
      if (!items[z].itemBase().isSuppressed())
        ++count;
    traceMng->info() << "\t" << family->itemKind() << " " << count;

    for( Integer z=0, zs=family->maxLocalId(); z<zs; ++z ){
      Item item = items[z];
      if (!item.itemBase().isSuppressed()){
        traceMng->info() << ItemPrinter(item);
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SubMeshTools::
_checkValidItemOwner()
{
  if (!arcaneIsCheck())
    return;

  eItemKind kinds[] = { IK_Cell, IK_Face, IK_Edge, IK_Node  };
  Integer nb_kind = sizeof(kinds)/sizeof(eItemKind);
  for(Integer i_kind=0;i_kind<nb_kind;++i_kind){
    IItemFamily * family = m_mesh->itemFamily(kinds[i_kind]);
    IItemFamily * parent_family = family->parentFamily();
    ItemInfoListView items(family);
    for( Integer z=0, zs=family->maxLocalId(); z<zs; ++z ){
      Item item = items[z];
      if (!item.itemBase().isSuppressed()){
        if (item.uniqueId() != item.itemBase().parentBase(0).uniqueId()){
          Int64UniqueArray uids; uids.add(item.uniqueId());
          Int32UniqueArray lids(1);
          parent_family->itemsUniqueIdToLocalId(lids,uids,false);
          ARCANE_FATAL("Inconsistent parent uid '{0}' now located in '{1}'",
                       ItemPrinter(item),lids[0]);
        }
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SubMeshTools::
_checkFloatingItems()
{
  Integer nerror = 0;
  eItemKind kinds[] = { IK_Face, IK_Edge, IK_Node  };
  Integer nb_kind = sizeof(kinds)/sizeof(eItemKind);
  for(Integer i_kind=0;i_kind<nb_kind;++i_kind){
    IItemFamily * family = m_mesh->itemFamily(kinds[i_kind]);
    // Calcul des items orphelins de cellules
    ItemInfoListView items(family);
    for( Integer z=0, zs=family->maxLocalId(); z<zs; ++z ){
      Item item = items[z];
      if (!item.itemBase().isSuppressed() && item.itemBase().nbCell() == 0) {
        error() << "Floating item detected : " << ItemPrinter(item);
        ++nerror;
      }
    }
  }
  if (nerror)
    fatal() << "ERROR " << String::plural(nerror,"floating item") << " detected; see above";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
