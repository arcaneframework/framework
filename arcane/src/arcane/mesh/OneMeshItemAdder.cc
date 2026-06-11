// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* OneMeshItemAdder.cc                                         (C) 2000-2025 */
/*                                                                           */
/* Adding entities one by one.                                               */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/mesh/OneMeshItemAdder.h"

#include "arcane/utils/NotSupportedException.h"
#include "arcane/utils/ValueConvert.h"
#include "arcane/utils/FixedArray.h"

#include "arcane/core/MeshUtils.h"
#include "arcane/core/MeshToMeshTransposer.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/ItemPrinter.h"

#include "arcane/mesh/DynamicMesh.h"
#include "arcane/mesh/DynamicMeshIncrementalBuilder.h"
#include "arcane/mesh/ItemTools.h"
#include "arcane/mesh/ConnectivityNewWithDependenciesTypes.h"
#include "arcane/mesh/GraphDoFs.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class OneMeshItemAdder::CellInfoProxy
{
 public:

  CellInfoProxy(ItemTypeInfo* type_info,
                Int64 cell_uid,
                Int32 sub_domain_id,
                Int64ConstArrayView info,
                bool allow_build_face = false)
  : m_type_info(type_info)
  , m_cell_uid(cell_uid)
  , m_info(info)
  , m_owner(sub_domain_id)
  , m_allow_build_face(allow_build_face)
  {}

  Int64 uniqueId() const { return m_cell_uid; }
  ItemTypeInfo* typeInfo() const { return m_type_info; }
  Int32 owner() const { return m_owner; }
  Integer nbNode() const { return m_info.size(); }
  Integer nbFace() const { return m_type_info->nbLocalFace(); }
  Integer nbEdge() const { return m_type_info->nbLocalEdge(); }
  Int64 nodeUniqueId(Integer i_node) const { return m_info[i_node]; }
  Int32 nodeOwner(Integer) const { return m_owner; }
  Int32 faceOwner(Integer) const { return m_owner; }
  Int32 edgeOwner(Integer) const { return m_owner; }
  ItemTypeInfo::LocalFace localFace(Integer i_face) const { return m_type_info->localFace(i_face); }
  bool allowBuildFace() const { return m_allow_build_face; }
  bool allowBuildEdge() const { return m_allow_build_face; }

 private:

  ItemTypeInfo* m_type_info = nullptr;
  Int64 m_cell_uid = NULL_ITEM_UNIQUE_ID;
  Int64ConstArrayView m_info;
  Int32 m_owner = A_NULL_RANK;
  bool m_allow_build_face = false;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

OneMeshItemAdder::
OneMeshItemAdder(DynamicMeshIncrementalBuilder* mesh_builder)
: TraceAccessor(mesh_builder->mesh()->traceMng())
, m_mesh(mesh_builder->mesh())
, m_mesh_builder(mesh_builder)
, m_cell_family(m_mesh->trueCellFamily())
, m_node_family(m_mesh->trueNodeFamily())
, m_face_family(m_mesh->trueFaceFamily())
, m_edge_family(m_mesh->trueEdgeFamily())
, m_item_type_mng(m_mesh->itemTypeMng())
, m_mesh_info(m_mesh->meshPartInfo().partRank())
, m_face_reorderer(m_item_type_mng)
{
  //m_work_face_sorted_nodes.reserve(100);
  m_work_face_orig_nodes_uid.reserve(100);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemInternal* OneMeshItemAdder::
addOneNode(Int64 node_uid, Int32 owner)
{
  bool is_add = false;
  ItemInternal* node = m_node_family.findOrAllocOne(node_uid, is_add);
  if (is_add) {
    ++m_mesh_info.nbNode();
    node->setOwner(owner, owner);
  }
  return node;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Checks the coherence of nodes for an already added item.
 *
 * When attempting to add an item that already exists,
 * checks that the nodes provided for the addition are the same as those of
 * the already existing item.
 */
void OneMeshItemAdder::
_checkSameItemCoherency(ItemWithNodes item, ConstArrayView<Int64> nodes_uid)
{
  Int32 nb_node = nodes_uid.size();
  // Checks that the number of nodes is the same
  if (item.nbNode() != nb_node)
    ARCANE_FATAL("Trying to add existing item (kind='{0}', uid={1}) with different number of node (existing={2} new={3})",
                 item.kind(), item.uniqueId(), item.nbNode(), nb_node);

  // Checks that the nodes correspond to the existing ones
  for (Int32 i = 0; i < nb_node; ++i) {
    Int64 new_uid = nodes_uid[i];
    Int64 current_uid = item.node(i).uniqueId();
    if (new_uid != current_uid) {
      std::ostringstream ostr;
      for (Int32 k = 0; k < nb_node; ++k)
        ostr << " " << item.node(k).uniqueId();
      ARCANE_FATAL("Trying to add existing item (kind='{0}', uid={1}) with different nodes (index={2} existing='{3}' new='{4}')",
                   item.kind(), item.uniqueId(), i, ostr.str(), nodes_uid);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Generates a uniqueId() for the face if \a uid is null.
 *
 * If \a uid equals NULL_ITEM_UNIQUE_ID, a uniqueId() is generated for the face.
 */
Int64 OneMeshItemAdder::
_checkGenerateFaceUniqueId(Int64 uid, ConstArrayView<Int64> nodes_uid)
{
  if (uid != NULL_ITEM_UNIQUE_ID)
    return uid;
  if (m_use_hash_for_edge_and_face_unique_id)
    uid = MeshUtils::generateHashUniqueId(nodes_uid);
  else
    uid = m_next_face_uid++;
  return uid;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Adds a face.
 *
 * This method is called when one wishes to directly create a face
 * that may not necessarily be connected to a cell. Generally, faces
 * are created automatically when meshes are added.
 *
 * Adds a face by providing the face's unique_id and the unique_ids
 * of the nodes to connect.
 *
 * If \a face_uid is equal to NULL_ITEM_UNIQUE_ID, the identifier is generated.
 */
ItemInternal* OneMeshItemAdder::
addOneFace(ItemTypeId type_id, Int64 face_uid, Int32 owner_rank, Int64ConstArrayView nodes_uid)
{
  const Integer face_nb_node = nodes_uid.size();

  //m_work_face_sorted_nodes.resize(face_nb_node);
  //m_work_face_orig_nodes_uid.resize(face_nb_node);
  //for( Integer z=0; z<face_nb_node; ++z )
  //m_work_face_orig_nodes_uid[z] = nodes_uid[z];
  m_face_reorderer.reorder(type_id, nodes_uid);
  ConstArrayView<Int64> face_sorted_nodes = m_face_reorderer.sortedNodes();
  // TODO: in the case where the face will be orphaned (not connected to a cell),
  // check if the face needs reorientation because this risks introducing
  // an inconsistency if one later wishes to calculate a normal.
  //MeshUtils::reorderNodesOfFace(m_work_face_orig_nodes_uid,m_work_face_sorted_nodes);
  face_uid = _checkGenerateFaceUniqueId(face_uid, face_sorted_nodes);
  bool is_add_face = false;
  Face face = m_face_family.findOrAllocOne(face_uid, type_id, is_add_face);
  // The face does not exist
  if (is_add_face) {
    ++m_mesh_info.nbFace();
    face.mutableItemBase().setOwner(owner_rank, m_mesh_info.rank());
    for (Integer i_node = 0; i_node < face_nb_node; ++i_node) {
      Node node = addOneNode(face_sorted_nodes[i_node], m_mesh_info.rank());
      m_face_family.replaceNode(face, i_node, node);
      m_node_family.addFaceToNode(node, face);
    }
  }
  else {
    if (arcaneIsCheck())
      _checkSameItemCoherency(face, face_sorted_nodes);
  }

  return ItemCompatibility::_itemInternal(face);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemInternal* OneMeshItemAdder::
addOneEdge(Int64 edge_uid, Int32 rank, Int64ConstArrayView nodes_uid)
{
  m_work_edge_sorted_nodes.resize(2);
  m_work_edge_orig_nodes_uid.resize(2);

  for (Integer z = 0; z < 2; ++z)
    m_work_edge_orig_nodes_uid[z] = nodes_uid[z];
  // reorderNodesOfFace behaves correctly here for edges == faces in 2D
  MeshUtils::reorderNodesOfFace(m_work_edge_orig_nodes_uid, m_work_edge_sorted_nodes);

  bool is_add_edge = false;
  ItemInternal* edge = m_edge_family.findOrAllocOne(edge_uid, is_add_edge);

  // The edge does not exist
  if (is_add_edge) {
    ++m_mesh_info.nbEdge();
    edge->setOwner(rank, m_mesh_info.rank());
    for (Integer i_node = 0; i_node < 2; ++i_node) {
      ItemInternal* current_node_internal = addOneNode(m_work_edge_sorted_nodes[i_node], m_mesh_info.rank());
      m_edge_family.replaceNode(ItemLocalId(edge), i_node, ItemLocalId(current_node_internal));
      m_node_family.addEdgeToNode(current_node_internal, edge);
    }
  }
  return edge;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <>
Face OneMeshItemAdder::
_findInternalFace(Integer i_face, const FullCellInfo& cell_info, bool& is_add)
{
  const Int64 face_unique_id = cell_info.faceUniqueId(i_face);
  ItemTypeInfo* cell_type_info = cell_info.typeInfo();
  const ItemTypeInfo::LocalFace& lf = cell_type_info->localFace(i_face);
  ItemTypeInfo* face_type_info = m_item_type_mng->typeFromId(lf.typeId());
  return m_face_family.findOrAllocOne(face_unique_id, face_type_info, is_add);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <>
Face OneMeshItemAdder::
_findInternalFace(Integer i_face, const CellInfoProxy& cell_info, bool& is_add)
{
  const ItemInternalMap& nodes_map = m_mesh->nodesMap();
  ItemTypeInfo* cell_type_info = cell_info.typeInfo();
  const ItemTypeInfo::LocalFace& lf = cell_type_info->localFace(i_face);
  ConstArrayView<Int64> face_sorted_nodes = m_face_reorderer.sortedNodes();
  Int64 face_unique_id = NULL_ITEM_UNIQUE_ID;
  Face face_internal;
  const bool use_hash = m_use_hash_for_edge_and_face_unique_id;
  // If calculating face uniqueIds from nodes,
  // we can directly know if the face already exists. Otherwise, we
  // call ItemTools::findFaceInNode2(), but this function is more
  // costly because it searches for the face by traversing the faces connected
  // to the first node of the face.
  if (use_hash) {
    face_unique_id = _checkGenerateFaceUniqueId(NULL_ITEM_UNIQUE_ID, face_sorted_nodes);
    face_internal = m_face_family.itemsMap().tryFind(face_unique_id);
  }
  else {
    // Retrieves the oriented nodes of the face
    // (One must have called one of the face addition methods before)
    Node nbi = nodes_map.findItem(face_sorted_nodes[0]);
    face_internal = ItemTools::findFaceInNode2(nbi, lf.typeId(), face_sorted_nodes);
  }
  if (face_internal.null()) {
    // The face was not found. It therefore does not exist in our sub-domain.
    // If this is allowed, the new face is created.
    if (!cell_info.allowBuildFace() && !m_use_hash_for_edge_and_face_unique_id) {
      info() << "BadCell uid=" << cell_info.uniqueId();
      for (Int32 i = 0; i < cell_info.nbNode(); ++i)
        info() << "Cell node I=" << i << " uid=" << cell_info.nodeUniqueId(i);
      ARCANE_FATAL("On the fly face allocation is not allowed here.\n"
                   " You need to add faces with IMeshModifier::addFaces().\n"
                   " CellUid={0} LocalFace={1} FaceNodes={2}",
                   cell_info.uniqueId(), i_face, face_sorted_nodes);
    }
    // Only recalculate the hash if it hasn't been done already.
    if (!use_hash)
      face_unique_id = _checkGenerateFaceUniqueId(NULL_ITEM_UNIQUE_ID, face_sorted_nodes);
    is_add = true;
    ItemTypeInfo* face_type = m_item_type_mng->typeFromId(lf.typeId());
    return m_face_family.allocOne(face_unique_id, face_type);
  }
  else {
    is_add = false;
    return face_internal;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <>
Edge OneMeshItemAdder::
_findInternalEdge(Integer i_edge, const FullCellInfo& cell_info,
                  Int64 first_node, Int64 second_node, bool& is_add)
{
  ARCANE_UNUSED(first_node);
  ARCANE_UNUSED(second_node);

  const Int64 edge_unique_id = cell_info.edgeUniqueId(i_edge);
  return m_edge_family.findOrAllocOne(edge_unique_id, is_add);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <>
Edge OneMeshItemAdder::
_findInternalEdge(Integer i_edge, const CellInfoProxy& cell_info, Int64 first_node, Int64 second_node, bool& is_add)
{
  ARCANE_UNUSED(i_edge);

  const bool use_hash = m_use_hash_for_edge_and_face_unique_id;
  FixedArray<Int64, 2> nodes;
  Int64 edge_unique_id = NULL_ITEM_UNIQUE_ID;
  Edge edge_internal;
  if (use_hash) {
    nodes[0] = first_node;
    nodes[1] = second_node;
    edge_unique_id = MeshUtils::generateHashUniqueId(nodes.view());
    edge_internal = m_edge_family.itemsMap().tryFind(edge_unique_id);
  }
  else {
    const ItemInternalMap& nodes_map = m_mesh->nodesMap();
    Node nbi = nodes_map.findItem(first_node);
    edge_internal = ItemTools::findEdgeInNode2(nbi, first_node, second_node);
  }
  if (edge_internal.null()) {
    if (!cell_info.allowBuildEdge() && !use_hash)
      ARCANE_FATAL("On the fly edge allocation is not allowed here."
                   " You need to add edges before with IMeshModifier::addEdges()");
    if (!use_hash)
      edge_unique_id = m_next_edge_uid++;
    is_add = true;
    return m_edge_family.allocOne(edge_unique_id);
  }
  else {
    is_add = false;
    return edge_internal;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Adds a cell.
 *
 * When adding a cell, the nodes and faces belonging to it are
 * automatically added to the mesh if they are not already present.
 *
 * \param type type of the cell
 * \param cell_uid unique ID of the cell. If a cell with this ID
 * exists already, it means the cell is already present. In this case,
 * this method performs no operation.
 * \param sub_domain_id ID of the sub-domain to which the cell belongs
 * \param nodes_uid list of unique IDs of the cell. The number
 * of elements in this array must correspond to the cell type.

 * \retval true if the cell is actually added
*/
ItemInternal* OneMeshItemAdder::
addOneCell(ItemTypeId type_id,
           Int64 cell_uid,
           Int32 sub_domain_id,
           Int64ConstArrayView nodes_uid,
           bool allow_build_face)
{
  CellInfoProxy cell_info_proxy(m_item_type_mng->typeFromId(type_id), cell_uid, sub_domain_id, nodes_uid, allow_build_face);

  return _addOneCell(cell_info_proxy);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Adds a cell.
 *
 * \retval true if the cell was actually added
*/
ItemInternal* OneMeshItemAdder::
addOneCell(const FullCellInfo& cell_info)
{
  return _addOneCell(cell_info);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ItemInternal* OneMeshItemAdder::
addOneItem(IItemFamily* family,
           IItemFamilyModifier* family_modifier,
           ItemTypeId type_id,
           Int64 item_uid,
           Integer item_owner,
           Integer sub_domain_id,
           Integer nb_connected_family,
           Int64ConstArrayView connectivity_info)
{
  ARCANE_ASSERT(m_mesh->itemFamilyNetwork(), ("ItemFamilyNetwork is required to call OneMeshItemAdder::addOneItem"));
  bool is_alloc = true;
  Item item = family_modifier->findOrAllocOne(item_uid, type_id, m_mesh_info, is_alloc); // don't forget to add print in the class method
  item.mutableItemBase().setOwner(item_owner, sub_domain_id);
  // Add connectivities if needed
  Integer info_index = 0;
  for (Integer family_index = 0; family_index < nb_connected_family; ++family_index) {
    // get connected family
    eItemKind family_kind = static_cast<eItemKind>(connectivity_info[info_index++]); // another way ?
    Int32 nb_connected_item = CheckedConvert::toInt32(connectivity_info[info_index++]);
    IItemFamily* connected_family = m_mesh->itemFamily(family_kind);
    // get connectivities family -> connected_family and reverse
    String connectivity_name = mesh::connectivityName(family, connected_family);
    String reverse_connectivity_name = mesh::connectivityName(connected_family, family);
    bool is_dependency = false;
    IIncrementalItemConnectivity* family_to_connected_family = m_mesh->itemFamilyNetwork()->getStoredConnectivity(family, connected_family, connectivity_name, is_dependency);
    IIncrementalItemConnectivity* connected_family_to_family = m_mesh->itemFamilyNetwork()->getStoredConnectivity(connected_family, family, reverse_connectivity_name);
    // Clear connectivities for already allocated items (except dependencies since replace is used)
    if (!is_alloc) {
      if (!is_dependency)
        _clearConnectivity(ItemLocalId(item), family_to_connected_family);
      if (connected_family_to_family)
        _clearReverseConnectivity(ItemLocalId(item), family_to_connected_family, connected_family_to_family);
    }
    // get connected item lids
    Int32UniqueArray connected_item_lids(nb_connected_item);
    connected_family->itemsUniqueIdToLocalId(connected_item_lids, connectivity_info.subView(info_index, nb_connected_item), true);
    for (Integer connected_item_index = 0; connected_item_index < nb_connected_item; ++connected_item_index) {
      if (family_to_connected_family) {
        // pre-alloc are done (=> use replace) when a dependency relation ("owning relation") while not when only a relation (use add)
        if (!is_dependency)
          family_to_connected_family->addConnectedItem(ItemLocalId(item), ItemLocalId(connected_item_lids[connected_item_index]));
        else
          family_to_connected_family->replaceConnectedItem(ItemLocalId(item), connected_item_index, ItemLocalId(connected_item_lids[connected_item_index])); // does not work with face to edges...
      }
      if (connected_family_to_family)
        connected_family_to_family->addConnectedItem(ItemLocalId(connected_item_lids[connected_item_index]), ItemLocalId(item));
    }
    info_index += nb_connected_item;
  }
  //  debug(Trace::Highest) << "[addItems] ADD_ITEM " << ItemPrinter(item) << " in " << family->name();
  debug(Trace::Highest) << "[addItems] ADD_ITEM " << ItemPrinter(item) << " in " << family->name();
  return ItemCompatibility::_itemInternal(item);
}

/*---------------------------------------------------------------------------*/

ItemInternal* OneMeshItemAdder::
addOneItem2(IItemFamily* family,
            IItemFamilyModifier* family_modifier,
            ItemTypeId type_id,
            Int64 item_uid,
            Integer item_owner,
            Integer sub_domain_id,
            Integer nb_connected_family,
            Int64ConstArrayView connectivity_info)
{
  ARCANE_ASSERT(m_mesh->itemFamilyNetwork(), ("ItemFamilyNetwork is required to call OneMeshItemAdder::addOneItem"));
  bool is_alloc = true;
  Item item = family_modifier->findOrAllocOne(item_uid, type_id, m_mesh_info, is_alloc); // don't forget to add print in the class method
  item.mutableItemBase().setOwner(item_owner, sub_domain_id);
  // Add connectivities if needed
  Integer info_index = 0;
  for (Integer family_index = 0; family_index < nb_connected_family; ++family_index) {
    // Prepare connection
    // get connected family
    eItemKind family_kind = static_cast<eItemKind>(connectivity_info[info_index++]); // another way ?
    Int32 nb_connected_item = CheckedConvert::toInt32(connectivity_info[info_index++]);
    if (nb_connected_item == 0)
      continue;
    IItemFamily* connected_family = nullptr;
    switch (family_kind) {
    case IK_Particle:
      connected_family = m_mesh->findItemFamily(family_kind, ParticleFamily::defaultFamilyName(), false, false);
      break;
    case IK_DoF:
      if (family->name() == GraphDoFs::dualNodeFamilyName())
        connected_family = m_mesh->findItemFamily(family_kind, GraphDoFs::linkFamilyName(), false, false);
      else
        connected_family = m_mesh->findItemFamily(family_kind, GraphDoFs::dualNodeFamilyName(), false, false);
      break;
    default:
      connected_family = m_mesh->itemFamily(family_kind);
      break;
    }
    // get connectivities family -> connected_family and reverse
    String connectivity_name = mesh::connectivityName(family, connected_family);
    bool is_dependency = false;
    IIncrementalItemConnectivity* family_to_connected_family = m_mesh->itemFamilyNetwork()->getConnectivity(family, connected_family, connectivity_name, is_dependency);
    if (!family_to_connected_family)
      ARCANE_FATAL("Cannot find connectivity name={0}", connectivity_name);
    bool is_deep_connectivity = m_mesh->itemFamilyNetwork()->isDeep(family_to_connected_family);
    bool is_relation = !(is_dependency && is_deep_connectivity);
    // Build connection
    // get connected item lids
    Int32UniqueArray connected_item_lids(nb_connected_item);
    bool do_fatal = is_relation ? false : true; // for relations, connected items may not be present and will be skipped.
    connected_family->itemsUniqueIdToLocalId(connected_item_lids, connectivity_info.subView(info_index, nb_connected_item), do_fatal);
    // if connection is relation, connected item not necessarily present: remove absent (ie null) items
    Integer nb_connected_item_found = nb_connected_item;
    if (is_relation) {
      for (Integer index = 0; index < connected_item_lids.size();) {
        if (connected_item_lids[index] == NULL_ITEM_LOCAL_ID) {
          connected_item_lids.remove(index);
          --nb_connected_item_found;
        }
        else
          ++index;
      }
    }
    for (Integer connected_item_index = 0; connected_item_index < nb_connected_item_found; ++connected_item_index) {
      if (family_to_connected_family) {
        // Only strategy : check and add
        auto connected_item_lid = ItemLocalId{ connected_item_lids[connected_item_index] };
        if (is_relation) {
          if (!family_to_connected_family->hasConnectedItem(ItemLocalId(item), connected_item_lid)) {
            family_to_connected_family->addConnectedItem(ItemLocalId(item), connected_item_lid);
          }
        }
        else {
          family_to_connected_family->replaceConnectedItem(ItemLocalId(item), connected_item_index, connected_item_lid);
        }
      }
    }
    info_index += nb_connected_item;
  }
  if (is_alloc)
    debug(Trace::Highest) << "[addItems] ADD_ITEM " << ItemPrinter(item) << " in " << family->name();
  //  debug(Trace::Highest) << "[addItems] DEPENDENCIES for " << family->name() << FullItemPrinter(item) ; // debug info
  //  _printRelations(item); // debug info
  return ItemCompatibility::_itemInternal(item);
}

/*---------------------------------------------------------------------------*/

void OneMeshItemAdder::
_printRelations(ItemInternal* item)
{
  debug(Trace::Highest) << "[addItems] RELATIONS for " << ItemPrinter(item) << " in " << item->family()->name();
  for (const auto& relation : m_mesh->itemFamilyNetwork()->getChildRelations(item->family())) {
    //    debug(Trace::Highest) << " Relation " << relation->name();
    //    debug(Trace::Highest) << " Relation " << relation->nbConnectedItem(ItemLocalId(item));
    ConnectivityItemVector connected_items(relation);
    for (const auto& connected_item : connected_items.connectedItems(ItemLocalId(item))) {
      debug(Trace::Highest) << ItemPrinter(connected_item);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void OneMeshItemAdder::
_clearConnectivity(ItemLocalId item, IIncrementalItemConnectivity* connectivity)
{

  ConnectivityItemVector accessor(connectivity);
  ENUMERATE_ITEM (connected_item, accessor.connectedItems(item)) {
    connectivity->removeConnectedItem(item, connected_item);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void OneMeshItemAdder::
_clearReverseConnectivity(ItemLocalId item, IIncrementalItemConnectivity* connectivity, IIncrementalItemConnectivity* reverse_connectivity)
{
  ConnectivityItemVector accessor(connectivity);
  ENUMERATE_ITEM (connected_item, accessor.connectedItems(item)) {
    reverse_connectivity->removeConnectedItem(connected_item, item);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <> void OneMeshItemAdder::
_AMR_Patch(Cell new_cell, const FullCellInfo& cell_info)
{
  //! AMR
  if (m_mesh->isAmrActivated()) {
    // FIXME doit-on traiter les mailles de niveau 0
    //comme celles de niveau superieur
    if (cell_info.level() != 0) {
      Integer child_rank = cell_info.whichChildAmI();
      Int64 hParent_uid = cell_info.hParentCellUniqueId();
      ItemTypeId cell_type = ItemTypeId::fromInteger(cell_info.typeId());
      bool is_add;
      Cell hParent_cell = m_cell_family.findOrAllocOne(hParent_uid, cell_type, is_add);
      m_cell_family._addParentCellToCell(new_cell, hParent_cell);
      m_cell_family._addChildCellToCell(hParent_cell, child_rank, new_cell);
    }
  }
}

template <> void OneMeshItemAdder::
_AMR_Patch(Cell cell, const CellInfoProxy& cell_info)
{
  ARCANE_UNUSED(cell);
  ARCANE_UNUSED(cell_info);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Adds a cell.
 *
 * \retval true if the cell was actually added
*/
template <typename CellInfo>
ItemInternal* OneMeshItemAdder::
_addOneCell(const CellInfo& cell_info)
{
  bool is_check = arcaneIsCheck();

  ItemTypeInfo* cell_type_info = cell_info.typeInfo();
  ItemTypeId cell_type_id = cell_type_info->itemTypeId();
  const bool is_verbose = false;
  if (is_verbose)
    info() << "AddNewCell type_id=" << cell_type_id << " nb_node=" << cell_info.nbNode()
           << " type=" << cell_type_info->typeName();
  // Checks if the cell already exists (in which case nothing is done).
  Cell inew_cell;
  {
    bool is_add; // ce flag est toujours correctement positionné via les findOrAllocOne
    inew_cell = m_cell_family.findOrAllocOne(cell_info.uniqueId(), cell_type_id, is_add);
    if (!is_add) {
      if (is_check) {
        Cell cell2(inew_cell);
        // Verifies that the nodes match those that exist.
        for (Integer i = 0, is = cell_info.nbNode(); i < is; ++i)
          if (cell_info.nodeUniqueId(i) != cell2.node(i).uniqueId())
            ARCANE_FATAL("trying to add existing cell (uid={0}) with different nodes",
                         cell_info.uniqueId());
      }
      return ItemCompatibility::_itemInternal(inew_cell);
    }
  }

  Cell new_cell(inew_cell);

  ++m_mesh_info.nbCell();

  const MeshKind& mesh_kind = m_mesh->meshKind();
  const bool allow_multi_dim_cell = !mesh_kind.isMonoDimension();
  const Int32 cell_nb_face = cell_info.nbFace();
  MutableItemBase mut_cell = new_cell.mutableItemBase();
  if (cell_type_info->dimension() == 2 && cell_type_info->nbLocalFace() == 0)
    mut_cell.addFlags(ItemFlags::II_HasEdgeFor1DItems);
  mut_cell.setOwner(cell_info.owner(), m_mesh_info.rank());
  // Vérifie la cohérence entre le type local et la maille créée.
  if (is_check) {
    if (cell_info.nbNode() != inew_cell.nbNode())
      ARCANE_FATAL("Incoherent number of nodes v={0} expected={1}", inew_cell.nbNode(), cell_info.nbNode());
    if (cell_nb_face != inew_cell.nbFace())
      ARCANE_FATAL("Incoherent number of faces v={0} expected={1}", inew_cell.nbFace(), cell_nb_face);
    if (!cell_type_info->isValidForCell())
      ARCANE_FATAL("Type '{0}' is not allowed for 'Cell' (cell_uid={1})",
                   cell_type_info->typeName(), cell_info.uniqueId());
    if (!allow_multi_dim_cell) {
      Int32 cell_dimension = cell_type_info->dimension();
      Int32 mesh_dimension = m_mesh->dimension();
      if (cell_dimension >= 0 && cell_dimension != mesh_dimension)
        ARCANE_FATAL("Incoherent dimension for cell uid={0} cell_dim={1} mesh_dim={2} type={3}",
                     cell_info.uniqueId(), cell_dimension, mesh_dimension, cell_type_info->typeName());
    }
  }

  //! Maps the uniqueId() table to ItemInternal*
  ItemInternalMap& nodes_map = m_node_family.itemsMap();

  _addNodesToCell(inew_cell, cell_info);

  if (m_mesh_builder->hasEdge()) {
    const Int32 cell_nb_edge = cell_info.nbEdge();
    // Adds the necessary edges.
    for (Integer i_edge = 0; i_edge < cell_nb_edge; ++i_edge) {
      const ItemTypeInfo::LocalEdge& le = cell_type_info->localEdge(i_edge);
      Int64 first_node = cell_info.nodeUniqueId(le.beginNode());
      Int64 second_node = cell_info.nodeUniqueId(le.endNode());
      if (first_node > second_node)
        std::swap(first_node, second_node);

      bool is_add = false;
      Edge edge_internal = _findInternalEdge(i_edge, cell_info, first_node, second_node, is_add);
      if (is_add) {
        if (is_verbose)
          info() << "Create edge " << edge_internal.uniqueId() << ' ' << edge_internal.localId();

        edge_internal.mutableItemBase().setOwner(cell_info.edgeOwner(i_edge), m_mesh_info.rank());
        {
          Node current_node = nodes_map.findItem(first_node);
          m_edge_family.replaceNode(ItemLocalId(edge_internal), 0, current_node);
          m_node_family.addEdgeToNode(current_node, edge_internal);
        }
        {
          Node current_node = nodes_map.findItem(second_node);
          m_edge_family.replaceNode(ItemLocalId(edge_internal), 1, current_node);
          m_node_family.addEdgeToNode(current_node, edge_internal);
        }
        ++m_mesh_info.nbEdge();
      }

      m_cell_family.replaceEdge(ItemLocalId(new_cell), i_edge, ItemLocalId(edge_internal));
      m_edge_family.addCellToEdge(edge_internal, inew_cell);
    }
  }

  // Adds the necessary faces.
  for (Integer i_face = 0; i_face < cell_nb_face; ++i_face) {
    const ItemTypeInfo::LocalFace& lf = cell_type_info->localFace(i_face);
    const Integer face_nb_node = lf.nbNode();
    // en effet de bord, _isReorder positionne m_face_reorderer.sortedNodes();
    const bool is_reorder = _isReorder(i_face, lf, cell_info);
    ConstArrayView<Int64> face_sorted_nodes = m_face_reorderer.sortedNodes();
    bool is_add = false;
    Face face = _findInternalFace(i_face, cell_info, is_add);
    if (is_add) {
      // For elements of order greater than 1, faces are added to the nodes
      // corresponding to the associated order 1 (linear) element.
      const ItemTypeInfo* face_type_info = m_item_type_mng->typeFromId(lf.typeId());
      const Int32 face_nb_linear_node = face_type_info->linearTypeInfo()->nbLocalNode();
      if (is_verbose) {
        info() << "AddFaceToCell (cell_uid=" << new_cell.uniqueId() << ": Create face (index=" << i_face
               << ") uid=" << face.uniqueId()
               << " lid=" << face.localId()
               << " type=" << face_type_info->typeName()
               << " face_nb_linear_node=" << face_nb_linear_node
               << " sorted_nodes=" << face_sorted_nodes;
      }
      face.mutableItemBase().setOwner(cell_info.faceOwner(i_face), m_mesh_info.rank());

      for (Integer i_node = 0; i_node < face_nb_node; ++i_node) {
        if (is_verbose)
          info() << "AddNodeToFace i_node=" << i_node << " uid=" << face_sorted_nodes[i_node];
        Node current_node = nodes_map.findItem(face_sorted_nodes[i_node]);
        m_face_family.replaceNode(face, i_node, current_node);
        if (i_node < face_nb_linear_node) {
          m_node_family.addFaceToNode(current_node, face);
        }
      }

      if (m_mesh_builder->hasEdge()) {
        Integer face_nb_edge = lf.nbEdge();
        for (Integer i_edge = 0; i_edge < face_nb_edge; ++i_edge) {
          Edge current_edge = new_cell.edge(lf.edge(i_edge));
          m_face_family.addEdgeToFace(face, current_edge);
          m_edge_family.addFaceToEdge(current_edge, face);
        }
      }
      ++m_mesh_info.nbFace();
    }
    m_cell_family.replaceFace(new_cell, i_face, face);

    //! AMR
    if (m_mesh->isAmrActivated()) {
      if (is_reorder) {
        if (face.nbCell() == 2)
          m_face_family.replaceFrontCellToFace(face, new_cell);
        else
          m_face_family.addFrontCellToFace(face, inew_cell);
      }
      else {
        if (face.nbCell() == 2)
          m_face_family.replaceBackCellToFace(face, new_cell);
        else
          m_face_family.addBackCellToFace(face, inew_cell);
      }
    }
    else {
      if (is_reorder) {
        m_face_family.addFrontCellToFace(face, inew_cell);
      }
      else {
        m_face_family.addBackCellToFace(face, inew_cell);
      }
    }
  }

  _AMR_Patch(inew_cell, cell_info);
  return ItemCompatibility::_itemInternal(inew_cell);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Adds a parent item
 *
 * The provided item serves as a description for the item to be added to the submesh
 * (at the level of its decomposition into sub-items).
 * The \a submesh_kind argument determines the expected kind of \a item in
 * the submesh.
 *
 * This method allows an item to be added consistently to a submesh
 * from a parent item. The added item will only be connected to items of a lower kind.
 *
 * The item/parent item relationship is materialized by the conserved uid.
 *
 * \retval the added item
 */
ItemInternal* OneMeshItemAdder::
addOneParentItem(const Item& item, const eItemKind submesh_kind, const bool fatal_on_existing_item)
{
  //bool is_check = arcaneIsCheck();
  ItemTypeMng* itm = m_mesh->itemTypeMng();
  eItemKind kind = item.kind();
  ItemTypeInfo* type = itm->typeFromId(item.type());

  if (item.type() == IT_Line2 && submesh_kind == IK_Cell)
    type = itm->typeFromId(IT_CellLine2);
  if (item.type() == IT_Vertex && submesh_kind == IK_Face)
    type = itm->typeFromId(IT_FaceVertex);
  ItemTypeId type_id = type->itemTypeId();

  if (MeshToMeshTransposer::kindTranspose(submesh_kind, m_mesh, m_mesh->parentMesh()) != kind)
    ARCANE_FATAL("Incompatible kind/sub-kind");

  // Checks if the cell already exists
  bool is_add; // this flag is always correctly set via findOrAllocOne
  Item new_item;

  switch (submesh_kind) {
  case IK_Node:
    new_item = m_node_family.findOrAllocOne(item.uniqueId(), is_add);
    ++m_mesh_info.nbNode();
    break;
  case IK_Edge:
    new_item = m_edge_family.findOrAllocOne(item.uniqueId(), is_add);
    ++m_mesh_info.nbEdge();
    break;
  case IK_Face:
    new_item = m_face_family.findOrAllocOne(item.uniqueId(), type, is_add);
    ++m_mesh_info.nbFace();
    break;
  case IK_Cell:
    if (kind == IK_Face && !(item.toFace().isSubDomainBoundary()))
      ARCANE_FATAL("Bad boundary face");
    new_item = m_cell_family.findOrAllocOne(item.uniqueId(), type_id, is_add);
    ++m_mesh_info.nbCell();
    break;
  default:
    throw NotSupportedException(A_FUNCINFO, String::format("Kind {0} not supported", submesh_kind));
  }

  if (!is_add) {
    if (fatal_on_existing_item)
      ARCANE_FATAL("Cannot add already existing parent item in submesh");
    else
      return ItemCompatibility::_itemInternal(new_item);
  }

  new_item.mutableItemBase().setParent(0, item.localId());
  new_item.mutableItemBase().setOwner(item.owner(), m_mesh_info.rank());

  // Localizes these sub-items relative to the item to be inserted
  // By default everything is 0, which also corresponds to the submesh_kind==IK_Node case
  Integer item_nb_node = 0;
  Integer item_nb_face = 0;
  Integer item_nb_edge = 0;

  switch (submesh_kind) {
  case IK_Cell:
    item_nb_face = type->nbLocalFace();
    item_nb_edge = type->nbLocalEdge();
    item_nb_node = type->nbLocalNode();
    break;
  case IK_Face:
    item_nb_edge = type->nbLocalEdge();
    item_nb_node = type->nbLocalNode();
    break;
  case IK_Edge:
    item_nb_node = type->nbLocalNode();
  default: // les autres sont déjà filtrés avant
    break;
  }

  // Handling the case of edge deactivation
  if (!m_mesh_builder->hasEdge())
    item_nb_edge = 0;

#if OLD
  // Ne fonctionne plus si on désactive les anciennes connectivités
  if (is_check) {
    for (Integer z = 0; z < item_nb_face; ++z)
      new_item->_setFace(z, NULL_ITEM_ID);
    for (Integer z = 0; z < item_nb_edge; ++z)
      new_item->_setEdge(z, NULL_ITEM_ID);
    for (Integer z = 0; z < item_nb_node; ++z)
      new_item->_setNode(z, NULL_ITEM_ID);
  }
#endif

  //! Type la table de hashage uniqueId()->ItemInternal*
  DynamicMeshKindInfos::ItemInternalMap& nodes_map = m_mesh->nodesMap();
  auto* parent_mesh = ARCANE_CHECK_POINTER(dynamic_cast<DynamicMesh*>(m_mesh->parentMesh()));
  DynamicMeshKindInfos::ItemInternalMap& parent_nodes_map = parent_mesh->nodesMap();

  // Processing new nodes

  // The vertices are used in the item order except for surface submeshes.
  const bool direct_node_order =
  !(submesh_kind == IK_Cell && kind == IK_Face && !(item.toFace().isSubDomainBoundaryOutside()));

  Int64UniqueArray nodes_uid(item_nb_node, NULL_ITEM_UNIQUE_ID);
  for (Integer i_node = 0; i_node < item_nb_node; ++i_node) {
    Item parent_item;
    if (type->typeId() == IT_FaceVertex)
      parent_item = item;
    else {
      Int32 idx = ((direct_node_order) ? i_node : (item_nb_node - 1 - i_node));
      parent_item = item.toItemWithNodes().node(idx);
    }
    Int64 new_node_uid = nodes_uid[i_node] = parent_item.uniqueId();
    ItemInternal* node_internal = m_node_family.findOrAllocOne(new_node_uid, is_add);
    if (is_add) {
#ifdef ARCANE_DEBUG_DYNAMIC_MESH
      info() << "Création node " << new_node_uid << ' '
             << node_internal->uniqueId() << ' ' << node_internal->localId();
#endif /* ARCANE_DEBUG_DYNAMIC_MESH */
      node_internal->setParent(0, parent_item.localId());
      node_internal->setOwner(parent_item.owner(), m_mesh_info.rank());
      ++m_mesh_info.nbNode();
    }

    // Connecting the item to the nodes
    ItemLocalId node_lid(node_internal);
    ItemLocalId new_item_lid(new_item);
    switch (submesh_kind) {
    case IK_Cell:
      m_cell_family.replaceNode(new_item_lid, i_node, node_lid);
      m_node_family.addCellToNode(node_internal, new_item.toCell());
      break;
    case IK_Face:
      m_face_family.replaceNode(new_item_lid, i_node, node_lid);
      m_node_family.addFaceToNode(node_internal, new_item.toFace());
      break;
    case IK_Edge:
      m_edge_family.replaceNode(new_item_lid, i_node, node_lid);
      m_node_family.addEdgeToNode(node_internal, new_item.toEdge());
      break;
    default: // les autres sont déjà filtrés avant
      break;
    }
  }

  // Processing new edges (the has_edge filter is already taken into account in item_nb_edge)
  for (Integer i_edge = 0; i_edge < item_nb_edge; ++i_edge) {
    const ItemTypeInfo::LocalEdge& le = type->localEdge(i_edge);

    Int64 first_node = nodes_uid[le.beginNode()];
    Int64 second_node = nodes_uid[le.endNode()];
    if (first_node > second_node)
      std::swap(first_node, second_node);

    Edge parent_item = item.itemBase().edgeBase(i_edge);
    if (parent_item.null())
      ARCANE_FATAL("Cannot find parent edge");

    Int64 new_edge_uid = parent_item.uniqueId();

    ItemInternal* edge_internal = m_edge_family.findOrAllocOne(new_edge_uid, is_add);
    if (is_add) {
#ifdef ARCANE_DEBUG_DYNAMIC_MESH
      info() << "Création edge " << new_edge_uid << ' '
             << edge_internal->uniqueId() << ' ' << edge_internal->localId();
#endif /* ARCANE_DEBUG_DYNAMIC_MESH */
      edge_internal->setParent(0, parent_item.localId());
      edge_internal->setOwner(parent_item.owner(), m_mesh_info.rank());

      {
        Node current_node = nodes_map.findItem(first_node);
        m_edge_family.replaceNode(ItemLocalId(edge_internal), 0, current_node);
        m_node_family.addEdgeToNode(current_node, edge_internal);
      }
      {
        Node current_node = nodes_map.findItem(second_node);
        m_edge_family.replaceNode(ItemLocalId(edge_internal), 1, current_node);
        m_node_family.addEdgeToNode(current_node, edge_internal);
      }
      ++m_mesh_info.nbEdge();
    }

    // Connecting the item to the nodes
    switch (submesh_kind) {
    case IK_Cell: {
      m_cell_family.replaceEdge(ItemLocalId(new_item), i_edge, ItemLocalId(edge_internal));
      m_edge_family.addCellToEdge(edge_internal, new_item.toCell());
    } break;
    case IK_Face: {
      m_face_family.replaceEdge(ItemLocalId(new_item), i_edge, ItemLocalId(edge_internal));
      m_edge_family.addFaceToEdge(edge_internal, new_item.toFace());
    } break;
    default: // les autres sont déjà filtrés avant
      break;
    }
  }

  // Processing new faces
  // item_nb_face already materializes that this context only occurs with submesh_kind==IK_Cell
  for (Integer i_face = 0; i_face < item_nb_face; ++i_face) {
    const ItemTypeInfo::LocalFace& lf = type->localFace(i_face);
    Integer face_nb_node = lf.nbNode();

    m_work_face_orig_nodes_uid.resize(face_nb_node);
    for (Integer z = 0; z < face_nb_node; ++z)
      m_work_face_orig_nodes_uid[z] = nodes_uid[lf.node(z)];
    bool is_reorder = false;
    if (m_mesh->dimension() == 1) { // is 1d mesh
      is_reorder = m_face_reorderer.reorder1D(i_face, m_work_face_orig_nodes_uid[0]);
    }
    else {
      is_reorder = m_face_reorderer.reorder(ItemTypeId::fromInteger(lf.typeId()), m_work_face_orig_nodes_uid);
    }
    ConstArrayView<Int64> face_sorted_nodes(m_face_reorderer.sortedNodes());
    // find parent item
    Item parent_item;
    if (kind == IK_Cell) {
      parent_item = item.toCell().face(i_face);
    }
    else if (kind == IK_Face) {
      if (m_mesh->dimension() == 1) { // is 1d mesh
        parent_item = parent_nodes_map.findItem(face_sorted_nodes[0]);
      }
      else {
        // Algo sans CT_FaceToEdge
        Int64 first_node = face_sorted_nodes[0];
        Int64 second_node = face_sorted_nodes[1];
        if (first_node > second_node)
          std::swap(first_node, second_node);
        Node nbi = parent_nodes_map.findItem(first_node);
        parent_item = ItemTools::findEdgeInNode2(nbi, first_node, second_node);
      }
    }

    if (parent_item.null())
      ARCANE_FATAL("Cannot find parent face");
    Int64 new_face_uid = parent_item.uniqueId();
    ItemTypeInfo* face_type = itm->typeFromId(lf.typeId());

    ItemInternal* face_internal = m_face_family.findOrAllocOne(new_face_uid, face_type, is_add);
    if (is_add) {
#ifdef ARCANE_DEBUG_DYNAMIC_MESH
      info() << "Création face " << new_face_uid << ' '
             << face_internal->uniqueId() << ' ' << face_internal->localId();
#endif /* ARCANE_DEBUG_DYNAMIC_MESH */
      face_internal->setParent(0, parent_item.localId());
      face_internal->setOwner(parent_item.owner(), m_mesh_info.rank());

      for (Integer i_node = 0; i_node < face_nb_node; ++i_node) {
        Node current_node = nodes_map.findItem(face_sorted_nodes[i_node]);
        m_face_family.replaceNode(ItemLocalId(face_internal), i_node, current_node);
        m_node_family.addFaceToNode(current_node, face_internal);
      }

      if (m_mesh_builder->hasEdge()) {
        Integer face_nb_edge = lf.nbEdge();
        for (Integer i_edge = 0; i_edge < face_nb_edge; ++i_edge) {
          Int32 edge_idx = lf.edge(i_edge);
          Edge current_edge = new_item.itemBase().edgeBase(edge_idx);
          m_face_family.addEdgeToFace(face_internal, current_edge);
          m_edge_family.addFaceToEdge(current_edge, face_internal);
        }
      }
      ++m_mesh_info.nbFace();
    }
    m_cell_family.replaceFace(ItemLocalId(new_item), i_face, ItemLocalId(face_internal));
    if (is_reorder) {
      m_face_family.addFrontCellToFace(face_internal, new_item.toCell());
    }
    else {
      m_face_family.addBackCellToFace(face_internal, new_item.toCell());
    }
  }

  return ItemCompatibility::_itemInternal(new_item);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Adds the nodes \a nodes_uid to the cell \a cell
 */
template <typename CellInfo>
inline void OneMeshItemAdder::
_addNodesToCell(Cell cell, const CellInfo& cell_info)
{
  Integer cell_nb_node = cell_info.nbNode();

  // Adds new nodes if necessary
  for (Integer i_node = 0; i_node < cell_nb_node; ++i_node) {
    Int64 node_unique_id = cell_info.nodeUniqueId(i_node);
    bool is_add = false;
    ItemInternal* node_internal = m_node_family.findOrAllocOne(node_unique_id, is_add);
    if (is_add) {
      ++m_mesh_info.nbNode();
      node_internal->setOwner(cell_info.nodeOwner(i_node), m_mesh_info.rank());
    }
    m_node_family.addCellToNode(node_internal, cell);
    m_cell_family.replaceNode(cell, i_node, ItemLocalId(node_internal));
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename CellInfo>
bool OneMeshItemAdder::
_isReorder(Integer i_face, const ItemTypeInfo::LocalFace& lf, const CellInfo& cell_info)
{
  const Integer face_nb_node = lf.nbNode();
  m_work_face_orig_nodes_uid.resize(face_nb_node);
  for (Integer i_node = 0; i_node < face_nb_node; ++i_node)
    m_work_face_orig_nodes_uid[i_node] = cell_info.nodeUniqueId(lf.node(i_node));
  bool is_reorder = false;
  if (m_mesh->dimension() == 1) { // is 1d mesh
    is_reorder = m_face_reorderer.reorder1D(i_face, m_work_face_orig_nodes_uid[0]);
  }
  else {
    is_reorder = m_face_reorderer.reorder(ItemTypeId::fromInteger(lf.typeId()), m_work_face_orig_nodes_uid);
  }

  return is_reorder;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Resets structures to allow for another allocation
void OneMeshItemAdder::
resetAfterDeallocate()
{
  m_next_face_uid = 0;
  m_next_edge_uid = 0;
  m_mesh_info.reset();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void OneMeshItemAdder::
setUseNodeUniqueIdToGenerateEdgeAndFaceUniqueId(bool v)
{
  if (m_next_face_uid != 0 || m_next_edge_uid != 0)
    ARCANE_FATAL("Can not call this method when edge or face are already created");
  m_use_hash_for_edge_and_face_unique_id = v;
  info() << "Is Generate Edge and Face uniqueId() from Nodes=" << m_use_hash_for_edge_and_face_unique_id;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::mesh

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
