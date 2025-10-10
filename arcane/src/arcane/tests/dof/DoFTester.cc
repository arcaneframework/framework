// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DoFTester.cc                                                (C) 2000-2025 */
/*                                                                           */
/* Comment on file content                                                   */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/BasicUnitTest.h"

#include "arcane/IParticleExchanger.h"

#include "arcane/tests/dof/DoFTester_axl.h"

#include "arcane/tests/ArcaneTestGlobal.h"

#include "arcane/mesh/DoFManager.h"
#include "arcane/mesh/DoFFamily.h"
#include "arcane/mesh/ItemConnectivity.h"
#include "arcane/mesh/ItemConnectivityMng.h"
#include "arcane/mesh/GhostLayerFromConnectivityComputer.h"

#include "arcane/core/IItemConnectivitySynchronizer.h"
#include "arcane/core/IMesh.h"
#include "arcane/core/VariableTypedef.h"
#include "arcane/core/VariableBuildInfo.h"
#include "arcane/core/ItemEnumerator.h"
#include "arcane/core/ItemUniqueId.h"
#include "arcane/core/VariableTypes.h"
#include "arcane/core/ItemTypes.h"
#include "arcane/mesh/CellFamily.h"
#include "arcane/core/ItemVector.h"
#include "arcane/core/Item.h"
#include "arcane/mesh/ParticleFamily.h"
#include "arcane/mesh/NodeFamily.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/IMeshModifier.h"
#include "arcane/core/MeshKind.h"
#include "arcane/core/VariableCollection.h"

#include "arcane/core/internal/IPolyhedralMeshModifier.h"
#include "arcane/core/internal/IMeshInternal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace ArcaneTest
{

using namespace Arcane;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Service de test de DoF.
 */
class DoFTester
: public ArcaneDoFTesterObject
{
 public:

  DoFTester(const ServiceBuildInfo& sbi)
    : ArcaneDoFTesterObject(sbi)
    , m_dof_mng(sbi.mesh())
    , m_dof_family_name("DoFFamily")
    , m_dof_on_cell_family_name("DoFOnCell")
    , m_dofs_on_cell_family_name("DoFsOnCell")
    , m_dof_on_node_family_name("DoFOnNode")
    , m_dofs_on_node_family_name("DoFsOnNode")
    , m_dofs_multi_on_face_family_name("DoFsMultiOnFace")
    , m_dofs_multi_on_node_family_name("DoFsMultiOnNode"){}

public:

 virtual void executeTest();

 typedef ItemConnectivityT          <Cell,DoF>  CellToDoFConnectivity;
 typedef ItemArrayConnectivityT     <Cell, DoF> CellToDoFsConnectivity;
 typedef ItemConnectivityT          <Node, DoF> NodeToDoFConnectivity;
 typedef ItemArrayConnectivityT     <Node, DoF> NodeToDoFsConnectivity;
 typedef ItemMultiArrayConnectivityT<Face, DoF> FaceToDoFsMultiConnectivity;
 typedef ItemMultiArrayConnectivityT<Node, DoF> NodeToDoFsMultiConnectivity;

 typedef SharedArray2<Int32> Int32SharedArray2;
 typedef SharedArray2<Int64> Int64SharedArray2;

private:

 DoFManager& dofMng() {return m_dof_mng;}
 void addDoF(const Integer size, const Integer begin_index);
 void removeDoF();
 void doFGroups();
 void doFVariable();
 void doFConnectivity();


private:

 // Connectivity tests
 void _testItemProperty();
 void _node2DoFConnectivity();
 void _cell2DoFsConnectivity();
 void _Face2DoFsMultiConnectivity();
 void _node2DoFConnectivityRegistered();
 void _node2DoFsConnectivityRegistered();
 void _node2DoFsMultiConnectivityRegistered();

 // Test tools
 void _enumerateDoF(const DoFGroup& dof_group);
 void _printVariable(VariableDoFReal& dof_variable, bool do_check);
 void _printArrayVariable(VariableDoFArrayReal& dof_variable, bool do_check);
 void _removeGhost(IDoFFamily* dof_family);
 void _addNodes(Int32Array2View new_nodes_lids, Int64Array2View new_nodes_uids);
 void _removeNodes(Int32ConstArray2View new_nodes_lids,
                   const Integer nb_removed_nodes,
                   Int32SharedArray2& removed_node_lids,
                   Int64SharedArray2& removed_node_uids,
                   Int32SharedArray2& remaining_node_lids,
                   Int64SharedArray2& remaining_node_uids);
 void _checkConnectivityUpdateAfterAdd    (IItemConnectivity& node2dof, Int32Array2View new_nodes_lids, Int64ConstArray2View new_nodes_uids, IntegerConstArrayView nb_dof_per_item, bool is_scalar_connectivity=true);
 void _checkConnectivityUpdateAfterRemove (IItemConnectivity& node2dof, Int32Array2View new_nodes_lids, Int64ConstArray2View new_nodes_uids, bool is_scalar_connectivity=true);
 void _checkConnectivityUpdateAfterCompact(IItemConnectivity& node2dof, Int32Array2View remaining_nodes_lids,Int64ConstArray2View remaining_nodes_uids,Integer item_property_size, bool is_scalar_connectivity=true);

 void _checkTargetFamilyInfo(ItemVectorView tracked_new_dofs, Int32ConstArrayView new_dofs_lids);

 bool _checkIsSame(IItemConnectivity* connectivity1, IItemConnectivity* connectivity2);

 template <typename Connectivity>
 void _printNodeToDoFConnectivity(Connectivity& con, bool is_scalar_connectivity, bool do_check = false) ;


private:
  DoFManager m_dof_mng;
  String m_dof_family_name;
  String m_dof_on_cell_family_name;  // One dof per cell
  String m_dofs_on_cell_family_name; // Several dofs per cell
  String m_dof_on_node_family_name;  // One dof per node
  String m_dofs_on_node_family_name; // Several dofs per node
  String m_dofs_multi_on_face_family_name; // Several dofs per face (non constant size)
  String m_dofs_multi_on_node_family_name; // Several dofs per node (non constant size)
  bool m_do_compact = true;

};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DoFTester::
executeTest()
{
  info() << "================ WELCOME TO DOF TESTER =====================";

  if (mesh()->meshKind().meshStructure() == eMeshStructure::Polyhedral) m_do_compact = false;
  addDoF(5,0);
  removeDoF();
  doFGroups();
  doFConnectivity();
  doFVariable();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/


void
DoFTester::
addDoF(const Integer size, const Integer begin_index)
{
  // Create DoF

  Int64UniqueArray dof_uids(size);
  for (Integer i = 0; i < size; ++i) dof_uids[i] = i+begin_index;

  IDoFFamily* dof_family = dofMng().getFamily(m_dof_family_name); // Lazy
  info() << "=== add items to family " << dof_family->name();
  info() << "=== add items to family " << dof_family;

  Int32UniqueArray dof_lids(size);

  DoFVectorView dofs = dof_family->addDoFs(dof_uids,dof_lids);
  dof_family->endUpdate();

  // CHECK
  info() << "=== mesh()->findItemFamily(IK_DoF,m_dof_family_name) " << mesh()->findItemFamily(IK_DoF,m_dof_family_name)->name();
  info() << "=== dof_family nbItem " << dof_family->nbItem();
  //info() << "=== dof family uniqueIds " << Int64UniqueArray(*dof_family.uniqueIds());
  info() << "=== dof family localIds " <<  Int32UniqueArray(dof_family->itemFamily()->view().localIds());


  info() << "=== dofs.size = " << dofs.size();

  // Enumerate DoF
  Int64 dof_id;

  ENUMERATE_DOF(idof,dofs) {
    dof_id = idof->uniqueId().asInt64();
    info() << "= Dof id : " << dof_id;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DoFTester::
removeDoF()
{
  IDoFFamily* dof_family = dofMng().getFamily(m_dof_family_name);
  info() << "=== remove items from family " << dof_family->name();

  Int32ConstArrayView dof_lids = dof_family->itemFamily()->view().localIds();
  dof_family->removeDoFs(dof_lids.subConstView(3,2));
  dof_family->endUpdate();

  info() << "=== dofs.size = " << dof_family->nbItem();

  // Enumerate DoF
  Int64 dof_id;
  ENUMERATE_DOF(idof,dof_family->itemFamily()->view()) {
    dof_id = idof->uniqueId().asInt64();
    info() << "= Dof id : " << dof_id;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DoFTester::
doFGroups()
{
  IDoFFamily* dof_family = dofMng().getFamily(m_dof_family_name);
  IItemFamily* item_family = dof_family->itemFamily();

  info() << "=== Test groups and goup creation in family " << dof_family->name();

  // CREATE NEW GROUP; try ArcGeoSim ItemGroupBuilder
  item_family->createGroup("EmptyDoFGroup");
  Int32ConstArrayView lids = item_family->view().localIds().subConstView(0,dof_family->nbItem()-1);
  String test_dof_group_name("TestDoFGroup");
  item_family->createGroup(test_dof_group_name,lids);
  ItemGroup test_dof_group = item_family->findGroup(test_dof_group_name);
  item_family->createGroup("TestDoFGroupFromParent",test_dof_group);

  for (ItemGroupCollection::Iterator ite = item_family->groups().begin(); ite != item_family->groups().end();++ite) {
    _enumerateDoF(*ite);
  }

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DoFTester::
_enumerateDoF(const DoFGroup& dof_group)
{
  Int64 dof_id;
  info() << "=== Enumerate dof group " << dof_group.name();
  ENUMERATE_DOF(idof,dof_group) {
    dof_id = idof->uniqueId().asInt64();
    info() << "= Dof id : " << dof_id;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DoFTester::
doFConnectivity()
{
  info() << "================================";
  info() << "=== DoF Connectivity tests ===";
  info() << "================================";

  _testItemProperty();

  _node2DoFConnectivity();
  _cell2DoFsConnectivity();
  _Face2DoFsMultiConnectivity();

  _node2DoFConnectivityRegistered();
  _node2DoFsConnectivityRegistered();
  _node2DoFsMultiConnectivityRegistered();

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DoFTester::
_testItemProperty()
{
  ItemScalarProperty<Integer> item_property;
  item_property.resize(mesh()->cellFamily(),NULL_ITEM_LOCAL_ID);
  ENUMERATE_CELL(icell,ownCells()) {
    item_property[icell] = icell.localId();
  }
  ENUMERATE_CELL(icell,ownCells()) {
    if (item_property[icell] != icell.localId())
      ARCANE_FATAL("Error on item property");
  }

  // Test resize
  ItemMultiArrayProperty<Integer> item_multi_array_property;
  Integer nb_elements = mesh()->cellFamily()->maxLocalId();
  IntegerUniqueArray dim2_sizes(nb_elements);
  for (Arcane::Integer i = 0; i < nb_elements; ++i) {
    dim2_sizes[i] = Integer(math::pow(double(2),double(i%2))); // 1 or 2 elements per item.
  }
  item_multi_array_property.resize(mesh()->cellFamily(),dim2_sizes,NULL_ITEM_LOCAL_ID);

  // Check initialization
  ENUMERATE_CELL(icell,allCells()) {
    for ( Integer j = 0; j < dim2_sizes[icell.index()]; ++j) {
      if (item_multi_array_property[icell][j] != NULL_ITEM_LOCAL_ID)
        ARCANE_FATAL("Error in ItemMultiArrayProperty resize");
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DoFTester::
_node2DoFConnectivity()
{
  info() << "================================";
  info() << "=== NODE TO DOF CONNECTIVITY ===";
  info() << "================================";

  // Create a DoF Family to link with mesh nodes
  IDoFFamily* dof_on_node_family = dofMng().getFamily(m_dof_on_node_family_name);
  IItemFamily* node_family = mesh()->nodeFamily();
  // Generation des uids : proposition d'un utilitaire (basique) dans les outils des dof
  Int64UniqueArray uids;
  // Create dof on own node only. Ghost handled in the following by GhostDoFManager
  ENUMERATE_NODE(inode,ownNodes()) {
    info() << "Add dof on node " << inode.localId() << "," << inode->uniqueId().asInt64() << " owner " << inode->owner();
    uids.add(mesh::DoFUids::uid(inode->uniqueId().asInt64()));
  }
  Int32UniqueArray lids(uids.size());
  dof_on_node_family->addDoFs(uids,lids);
  dof_on_node_family->endUpdate();

  // Create connectivity
  NodeToDoFConnectivity node2dof(node_family,dof_on_node_family->itemFamily(),"NodeToDoF");

  info() << "== Create connectivity " << node2dof.name();

  // Get connected families
  ConstArrayView<IItemFamily*> families = node2dof.families();
  info() << "== Connect item family " << families[0]->name() << " with item family " << families[1]->name();

  // Construct dof ghost layer
  GhostLayerFromConnectivityComputer ghost_builder(&node2dof);
  IItemConnectivitySynchronizer* synchronizer= dofMng().connectivityMng()->createSynchronizer(&node2dof,&ghost_builder);
  synchronizer->synchronize();

  // Use it: get DoF from cell
  Int64 dof_uid;
  ENUMERATE_NODE(inode,allNodes()) {
    const DoF& connected_dof = node2dof(inode);
    dof_uid = connected_dof.uniqueId().asInt64();
    info() << "Node " << inode.localId() << "," << inode->uniqueId().asInt64() << " owner " << inode->owner();
    info() << "is connected to dof " << connected_dof.localId() << "," << connected_dof.uniqueId() << " owner " << connected_dof.owner();
    // Check Ghost policy node and dof have same owner
    if (connected_dof.owner() != inode->owner()) fatal() << "Error in node to dof connectivity ghost policy";
    info() << String::format("dof uid {0} owned by {1} connected to node uid {2} owned by {3} ",
                                      dof_uid, connected_dof.owner(), inode->uniqueId().asInt64(), inode->owner());
  }
  // New api to avoid ConnectivityItemVector uncomprehension
  // Inside an enumerate: create the connectivity vector only once
  ConnectivityItemVector dof_vec(node2dof);
  info() << "Try new API...";
  ENUMERATE_NODE(inode,allNodes())
  {
    dof_vec = node2dof._connectedItems(inode);
    ENUMERATE_DOF(idof,dof_vec){
      dof_uid = idof->uniqueId().asInt64();
      info() << String::format("dof uid {0} owned by {1}  connected to node uid {2} owned by {3}  ",
                                        dof_uid, idof->owner(), inode->uniqueId().asInt64(), inode->owner());
    }
  }
  // Outside an enumerate, for a one-shot use
  if (mesh()->nodeFamily()->nbItem() > 0) {
    NodeInfoListView nodes_view(mesh()->nodeFamily());
    Node my_node(nodes_view[0]);
    ConnectivityItemVector dof_vec2 = node2dof._connectedItems(my_node);
    ENUMERATE_DOF(idof,dof_vec2){
      info() << "dof lid " << idof->localId() ;
      dof_uid = idof->uniqueId().asInt64();
      info() << String::format("dof uid {0} owned by {1}  connected to node uid {2} owned by {3}  ",
                               dof_uid, idof->owner(), my_node.uniqueId().asInt64(), my_node.owner());
    }
  }

  // clean
  _removeGhost(dof_on_node_family);

  // Test constructor by item_property
  NodeToDoFConnectivity node2dof_2(node_family,dof_on_node_family->itemFamily(),node2dof.itemProperty(),"NodeToDoF2");
  if (!_checkIsSame(&node2dof,&node2dof_2))
    ARCANE_FATAL("Error in connectivity construction from ItemProperty");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void
DoFTester::
_cell2DoFsConnectivity()
{
  info() << "================================";
  info() << "== CELL TO DOFS CONNECTIVITY    ";
  info() << "================================";

  // Create a DoF Family to link with mesh cells
  IDoFFamily* dofs_on_cell_family = dofMng().getFamily(m_dofs_on_cell_family_name);

  Integer nb_dof_per_cell = 3;

  // Generation des uids
  Int64UniqueArray uids;
  Int64 max_cell_uid = mesh::DoFUids::getMaxItemUid(mesh()->cellFamily());
  Int64 max_dof_uid  = mesh::DoFUids::getMaxItemUid(dofs_on_cell_family->itemFamily());
  ENUMERATE_CELL(icell,ownCells())
  {
    for (Integer i = 0; i < nb_dof_per_cell; ++i)
      {
        uids.add(mesh::DoFUids::uid(max_dof_uid,max_cell_uid,icell->uniqueId().asInt64(),i)); // unique id parallel generation = TODO (voir ce que fait Arcane !)
      }
  }

  Int32UniqueArray lids(uids.size());
  dofs_on_cell_family->addDoFs(uids,lids);
  dofs_on_cell_family->endUpdate();

  // Create connectivity
  CellToDoFsConnectivity cell2dofs(mesh()->cellFamily(),dofs_on_cell_family->itemFamily(),nb_dof_per_cell,"CellToDoFs");

  info() << "== Create connectivity " << cell2dofs.name();

  // Create ghost
  GhostLayerFromConnectivityComputer ghost_builder(&cell2dofs);
  IItemConnectivitySynchronizer* synchronizer = dofMng().connectivityMng()->createSynchronizer(&cell2dofs,&ghost_builder);
  synchronizer->synchronize();

  // Get connected families
  ConstArrayView<IItemFamily*> families = cell2dofs.families();
  info() << "== Connect item family " << families[0]->name() << " with item family " << families[1]->name();

  // Use it: get DoF from cell
  Int64 dof_uid;
  ConnectivityItemVector dof_vec(cell2dofs);
  ENUMERATE_CELL(icell,allCells())
  {
    cell2dofs(icell,dof_vec);
    ENUMERATE_DOF(idof,dof_vec){
      dof_uid = idof->uniqueId().asInt64();
      info() << String::format("dof uid {0} owned by {1}  connected to cell uid {3} owned by {4}  ",
                                        dof_uid, idof->owner(), icell->uniqueId().asInt64(), icell->owner());
    }
  }
  info() << "Test new API...";
  // New api to avoid ConnectivityItemVector misuse
  ENUMERATE_CELL(icell,allCells())
  {
    dof_vec = cell2dofs(icell);
    ENUMERATE_DOF(idof,dof_vec){
      dof_uid = idof->uniqueId().asInt64();
      info() << String::format("dof uid {0} owned by {1}  connected to cell uid {3} owned by {4}  ",
                                        dof_uid, idof->owner(), icell->uniqueId().asInt64(), icell->owner());
    }
  }
  // For one-shot use
  if (mesh()->cellFamily()->nbItem() > 0) {
    CellInfoListView cells_view(mesh()->cellFamily());
    Cell my_cell(cells_view[0]);
    ConnectivityItemVector dof_vec2 = cell2dofs(my_cell);
    ENUMERATE_DOF(idof,dof_vec2){
      dof_uid = idof->uniqueId().asInt64();
      info() << String::format("dof uid {0} owned by {1}  connected to cell uid {2} owned by {3}  ",
                               dof_uid, idof->owner(), my_cell.uniqueId().asInt64(), my_cell.owner());
    }
  }

  // Test constructor by item_property
  CellToDoFsConnectivity cell2dofs_2(mesh()->cellFamily(),dofs_on_cell_family->itemFamily(),cell2dofs.itemProperty(),"CellToDoFs2");
  if (!_checkIsSame(&cell2dofs,&cell2dofs_2)) fatal() << "Error in connectivity construction from ItemArrayProperty";

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void
DoFTester::
_Face2DoFsMultiConnectivity()
{
  info() << "================================";
  info() << "== FACE TO DOFS MULTI CONNECTIVITY ";
  info() << "================================";

  // Create a DoF Family to link with mesh Faces : variable number of DoFs per face
  IDoFFamily* dofs_multi_on_face_family = dofMng().getFamily(m_dofs_multi_on_face_family_name);

  // Generation des uids
  Int64UniqueArray uids;
  IntegerUniqueArray nb_dof_per_faces(mesh()->faceFamily()->maxLocalId(),0); // Be careful not to have any non initialized values in this size array
  Int64 max_face_uid = mesh::DoFUids::getMaxItemUid(mesh()->faceFamily());
  Int64 max_dof_uid  = mesh::DoFUids::getMaxItemUid(dofs_multi_on_face_family->itemFamily());
  // silly construction for demo : one dof on face with odd local_id, two on faces with even local_id
  Integer nb_dof_on_face ;
  ENUMERATE_FACE(iface,ownFaces())
  {
    nb_dof_on_face = iface->localId()% 2 + 1;
    nb_dof_per_faces[iface->localId()] = nb_dof_on_face;
    for (Integer i = 0; i < nb_dof_on_face; ++i) uids.add(mesh::DoFUids::uid(max_dof_uid,max_face_uid,iface->uniqueId().asInt64(),i));
  }

  Int32UniqueArray lids(uids.size());
  dofs_multi_on_face_family->addDoFs(uids,lids);
  dofs_multi_on_face_family->endUpdate();

  // Create connectivity
  FaceToDoFsMultiConnectivity face2dofs(mesh()->faceFamily(),dofs_multi_on_face_family->itemFamily(),nb_dof_per_faces,"FaceToDoFsMulti");

  info() << "== Create connectivity " << face2dofs.name();

  // Create ghost
  GhostLayerFromConnectivityComputer ghost_builder(&face2dofs);
  IItemConnectivitySynchronizer* connectivity_synchronizer = dofMng().connectivityMng()->createSynchronizer(&face2dofs,&ghost_builder);
  connectivity_synchronizer->synchronize();

  // Get connected families
  ConstArrayView<IItemFamily*> families = face2dofs.families();
  info() << "== Connect item family " << families[0]->name() << " with item family " << families[1]->name();

  // Use it: get DoF from cell
  ConnectivityItemVector dof_vec(face2dofs);
  Int64 dof_uid;
  ENUMERATE_FACE(iface,allFaces()){
    ENUMERATE_DOF(idof,face2dofs(iface,dof_vec)){
      dof_uid = idof->uniqueId().asInt64();
      info() << String::format("dof uid {0} owned by {1} is own {2} connected to cell uid {3} owned by {4} is own {5} ",
                                       dof_uid, idof->owner(), idof->isOwn(), iface->uniqueId().asInt64(), iface->owner(), iface->isOwn());
    }
  }
  // New api to avoid ConnectivityItemVector misuse
  info() << "Test new API...";
  ENUMERATE_FACE(iface,allFaces()){
      dof_vec = face2dofs(iface);
      ENUMERATE_DOF(idof,dof_vec){
        dof_uid = idof->uniqueId().asInt64();
        info() << String::format("dof uid {0} owned by {1} is own {2} connected to cell uid {3} owned by {4} is own {5} ",
                                         dof_uid, idof->owner(), idof->isOwn(), iface->uniqueId().asInt64(), iface->owner(), iface->isOwn());
      }
    }
  // For one-shot use
  if (mesh()->faceFamily()->nbItem() > 0) {
    FaceInfoListView faces_view(mesh()->faceFamily());
    Face my_face(faces_view[0]);
    ConnectivityItemVector dof_vec2 = face2dofs(my_face);
    ENUMERATE_DOF(idof,dof_vec2){
      dof_uid = idof->uniqueId().asInt64();
      info() << String::format("dof uid {0} owned by {1}  connected to face uid {2} owned by {3}  ",
                               dof_uid, idof->owner(), my_face.uniqueId().asInt64(), my_face.owner());
    }
  }

  // Test constructor by item_property
  FaceToDoFsMultiConnectivity face2dofs_2(mesh()->faceFamily(),dofs_multi_on_face_family->itemFamily(),face2dofs.itemProperty(),"FaceToDoFsMulti2");
  if (!_checkIsSame(&face2dofs,&face2dofs_2)) fatal() << "Error in connectivity construction from ItemMultiArrayProperty";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void
DoFTester::
_node2DoFConnectivityRegistered()
{
  info() << "====================================================";
  info() << "== REGISTER A CONNECTIVITY AND FOLLOW MESH EVOLUTION ";
  info() << "====================================================";

  // Done for node since it's easier to add new items
  IDoFFamily* dof_on_node_family = dofMng().getFamily(m_dof_on_node_family_name);
  IItemFamily* node_family = mesh()->nodeFamily();

  NodeToDoFConnectivity node2dof(node_family,dof_on_node_family->itemFamily(),"NodeToDoF");
  GhostLayerFromConnectivityComputer ghost_builder(&node2dof);
  IItemConnectivitySynchronizer* synchronizer = dofMng().connectivityMng()->createSynchronizer(&node2dof,&ghost_builder);
  synchronizer->synchronize();

  // Save your connectivity
  dofMng().connectivityMng()->registerConnectivity(&node2dof);

  // Local mesh changes: add own and ghost nodes
  Integer nb_subdomain = subDomain()->parallelMng()->commSize();
  Integer nb_new_nodes_per_subdomain = 3;
  Int64UniqueArray2 new_nodes_uids(nb_subdomain,nb_new_nodes_per_subdomain);
  Int32UniqueArray2 new_nodes_lids(nb_subdomain,nb_new_nodes_per_subdomain);
  _addNodes(new_nodes_lids,new_nodes_uids);

  // put in addNodes
  node_family->endUpdate();
  node_family->computeSynchronizeInfos();

  // Follow mesh changes in connectivity
  // Add and remove in FromFamily
  IItemConnectivityMng* connectivity_mng = dofMng().connectivityMng();
  if (! connectivity_mng->isUpToDate(&node2dof))
    {
      // Handle added nodes : create a dof for each own node added
      Int32ArrayView source_family_added_items_lids;
      Int32ArrayView source_family_removed_items_lids;
      connectivity_mng->getSourceFamilyModifiedItems(&node2dof,source_family_added_items_lids,source_family_removed_items_lids);
      ItemVector source_family_added_items_own(node_family);
      ENUMERATE_NODE(inode,node_family->view(source_family_added_items_lids)) if (inode->isOwn()) source_family_added_items_own.add(inode.localId());
      Int32ConstArrayView source_family_added_items_own_lids = source_family_added_items_own.viewAsArray();
      // Create new dofs on these new nodes : on the owned node only
      Int64UniqueArray uids(source_family_added_items_own.size());
      Integer i = 0;
      ENUMERATE_NODE(inode,source_family_added_items_own) {uids[i++] = mesh::DoFUids::uid(inode->uniqueId().asInt64());}
      Int32SharedArray lids(uids.size());
      dof_on_node_family->addDoFs(uids,lids);
      dof_on_node_family->endUpdate();
      // Update connectivity
      node2dof.updateConnectivity(source_family_added_items_own_lids,lids);
      // Update ghost
      synchronizer->synchronize();
      // For test purpose only : try getSourceFamilyModifiedItem (must give back the new dofs created)
      Int32ArrayView target_family_added_item_lids, target_family_removed_item_lids;
      connectivity_mng->getTargetFamilyModifiedItems(&node2dof,target_family_added_item_lids,target_family_removed_item_lids);
      _checkTargetFamilyInfo(dof_on_node_family->itemFamily()->view(target_family_added_item_lids),lids);
      // Finalize connectivity update
      connectivity_mng->setUpToDate(&node2dof);
    }

  if (! connectivity_mng->isUpToDate(&node2dof)) fatal() << "Error in connectivity update tracking.";

  // Check Connectivity update
  _checkConnectivityUpdateAfterAdd(node2dof, new_nodes_lids, new_nodes_uids,IntegerSharedArray(nb_new_nodes_per_subdomain,1),true);

  // Remove a first node
  Integer nb_removed_nodes = 1;
  Int32SharedArray2 removed_node_lids(nb_subdomain,nb_removed_nodes);
  Int64SharedArray2 removed_node_uids(nb_subdomain,nb_removed_nodes);
  Integer nb_remaining_nodes = new_nodes_lids.dim2Size()-nb_removed_nodes;
  Int32SharedArray2 remaining_node_lids(nb_subdomain,nb_remaining_nodes);
  Int64SharedArray2 remaining_node_uids(nb_subdomain,nb_remaining_nodes);
  _removeNodes(new_nodes_lids,nb_removed_nodes,removed_node_lids,removed_node_uids,remaining_node_lids,remaining_node_uids);
  node_family->endUpdate();
  node_family->computeSynchronizeInfos();

  if(!connectivity_mng->isUpToDate(&node2dof))
     {
       Int32ArrayView source_family_added_items_lids;
       Int32ArrayView source_family_removed_items_lids;
       connectivity_mng->getSourceFamilyModifiedItems(&node2dof,source_family_added_items_lids,source_family_removed_items_lids);
       // Get dof connected to removed nodes, to remove them
       Int32UniqueArray removed_dofs(nb_removed_nodes);
       ItemInternal internal;
       for (Integer i = 0; i < nb_removed_nodes; ++i)
         {
           internal.setLocalId(source_family_removed_items_lids[i]);
           Node node(&internal);
           removed_dofs[i] = node2dof(node).localId();
         }
       // Update connectivity : removed nodes are no longer connected
       Int32SharedArray null_item_lids(source_family_removed_items_lids.size(),NULL_ITEM_LOCAL_ID);
       node2dof.updateConnectivity(source_family_removed_items_lids,null_item_lids);
       // unused dof can be removed if desired.
       connectivity_mng->setUpToDate(&node2dof);
       // remove unused dof
       debug() << "REMOVED DOF " << removed_dofs;
       dof_on_node_family->removeDoFs(removed_dofs);
       dof_on_node_family->endUpdate();
  }

  // Check Connectivity update
  _checkConnectivityUpdateAfterRemove(node2dof, removed_node_lids, removed_node_uids);

  // Compact Source family
  if (m_do_compact) {
    debug() << "NODES " << node_family->view().localIds();
    node_family->compactItems(true);
    debug() << "NODES " << node_family->view().localIds();
    // update lids
    for (Arcane::Integer rank = 0; rank < nb_subdomain; ++rank) {
      node_family->itemsUniqueIdToLocalId(remaining_node_lids[rank], remaining_node_uids[rank], true);
    }
  }

  // Compact Target family
  if (m_do_compact) {
    debug() << "DOFS " << dof_on_node_family->itemFamily()->view().localIds();
    debug() << "DOF family size " << dof_on_node_family->nbItem();
    dof_on_node_family->itemFamily()->compactItems(true);
    debug() << "DOF family size " << dof_on_node_family->nbItem();
    debug() << "DOFS " << dof_on_node_family->itemFamily()->view().localIds();

    _checkConnectivityUpdateAfterCompact(node2dof, remaining_node_lids, remaining_node_uids, node2dof.itemProperty().size());
  }

  // Remove a second node
  // update node lids
  new_nodes_lids = remaining_node_lids;
  info() << "New node lids " << new_nodes_lids[0];
  nb_remaining_nodes = new_nodes_lids.dim2Size() - (nb_removed_nodes + 1);
  remaining_node_lids.resize(nb_subdomain, nb_remaining_nodes);
  remaining_node_uids.resize(nb_subdomain, nb_remaining_nodes);
  _removeNodes(new_nodes_lids, nb_removed_nodes, removed_node_lids, removed_node_uids, remaining_node_lids, remaining_node_uids);
  node_family->endUpdate();
  node_family->computeSynchronizeInfos();

  // Check if compaction works when add&remove
  new_nodes_lids.resize(nb_subdomain, nb_new_nodes_per_subdomain);
  _addNodes(new_nodes_lids, new_nodes_uids);
  node_family->endUpdate(); // Connectivity and ghosts are updated (since own and ghost dof are removed)
  node_family->computeSynchronizeInfos(); // Not needed by connectivity but needed to have NodeFamily synchronization info up to date
  _removeNodes(new_nodes_lids, nb_removed_nodes, removed_node_lids, removed_node_uids, remaining_node_lids, remaining_node_uids);
  node_family->endUpdate(); // Connectivity and ghosts are updated (since own and ghost dof are removed)
  node_family->computeSynchronizeInfos(); // Not needed by connectivity but needed to have NodeFamily synchronization info up to date

  if (m_do_compact)
    node_family->compactItems(true);

  dofMng().connectivityMng()->unregisterConnectivity(&node2dof);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void
DoFTester::
_node2DoFsConnectivityRegistered()
{
  info() << "====================================================";
  info() << "== REGISTER A CONNECTIVITY (ARRAY) AND FOLLOW MESH EVOLUTION ";
  info() << "====================================================";

  // Done for node since it's easier to add new items
  IDoFFamily* dofs_on_node_family = dofMng().getFamily(m_dofs_on_node_family_name);
  Integer nb_dof_per_node = 3;
  // Create the DoFs
  Int64UniqueArray uids(ownNodes().size()*nb_dof_per_node);
  Int64 max_node_uid = mesh::DoFUids::getMaxItemUid(mesh()->nodeFamily());
  Int64 max_dof_uid  = mesh::DoFUids::getMaxItemUid(dofs_on_node_family->itemFamily());
  Integer j = 0;
  ENUMERATE_NODE(inode,ownNodes())
  {
    for (Integer i = 0; i < nb_dof_per_node; ++i) uids[j++] = mesh::DoFUids::uid(max_dof_uid,max_node_uid,inode->uniqueId().asInt64(),i);
  }
  Int32UniqueArray lids(uids.size());
  dofs_on_node_family->addDoFs(uids,lids);
  dofs_on_node_family->endUpdate();

  IItemFamily* node_family = mesh()->nodeFamily();

  NodeToDoFsConnectivity node2dofs(node_family,dofs_on_node_family->itemFamily(),nb_dof_per_node,"NodeToDoFs");

  // Create the ghosts
  GhostLayerFromConnectivityComputer ghost_builder(&node2dofs);
  IItemConnectivitySynchronizer* synchronizer = dofMng().connectivityMng()->createSynchronizer(&node2dofs,&ghost_builder);
  synchronizer->synchronize();

  // Save your connectivity
  dofMng().connectivityMng()->registerConnectivity(&node2dofs);

  // Local mesh change: add own and ghost nodes
  Integer nb_subdomain = subDomain()->parallelMng()->commSize();
  Integer nb_new_nodes = 2;
  Int64UniqueArray2 new_nodes_uids(nb_subdomain,nb_new_nodes);
  Int32UniqueArray2 new_nodes_lids(nb_subdomain,nb_new_nodes);
  _addNodes(new_nodes_lids,new_nodes_uids);

  node_family->endUpdate(); // Connectivity is updated in this call
  node_family->computeSynchronizeInfos(); // Connectivity ghosts are updated in this call

  // Follow mesh changes in connectivity
  // Add and remove in FromFamily
  IItemConnectivityMng* connectivity_mng = dofMng().connectivityMng();
  if (! connectivity_mng->isUpToDate(&node2dofs))
    {
      // Handle added nodes : create nb_dof_per_node dofs for each own node added
      Int32ArrayView source_family_added_items_lids;
      Int32ArrayView source_family_removed_items_lids;
      connectivity_mng->getSourceFamilyModifiedItems(&node2dofs,source_family_added_items_lids,source_family_removed_items_lids);
      ItemVector source_family_added_items_own(node_family);
      ENUMERATE_NODE(inode,node_family->view(source_family_added_items_lids)) if (inode->isOwn()) source_family_added_items_own.add(inode.localId());
      // Create new dofs on these new nodes : on the owned node only
      Int64UniqueArray uids(source_family_added_items_own.size()*nb_dof_per_node);
      Integer j = 0;
      Int32SharedArray source_family_lids_in_connectivity(source_family_added_items_own.size()*nb_dof_per_node);
      Int64 max_item_uid= mesh::DoFUids::getMaxItemUid(node_family);
      Int64 max_dof_uid = mesh::DoFUids::getMaxItemUid(dofs_on_node_family->itemFamily());
      ENUMERATE_NODE(inode,source_family_added_items_own)
      {
        for (Integer i = 0; i < nb_dof_per_node; ++i)
          {
            uids[j] = mesh::DoFUids::uid(max_dof_uid,max_item_uid,inode->uniqueId().asInt64(),i);
            source_family_lids_in_connectivity[j++] = inode.localId(); // Replicate the from item lid each time it is used in a connectivity (needed to use updateConnectivity)
          }
      }
      Int32SharedArray lids(uids.size());
      dofs_on_node_family->addDoFs(uids,lids);
      dofs_on_node_family->endUpdate();
      // Update connectivity
      node2dofs.updateConnectivity(source_family_lids_in_connectivity,lids);
      // Update ghost
      synchronizer->synchronize();
      connectivity_mng->setUpToDate(&node2dofs);
    }

  // Check Connectivity update
  _checkConnectivityUpdateAfterAdd(node2dofs, new_nodes_lids, new_nodes_uids,IntegerSharedArray(nb_new_nodes,nb_dof_per_node),false);


  // Remove the added nodes
  Integer nb_removed_nodes = 1;
  Int32SharedArray2 removed_nodes_lids(nb_subdomain,nb_removed_nodes);
  Int64SharedArray2 removed_nodes_uids(nb_subdomain,nb_removed_nodes);
  Integer nb_remaining_nodes = new_nodes_lids.dim2Size()-nb_removed_nodes;
  Int32SharedArray2 remaining_nodes_lids(nb_subdomain,nb_remaining_nodes);
  Int64SharedArray2 remaining_nodes_uids(nb_subdomain,nb_remaining_nodes);
  _removeNodes(new_nodes_lids,nb_removed_nodes,removed_nodes_lids,removed_nodes_uids,remaining_nodes_lids,remaining_nodes_uids);
  node_family->endUpdate(); // Connectivity and ghosts are updated (since own and ghost dof are removed)
  node_family->computeSynchronizeInfos(); // Not needed by connectivity but needed to have NodeFamily synchronization info up to date

  // Update connectivity : set the removed nodes to Null item lid
  if(!connectivity_mng->isUpToDate(&node2dofs))
    {
      Int32ArrayView source_family_added_items_lids;
      Int32ArrayView source_family_removed_items_lids;
      connectivity_mng->getSourceFamilyModifiedItems(&node2dofs,source_family_added_items_lids,source_family_removed_items_lids);
      // Get dof connected to removed nodes, to remove them (used to test dof family compaction)
      Integer nb_removed_dofs = nb_removed_nodes*nb_dof_per_node;
      Int32UniqueArray removed_dofs;
      removed_dofs.reserve(nb_removed_dofs);
      ItemInternal internal;
      ConnectivityItemVector con(node2dofs);
      for (Integer i = 0; i < nb_removed_nodes; ++i)
        {
          internal.setLocalId(source_family_removed_items_lids[i]);
          Node node(&internal);
          removed_dofs.addRange(node2dofs(node,con).localIds());
        }
      // Prepare data to update connectivity
      Integer nb_connections = source_family_removed_items_lids.size()*nb_dof_per_node;
      Int32SharedArray source_family_removed_items_lids_in_connectivity(nb_connections);
      for (Integer i = 0; i < source_family_removed_items_lids.size(); ++i)
        {
          for(Integer j = 0; j< nb_dof_per_node;++j) source_family_removed_items_lids_in_connectivity[i*nb_dof_per_node+j] = source_family_removed_items_lids[i];
        }
      Int32SharedArray null_item_lids(nb_connections,NULL_ITEM_LOCAL_ID);
      // Update connectivity
      node2dofs.updateConnectivity(source_family_removed_items_lids_in_connectivity,null_item_lids);
      connectivity_mng->setUpToDate(&node2dofs);
      // Unused dof can be removed if desired. Usefull to remove them to test compaction
      dofs_on_node_family->removeDoFs(removed_dofs);
      dofs_on_node_family->endUpdate();
      debug() << "*** REMOVED DOFS " << removed_dofs;
  }

  // Check Connectivity update
  _checkConnectivityUpdateAfterRemove(node2dofs, removed_nodes_lids, removed_nodes_uids, false);

  // Compact Source family
  if (m_do_compact) {
    node_family->compactItems(true);
    debug() << "NODES " << Int32UniqueArray(node_family->view().localIds());
    // update lids
    for (Arcane::Integer rank = 0; rank < nb_subdomain; ++rank) {
      node_family->itemsUniqueIdToLocalId(remaining_nodes_lids[rank], remaining_nodes_uids[rank], true);
    }

  // Compact Target family
    debug() << "DOFS " << dofs_on_node_family->itemFamily()->view().localIds();
    debug() << "DOF family size " << dofs_on_node_family->nbItem();
    dofs_on_node_family->itemFamily()->compactItems(true);
    debug() << "DOF family size " << dofs_on_node_family->nbItem();
    debug() << "DOFS " << dofs_on_node_family->itemFamily()->view().localIds();
    _checkConnectivityUpdateAfterCompact(node2dofs, remaining_nodes_lids, remaining_nodes_uids, node2dofs.itemProperty().dim1Size(), false);
  }

  // Check if compaction works when add&remove
  _addNodes(new_nodes_lids, new_nodes_uids);
  node_family->endUpdate(); // Connectivity and ghosts are updated (since own and ghost dof are removed)
  node_family->computeSynchronizeInfos(); // Not needed by connectivity but needed to have NodeFamily synchronization info up to date
  _removeNodes(new_nodes_lids, nb_removed_nodes, removed_nodes_lids, removed_nodes_uids, remaining_nodes_lids, remaining_nodes_uids);
  node_family->endUpdate(); // Connectivity and ghosts are updated (since own and ghost dof are removed)
  node_family->computeSynchronizeInfos(); // Not needed by connectivity but needed to have NodeFamily synchronization info up to date

  if (m_do_compact) node_family->compactItems(true);

  dofMng().connectivityMng()->unregisterConnectivity(&node2dofs);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void
DoFTester::
_node2DoFsMultiConnectivityRegistered()
{
  info() << "====================================================";
  info() << "== REGISTER A CONNECTIVITY (MULTI ARRAY) AND FOLLOW MESH EVOLUTION ";
  info() << "====================================================";
  // Done for node since it's easier to add new items
  IDoFFamily* dofs_multi_on_node_family = dofMng().getFamily(m_dofs_multi_on_node_family_name);
  IItemFamily* node_family = mesh()->nodeFamily();
  IntegerUniqueArray nb_dof_per_node(node_family->maxLocalId(),0);
  // Create the DoFs
  Int64UniqueArray uids;
  Int64 max_node_uid = mesh::DoFUids::getMaxItemUid(mesh()->nodeFamily());
  Int64 max_dof_uid =  mesh::DoFUids::getMaxItemUid(dofs_multi_on_node_family->itemFamily());
  ENUMERATE_NODE(inode,ownNodes())
  {
    nb_dof_per_node[inode->localId()] = 2; // constant size in initialization
    for (Integer i = 0; i < nb_dof_per_node[inode->localId()]; ++i) uids.add(mesh::DoFUids::uid(max_dof_uid,max_node_uid,inode->uniqueId().asInt64(),i));
  }
  Int32UniqueArray lids(uids.size());
  dofs_multi_on_node_family->addDoFs(uids,lids);
  dofs_multi_on_node_family->endUpdate();

  NodeToDoFsMultiConnectivity node2dofs(node_family,dofs_multi_on_node_family->itemFamily(),nb_dof_per_node,"NodeToDoFsMulti");

  // Create the ghosts
  GhostLayerFromConnectivityComputer ghost_builder(&node2dofs);
  IItemConnectivitySynchronizer* synchronizer = dofMng().connectivityMng()->createSynchronizer(&node2dofs,&ghost_builder);
  synchronizer->synchronize();

  // Save your connectivity
  dofMng().connectivityMng()->registerConnectivity(&node2dofs);

  // Local mesh change: add own and ghost nodes
  Integer nb_subdomain = subDomain()->parallelMng()->commSize();
  Integer nb_new_nodes = 2;
  Int64UniqueArray2 new_nodes_uids(nb_subdomain,nb_new_nodes);
  Int32UniqueArray2 new_nodes_lids(nb_subdomain,nb_new_nodes);
  _addNodes(new_nodes_lids,new_nodes_uids);

  node_family->endUpdate(); // Connectivity is updated in this call
  node_family->computeSynchronizeInfos(); // Connectivity ghosts are updated in this call

  // Follow mesh changes in connectivity.
  IItemConnectivityMng* connectivity_mng = dofMng().connectivityMng();
  IntegerUniqueArray nb_dof_per_new_node;

  if (! connectivity_mng->isUpToDate(&node2dofs))
    {
      // Handle added nodes : create a variable number of dofs for each own node added
      Int32ArrayView source_family_added_items_lids;
      Int32ArrayView source_family_removed_items_lids;
      connectivity_mng->getSourceFamilyModifiedItems(&node2dofs,source_family_added_items_lids,source_family_removed_items_lids);
      ItemVector source_family_added_items_own(node_family);
      ENUMERATE_NODE(inode,node_family->view(source_family_added_items_lids)) if (inode->isOwn()) source_family_added_items_own.add(inode.localId());
      IntegerUniqueArray source_family_added_items_own_lids(source_family_added_items_own.viewAsArray());
      nb_dof_per_new_node.resize(source_family_added_items_own_lids.size());
      Integer nb_new_dofs = 0;
      for (Arcane::Integer i = 0; i < nb_dof_per_new_node.size(); ++i)
        {
          nb_dof_per_new_node[i] = i+1;
          nb_new_dofs += nb_dof_per_new_node[i];
        }
      Integer nb_connections  = nb_new_dofs;
      // Create new dofs on these new nodes : on the owned node only
      Int64UniqueArray uids(nb_connections);
      Integer j = 0;
      Int32SharedArray source_family_lids_in_connectivity(nb_connections);
      Int64 max_item_uid= mesh::DoFUids::getMaxItemUid(node_family);
      Int64 max_dof_uid = mesh::DoFUids::getMaxItemUid(dofs_multi_on_node_family->itemFamily());
      ENUMERATE_NODE(inode,source_family_added_items_own)
      {
        for (Integer i = 0; i < nb_dof_per_new_node[inode.index()]; ++i)
          {
            uids[j] = mesh::DoFUids::uid(max_dof_uid,max_item_uid,inode->uniqueId().asInt64(),i);
            source_family_lids_in_connectivity[j++] = inode.localId(); // Replicate the from item lid each time it used in a connectivity (needed to use updateConnectivity)
          }
      }
      Int32SharedArray lids(uids.size());
      dofs_multi_on_node_family->addDoFs(uids,lids);
      dofs_multi_on_node_family->endUpdate();
      // Update connectivity
      node2dofs.updateConnectivity(source_family_lids_in_connectivity,lids);
      // Update ghost
      synchronizer->synchronize();
      connectivity_mng->setUpToDate(&node2dofs);
    }

  // Check Connectivity update
  _checkConnectivityUpdateAfterAdd(node2dofs, new_nodes_lids, new_nodes_uids,nb_dof_per_new_node,false);

  debug() << "NEW NODE LIDS " << new_nodes_lids[mesh()->parallelMng()->commRank()];

  // Remove the added nodes
  Integer nb_removed_nodes = 1;
  Int32SharedArray2 removed_nodes_lids(nb_subdomain,nb_removed_nodes);
  Int64SharedArray2 removed_nodes_uids(nb_subdomain,nb_removed_nodes);
  Integer nb_remaining_nodes = new_nodes_lids.dim2Size()-nb_removed_nodes;
  Int32SharedArray2 remaining_nodes_lids(nb_subdomain,nb_remaining_nodes);
  Int64SharedArray2 remaining_nodes_uids(nb_subdomain,nb_remaining_nodes);
  _removeNodes(new_nodes_lids, nb_removed_nodes,removed_nodes_lids,removed_nodes_uids, remaining_nodes_lids,remaining_nodes_uids);
  debug() << "REMOVED NODES " << removed_nodes_lids[mesh()->parallelMng()->commRank()] << removed_nodes_uids[mesh()->parallelMng()->commRank()];

  node_family->endUpdate(); // Connectivity and ghosts are updated (since own and ghost dof are removed)
  node_family->computeSynchronizeInfos(); // Not needed by connectivity but needed to have NodeFamily synchronization info up to date

  debug() << "NODES " << Int32UniqueArray(node_family->view().localIds());

  // Update connectivity : set the removed nodes to Null item lid
  if(!connectivity_mng->isUpToDate(&node2dofs))
    {
      Int32ArrayView source_family_added_items_lids;
      Int32ArrayView source_family_removed_items_lids;
      connectivity_mng->getSourceFamilyModifiedItems(&node2dofs,source_family_added_items_lids,source_family_removed_items_lids);
      // Prepare data to update connectivity
      Int32SharedArray source_family_removed_items_lids_in_connectivity;
      nb_dof_per_new_node = node2dofs.itemProperty().dim2Sizes();
      Integer nb_removed_dofs = 0;
      for (Integer i = 0; i < source_family_removed_items_lids.size(); ++i)
        {
          Int32 removed_item_lid = source_family_removed_items_lids[i];
          for(Integer j = 0; j< nb_dof_per_new_node[removed_item_lid];++j)
            {
              source_family_removed_items_lids_in_connectivity.add(source_family_removed_items_lids[i]);
              nb_removed_dofs++;
            }
        }
      // Get dof connected to removed nodes, to remove them (used to test dof family compaction)
      Int32UniqueArray removed_dofs;
      removed_dofs.reserve(nb_removed_dofs);
      ItemInternal internal;
      ConnectivityItemVector con(node2dofs);
      for (Integer i = 0; i < nb_removed_nodes; ++i)
        {
          internal.setLocalId(source_family_removed_items_lids[i]);
          Node node(&internal);
          removed_dofs.addRange(node2dofs(node,con).localIds());
        }
      Int32SharedArray null_item_lids(source_family_removed_items_lids_in_connectivity.size(),NULL_ITEM_LOCAL_ID);
      // Update connectivity
      node2dofs.updateConnectivity(source_family_removed_items_lids_in_connectivity,null_item_lids);
      // unused dof can be removed if desired. Usefull to remove them to test compaction
      dofs_multi_on_node_family->removeDoFs(removed_dofs);
      dofs_multi_on_node_family->endUpdate();
      connectivity_mng->setUpToDate(&node2dofs);
    }

  // Check Connectivity update
  _checkConnectivityUpdateAfterRemove(node2dofs, removed_nodes_lids, removed_nodes_uids, false);

  // Compact Source family
  if (m_do_compact) {
    node_family->compactItems(true);
    debug() << "NODES " << node_family->view().localIds();
    // update lids
    for (Arcane::Integer rank = 0; rank < nb_subdomain; ++rank)
    {
      node_family->itemsUniqueIdToLocalId(remaining_nodes_lids[rank],remaining_nodes_uids[rank],true);
    }

    // Compact Target family
    debug() << "DOFS " << dofs_multi_on_node_family->itemFamily()->view().localIds();
    debug() << "DOF family size " << dofs_multi_on_node_family->nbItem();
    dofs_multi_on_node_family->itemFamily()->compactItems(true);
    debug() << "DOF family size " << dofs_multi_on_node_family->nbItem();
    debug() << "DOFS " << dofs_multi_on_node_family->itemFamily()->view().localIds();

    _checkConnectivityUpdateAfterCompact(node2dofs, remaining_nodes_lids, remaining_nodes_uids, node2dofs.itemProperty().dim1Size(), false);
  }

  // Check if compaction works when add&remove
  _addNodes(new_nodes_lids, new_nodes_uids);
  node_family->endUpdate(); // Connectivity and ghosts are updated (since own and ghost dof are removed)
  node_family->computeSynchronizeInfos(); // Not needed by connectivity but needed to have NodeFamily synchronization info up to date
  _removeNodes(new_nodes_lids, nb_removed_nodes, removed_nodes_lids, removed_nodes_uids, remaining_nodes_lids, remaining_nodes_uids);
  node_family->endUpdate(); // Connectivity and ghosts are updated (since own and ghost dof are removed)
  node_family->computeSynchronizeInfos(); // Not needed by connectivity but needed to have NodeFamily synchronization info up to date

  if (m_do_compact) node_family->compactItems(true);

  dofMng().connectivityMng()->unregisterConnectivity(&node2dofs);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DoFTester::
_removeGhost(IDoFFamily* dof_family)
{
  Int32SharedArray removed_items;
  ENUMERATE_DOF(idof,dof_family->allItems().ghost()) {
    removed_items.add(idof->localId());
  }
  dof_family->removeDoFs(removed_items);
  dof_family->endUpdate();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void
DoFTester::
_addNodes(Int32Array2View new_nodes_lids, Int64Array2View new_nodes_uids)
{
  // Change mesh by removing and adding nodes
  Integer nb_new_nodes = new_nodes_lids.dim2Size();
  Integer nb_subdomain = subDomain()->parallelMng()->commSize();
  Integer local_rank   = subDomain()->parallelMng()->commRank();
  Int64 max_uid = mesh::DoFUids::getMaxItemUid(mesh()->nodeFamily());
  UniqueArray<ItemVectorView> added_items(nb_subdomain);
  // Add node and ghost nodes on each subdomain. Each subdomain has all the nodes of the other as ghosts (just for the demo)
  for (Integer rank = 0; rank < nb_subdomain; ++rank)
  {
    for (Integer i = 0; i < nb_new_nodes;++i) new_nodes_uids[rank][i] = max_uid*(rank+1) +i+1;
    IMeshModifier* mesh_modifier = mesh()->modifier();
    ARCANE_CHECK_POINTER(mesh_modifier);
      mesh_modifier->addNodes(new_nodes_uids[rank], new_nodes_lids[rank]);
    added_items[rank] = mesh()->nodeFamily()->view(new_nodes_lids[rank]);
    ENUMERATE_NODE(inode,added_items[rank])
    {
      inode->mutableItemBase().setOwner(rank,local_rank);
      info() << "== Add item " << inode->localId() << " on rank " << local_rank << " with owner " << rank;
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void
DoFTester::
_checkConnectivityUpdateAfterAdd(IItemConnectivity& node2dof, Int32Array2View new_nodes_lids, Int64ConstArray2View new_nodes_uids, IntegerConstArrayView nb_dof_per_item, bool is_scalar_connectivity)
{
  // Your connectivity is up to date : i.e. new cells must be connected to a dof:
  Integer nb_subdomain = subDomain()->parallelMng()->commSize();
  UniqueArray<ItemVectorView> added_items(nb_subdomain);
  for (Integer rank = 0; rank < nb_subdomain; ++rank){
    // local ids may have change in endUpdate() so we recompute them
    mesh()->nodeFamily()->itemsUniqueIdToLocalId(new_nodes_lids[rank],new_nodes_uids[rank],false);
    added_items[rank] = mesh()->nodeFamily()->view(new_nodes_lids[rank]);
    ConnectivityItemVector node2dof_vector(node2dof);
    ENUMERATE_NODE(inode,added_items[rank]){
      ItemVectorView dofs = node2dof_vector.connectedItems(inode);
      if (inode->isOwn())
        info() << "== New node with uid " << inode->uniqueId().asInt64();
      else
        info() << "== New ghost node with uid " << inode->uniqueId().asInt64();
      for (Integer i = 0; i < nb_dof_per_item[inode.index()]; ++i){
        info() << " connected with dof (uid " << dofs[i].uniqueId().asInt64() << " ).";
      }
    }
  }
  _printNodeToDoFConnectivity(node2dof,is_scalar_connectivity,true);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void
DoFTester::
_checkConnectivityUpdateAfterRemove(IItemConnectivity& node2dof, Int32Array2View new_nodes_lids,
                                    Int64ConstArray2View new_nodes_uids, bool is_scalar_connectivity)
{
  ItemInternal internal; // for test
  Integer nb_subdomain = subDomain()->parallelMng()->commSize();
  Integer nb_new_nodes = new_nodes_lids.dim2Size();
  ConnectivityItemVector node2dof_vector(node2dof);
  for (Integer rank = 0; rank < nb_subdomain; ++rank ){
    for (Integer i = 0; i < nb_new_nodes; ++i ){
      internal.setLocalId(new_nodes_lids[rank][i]);
      Node removed_node(&internal);
      if (is_scalar_connectivity){
        NodeToDoFConnectivity& concrete_node2dof = static_cast<NodeToDoFConnectivity&>(node2dof);
        info() << "== Connectivity value for removed node (uid) " << new_nodes_uids[rank][i] << " lid("<<new_nodes_lids[rank][i] << ")"
               << " = " << concrete_node2dof.itemProperty()[removed_node];
        if (concrete_node2dof.itemProperty()[removed_node] != NULL_ITEM_LOCAL_ID)
          fatal() << "Error in update connectivity after remove";
      }
      else{
        ItemVectorView dofs = node2dof_vector.connectedItems(removed_node);
        Int32ConstArrayView dofs_ids = dofs.localIds();
        info() << "== Connectivity value for removed node (uid) " << new_nodes_uids[rank][i]  << " lid("<<new_nodes_lids[rank][i] << ")"
               << " = " << dofs_ids;
        for (Integer j = 0 ; j < dofs.size(); ++ j)
          if (dofs_ids[j] != NULL_ITEM_LOCAL_ID)
            fatal() << "Error in update connectivity after remove";
      }
    }
  }
  _printNodeToDoFConnectivity(node2dof,is_scalar_connectivity,true);
}

void
DoFTester::
_checkConnectivityUpdateAfterCompact(IItemConnectivity& node2dof, Int32Array2View remaining_nodes_lids,
                                     Int64ConstArray2View remaining_nodes_uids, Integer item_property_size, bool is_scalar_connectivity)
{
  // Check target family compaction : remaining nodes must be associated to at least one dof
  ItemInternal internal; // for test
  Integer nb_subdomain = subDomain()->parallelMng()->commSize();
  Integer nb_remaining_nodes = remaining_nodes_lids.dim2Size();
  IItemFamily* dof_family = node2dof.targetFamily();
  ConnectivityItemVector node2dof_vector(node2dof);
  for (Integer rank = 0; rank < nb_subdomain; ++rank ){
    for (Integer i = 0; i < nb_remaining_nodes; ++i ){
      internal.setLocalId(remaining_nodes_lids[rank][i]);
      Node remaining_node(&internal);
      if (is_scalar_connectivity){
        NodeToDoFConnectivity& concrete_node2dof = static_cast<NodeToDoFConnectivity&>(node2dof);
        Int32 connected_dof_lids = concrete_node2dof.itemProperty()[remaining_node];
        info() << "== Connectivity value for node (uid) " << remaining_nodes_uids[rank][i] << " (lid=  " << remaining_nodes_lids[rank][i] << ")"
               << " = " << connected_dof_lids << " with nb item = " << dof_family->nbItem();
        if (connected_dof_lids == NULL_ITEM_LOCAL_ID || connected_dof_lids+1 > dof_family->nbItem())
          fatal() << "Error in check connectivity after compact";
      }
      else{
        ItemVectorView dofs = node2dof_vector.connectedItems(remaining_node);
        Int32ConstArrayView dofs_ids = dofs.localIds();
        info() << "== Connectivity value for node (uid) " << remaining_nodes_uids[rank][i] << " (lid= " << remaining_nodes_lids[rank][i] << ")"
               << " = " << dofs_ids << " with nb item = " << dof_family->nbItem();
        for (Integer j = 0 ; j < dofs.size(); ++ j)
          if (dofs_ids[j] == NULL_ITEM_LOCAL_ID || dofs_ids[j]+1 > dof_family->nbItem())
            fatal() << "Error in check connectivity after compact";
      }
    }
  }
  _printNodeToDoFConnectivity(node2dof, is_scalar_connectivity,true);
  // Check source family compaction : itemProperty must have family size
  if (item_property_size != node2dof.sourceFamily()->nbItem()) fatal() << "Error : connectivity is not correctly impacted by change in source family";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void
DoFTester::
_removeNodes(Int32ConstArray2View new_nodes_lids,
             const Integer nb_removed_nodes,
             Int32SharedArray2& removed_node_lids,
             Int64SharedArray2& removed_node_uids,
             Int32SharedArray2& remaining_node_lids,
             Int64SharedArray2& remaining_node_uids)
{
  ARCANE_ASSERT((nb_removed_nodes <= new_nodes_lids.dim2Size()),("Cannot removed more nodes than available"));
  auto* node_family = mesh()->nodeFamily();
  Integer nb_subdomain = subDomain()->parallelMng()->commSize();
  Integer nb_remaining_nodes = remaining_node_lids.dim2Size();
  UniqueArray<ItemVectorView> removed_nodes(nb_subdomain);
  UniqueArray<ItemVectorView> remaining_nodes(nb_subdomain);
  for (Integer rank = 0; rank < nb_subdomain; ++rank)
  {
    removed_node_lids[rank].copy(new_nodes_lids[rank].subConstView(0,nb_removed_nodes));
    info() << "== Remove nodes " << removed_node_lids[rank] << " on rank " << rank;
    removed_nodes[rank] = node_family->view(removed_node_lids[rank]);
    remaining_node_lids[rank].copy(new_nodes_lids[rank].subConstView(nb_removed_nodes,nb_remaining_nodes));
    info() << "== Remaining nodes " << remaining_node_lids[rank] << " on rank " << rank;
    remaining_nodes[rank] = node_family->view(remaining_node_lids[rank]);
    Int32 i = 0;
    mesh::NodeFamily* node_family_internal = dynamic_cast<mesh::NodeFamily*>(node_family);
    if (node_family_internal) {
      ENUMERATE_NODE (inode, removed_nodes[rank]) {
        removed_node_uids[rank][i++] = inode->uniqueId().asInt64();
        info() << "== Remove node " << inode->localId() << " on rank " << mesh()->parallelMng()->commRank() << " with owner " << rank;
        node_family_internal->removeNodeIfNotConnected(*inode);
      }
    }
    else // no NodeFamily in PolyhedralMesh
    {
      IPolyhedralMeshModifier* polyhedral_modifier = mesh()->_internalApi()->polyhedralMeshModifier();
      info() << "POLYHEDRAL CASE";
      ENUMERATE_NODE(inode,removed_nodes[rank])
      {
        removed_node_uids[rank][i++] = inode->uniqueId().asInt64();
        info() << "== Remove node " << inode->localId() << " on rank " << mesh()->parallelMng()->commRank() << " with owner " << rank;
      }
      polyhedral_modifier->removeItems(removed_node_lids[rank], IK_Node, node_family->name());
    }
    i = 0;
    ENUMERATE_NODE(inode,remaining_nodes[rank])
    {
      remaining_node_uids[rank][i++] = inode->uniqueId().asInt64();
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void
DoFTester::
_checkTargetFamilyInfo(ItemVectorView tracked_new_dofs, Int32ConstArrayView new_dofs_lids)
{
  // The restriction of lids_1 to own item must be equal to lids_2
  Int32SharedArray tracked_new_dofs_lids;
  ENUMERATE_DOF(idof,tracked_new_dofs)
  {
    if(idof->isOwn()) tracked_new_dofs_lids.add(idof.localId());
  }
  bool is_ok = tracked_new_dofs_lids.size() == new_dofs_lids.size();
  if (is_ok)
    {
      for (Integer i = 0; i < tracked_new_dofs_lids.size();++i) is_ok = ((tracked_new_dofs_lids[i] == new_dofs_lids[i]) && is_ok);
    }
  if (!is_ok) fatal() << "Error in target family modification tracking";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void
DoFTester::
doFVariable()
{
  IDoFFamily* dof_family = dofMng().getFamily(m_dofs_on_cell_family_name);

  // Create and fillDoF Scalar Variable (with internal support)
  info() << "=== CreateDoFVariable : ";
  VariableDoFReal dof_variable(VariableBuildInfo(mesh(),"DoFVariable",m_dofs_on_cell_family_name));

  dof_variable.fill(-1);
  ENUMERATE_DOF(idof,dof_family->allItems().own())
  {
    dof_variable[idof] = (Real)(idof->uniqueId().asInt64());
    info() << "= dof_variable[idof] = " << dof_variable[idof];
  }

  bool do_check = false;
  _printVariable(dof_variable, do_check);

  dof_variable.synchronize();

  do_check = true;
  _printVariable(dof_variable,do_check);


  VariableCollection var_collection;
  dof_family->itemFamily()->usedVariables(var_collection);
  info() << "==Used variables in family";
  for(VariableCollection::Enumerator ite = var_collection.enumerator();++ite ; )
    {
      info() << "= Variable " << (*ite)->name();
    }

  // Create and fillDoF Array Variable (with internal support)
  info() << "=== CreateDoFVariable : ";
  VariableDoFArrayReal dof_array_variable(VariableBuildInfo(mesh(),"DoFArrayVariable",m_dofs_on_cell_family_name));

  Integer size =2;
  dof_array_variable.resize(size);
  // Initialize with -1
  ENUMERATE_DOF(idof,dof_family->allItems()) {dof_array_variable[idof].fill(-1);}

  ENUMERATE_DOF(idof,dof_family->allItems().own())
  {
    for (Integer i = 0; i < size; ++i)
      {
        dof_array_variable[idof][i] = (Real)((i+1)*idof->uniqueId().asInt64());
        info() << "= dof_array_variable[idof] = " << dof_array_variable[idof][i];
      }
  }

  do_check = false;
  info() << "Print before synchronization";
  _printArrayVariable(dof_array_variable, do_check);

  dof_array_variable.synchronize();

  info() << "Print after synchronization";
  do_check = true;
  _printArrayVariable(dof_array_variable,do_check);



  // TODO DoF aggregated variable (cf. AnyItem)
//  // Create a DoF variable aggregating a cell and a node variable
//  DoFFamily& node_cell_dof_family = dofMng().family("MyFaceCellDoFFamily");
//  AggregatedVariableDoFInt64 aggregated_dof_variable;
//  aggregated_dof_variable[allCells()] << m_cell_variable;
//  aggregated_dof_variable[allNodes()] << m_node_variable;
//
//  DoFGroup node_cell_dof_group = node_cell_dof_family.allDoFs();
//
//  ENUMERATE_DOF(idof,node_cell_dof_group)
//  {
//    aggregated_dof_variable[idof] = idof.uniqueId();
//  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DoFTester::
_printVariable(VariableDoFReal& dof_variable, bool check)
{
  info() << "===Print dof variable " << dof_variable.name();
  ENUMERATE_DOF(idof,dof_variable.variable()->itemFamily()->allItems().own()){
    info() << String::format("DoF variable [{0}] = {1}",idof->uniqueId().asInt64(),dof_variable[idof]);
  }
  ENUMERATE_DOF(idof,dof_variable.variable()->itemFamily()->allItems().ghost()){
    info() << String::format("DoF variable [{0}] = {1}",idof->uniqueId().asInt64(),dof_variable[idof]);
    if (check){
      if (dof_variable[idof] < 0)
        fatal() << "DoF Variable synchronization failure";
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void
DoFTester::
_printArrayVariable(VariableDoFArrayReal& dof_array_variable, bool check)
{
  info() << "===Print dof array variable " << dof_array_variable.name();
  ENUMERATE_DOF(idof,dof_array_variable.variable()->itemFamily()->allItems().own()){
    for (Integer i = 0; i < dof_array_variable.arraySize(); ++i) {
      info() << String::format("DoF variable [{0}] = {1}",idof->uniqueId().asInt64(),dof_array_variable[idof][i]);
    }
  }
  ENUMERATE_DOF(idof,dof_array_variable.variable()->itemFamily()->allItems().ghost()){
    for (Integer i = 0; i < dof_array_variable.arraySize(); ++i) {
      info() << String::format("DoF variable [{0}] = {1}",idof->uniqueId().asInt64(),dof_array_variable[idof][i]);
    }
    if (check){
      for (Integer i = 0; i < dof_array_variable.arraySize(); ++i)
        if (dof_array_variable[idof][i] < 0)
          fatal() << "DoF Array Variable synchronization failure";
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool
DoFTester::
_checkIsSame(IItemConnectivity* connectivity1, IItemConnectivity* connectivity2)
{
  bool is_ok = true;
  is_ok =  is_ok && (connectivity1->sourceFamily()->name() ==connectivity2->sourceFamily()->name());
  is_ok =  is_ok && (connectivity1->targetFamily()->name() ==connectivity2->targetFamily()->name());
  ENUMERATE_ITEM(item,connectivity1->sourceFamily()->allItems())
  {
    is_ok = is_ok && (connectivity1->nbConnectedItem(item) == connectivity2->nbConnectedItem(item));
    for (Integer i = 0;  i < connectivity1->nbConnectedItem(item); ++ i)
      {
        is_ok = is_ok && (connectivity1->connectedItemLocalId(item,i) == connectivity2->connectedItemLocalId(item,i));
      }
  }
  return is_ok;

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename Connectivity>
void DoFTester::_printNodeToDoFConnectivity(Connectivity& node2dof, bool is_scalar_connectivity, bool do_check)
{
  bool has_error = false;
  if (is_scalar_connectivity) {
    NodeToDoFConnectivity& concrete_node2dof = static_cast<NodeToDoFConnectivity&>(node2dof);
    ENUMERATE_NODE (inode, allNodes()) {
      Int32 connected_dof_lid = concrete_node2dof.itemProperty()[*inode];
      info() << "== Connectivity value for node (uid) " << inode->uniqueId().asInt64() << " (lid=  " << inode.localId() << ")"
             << " = " << connected_dof_lid;
      if (connected_dof_lid != inode.localId()) has_error = true;
    }
  }
  else{
    ConnectivityItemVector node2dof_vector(node2dof);
    ENUMERATE_NODE(inode,allNodes()) {
      ItemVectorView dofs = node2dof_vector.connectedItems(inode->itemLocalId());
      Int32ConstArrayView dofs_ids = dofs.localIds();
      info() << "== Connectivity value for node (uid = " << inode->uniqueId().asInt64() << ") (lid = " << inode.localId() << ")"
             << " = " << dofs_ids;
    }
  }
  // the check is only valid in sequential
  if (subDomain()->parallelMng()->isParallel()) do_check = false;
  if (has_error && do_check)
    ARCANE_FATAL("The connectivity has error. Dof should be connected with dofs with the same local id");
}


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE_DOFTESTER(DoFTester,DoFTester);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
