// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SubMeshTestModule.cc                                        (C) 2000-2024 */
/*                                                                           */
/* Module de test du sous-maillage                                           */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/List.h"

#include "arcane/Directory.h"
#include "arcane/VariableTypes.h"
#include "arcane/ItemPrinter.h"
#include "arcane/IPrimaryMesh.h"
#include "arcane/MeshToMeshTransposer.h"
#include "arcane/IPostProcessorWriter.h"
#include "arcane/Connectivity.h"
#include "arcane/MeshStats.h"
#include "arcane/IMeshModifier.h"
#include "arcane/IMeshUtilities.h"
#include "arcane/IMainFactory.h"
#include "arcane/ITimeLoopMng.h"
#include "arcane/IVariableMng.h"
#include "arcane/SharedVariable.h"
#include "arcane/UnstructuredMeshConnectivity.h"

#include "arcane/tests/ArcaneTestGlobal.h"
#include "arcane/tests/SubMeshTest_axl.h"

#include <set>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace ArcaneTest
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arcane;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Module de test de sous-maillage dans Arcane.
 */
class SubMeshTestModule
: public ArcaneSubMeshTestObject
{
 public:

  SubMeshTestModule(const ModuleBuildInfo& cb);
  ~SubMeshTestModule();

 public:
	
  virtual VersionInfo versionInfo() const { return VersionInfo(0,0,2); }

 public:

  void build();
  void init();
  void startInit();
  void continueInit();
  void compute();

 private:

  VariableCellReal m_cell_real_values;
  VariableFaceReal m_face_real_values;

  void _checkCreateOutputDir();
  void _checkSubMeshIntegrity();
  void _postProcessSubMesh();

  void _compute1CreateMesh();
  void _compute2RemoveItems();
  void _compute3AddItems();
  void _compute4TransposeItems();
  void _compute5MoveItems();

 private:

  Directory m_output_directory; 
  bool m_output_dir_created;

 private:

  // Les variables de traitement du cas test

  // Variables de contrôle
  VariableNodeInt64* node_uids = nullptr;
  bool check_variable = false;
  VariableCellInt64* new_cell_uids = nullptr;
  VariableFaceInt64* new_face_uids = nullptr;
  VariableNodeInt64* new_node_uids = nullptr;

  IMesh* new_mesh = nullptr;

  // Génération d'un sous-maillage du genre demandé
  eItemKind parentKind = IK_Unknown;

  IItemFamily* myParentFamily = nullptr;
  ItemGroup myParentItems;
  ItemGroup allParentItems;
  ItemGroup myNewParentItems;
  ItemGroup myOldParentItems;

  // Post-processing
  RealUniqueArray times; 
  VariableCellReal* new_data = nullptr;

 private:

  void _checkEdgeConnectivity();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_DEFINE_STANDARD_MODULE(SubMeshTestModule,SubMeshTest);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

SubMeshTestModule::
SubMeshTestModule(const ModuleBuildInfo& mb)
: ArcaneSubMeshTestObject(mb)
, m_cell_real_values(VariableBuildInfo(this,"TestParallelCellRealValues"))
, m_face_real_values(VariableBuildInfo(this,"TestParallelFaceRealValues"))
, m_output_dir_created(false)
, node_uids(nullptr)
, check_variable(true)
, new_cell_uids(nullptr)
, new_face_uids(nullptr)
, new_node_uids(nullptr)
, new_data(nullptr)
{
  addEntryPoint(this,"Build",
                &SubMeshTestModule::build,
                IEntryPoint::WBuild,
                IEntryPoint::PAutoLoadBegin);
  addEntryPoint(this,"Init",
                &SubMeshTestModule::init,
                IEntryPoint::WInit);
  addEntryPoint(this,"StartInit",
                &SubMeshTestModule::startInit,
                IEntryPoint::WStartInit);
  addEntryPoint(this,"ContinueInit",
                &SubMeshTestModule::continueInit,
                IEntryPoint::WContinueInit);
  addEntryPoint(this,"compute",
                &SubMeshTestModule::compute);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

SubMeshTestModule::
~SubMeshTestModule()
{
  delete new_data;
  delete node_uids;
  delete new_cell_uids;
  delete new_face_uids;
  delete new_node_uids;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SubMeshTestModule::
_checkCreateOutputDir()
{
  if (m_output_dir_created)
    return;
  m_output_directory = Directory(subDomain()->exportDirectory(),"depouillement2");
  m_output_directory.createDirectory();
  m_output_dir_created = true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SubMeshTestModule::
_checkSubMeshIntegrity()
{
  ENUMERATE_NODE(inode,mesh()->allNodes()) {
    if ((*node_uids)[inode] != inode->uniqueId())
      fatal() << "Bad Global Node variable data : " << ItemPrinter(*inode) << " " << (*node_uids)[inode];
  }

  if (!new_mesh)
    return;

  info() << "TestMesh " << m_global_iteration() << " own/all cell sizes : " << new_mesh->ownCells().size()
         << " " << new_mesh->allCells().size() << " on mesh own/all : "
         << mesh()->ownCells().size() << " / " << mesh()->allCells().size();
  info() << "TestMesh " << m_global_iteration() << " own/all node sizes : " << new_mesh->ownNodes().size()
         << " " << new_mesh->allNodes().size() << " on mesh own/all : "
         << mesh()->ownNodes().size() << " / " << mesh()->allNodes().size();
  info() << "TestMesh " << m_global_iteration() << " parent group own/all sizes "
         << myParentItems.own().size() << " / " << myParentItems.size();

  MeshStats stats(traceMng(),new_mesh,subDomain()->parallelMng());
  stats.dumpStats();

//   ENUMERATE_NODE(inode,new_mesh->allNodes()) {
//     info() << "AllNodes : " << inode.index() << " : " << FullItemPrinter(*inode);
//   }
//   ENUMERATE_FACE(iface,new_mesh->allFaces()) {
//     info() << "AllFaces : " << iface.index() << " : " << FullItemPrinter(*iface);
//   }
//   ENUMERATE_CELL(icell,new_mesh->allCells()) {
//     info() << "AllCells : " << icell.index() << " : " << FullItemPrinter(*icell);
//   }

  if (check_variable) {
    const Integer nerror_max = 10;
    Integer nerror = 0;
    new_node_uids->synchronize();
    new_face_uids->synchronize();
    new_cell_uids->synchronize();

    ENUMERATE_NODE(inode, new_mesh->allNodes())
      if ((*new_node_uids)[inode] != inode->uniqueId())
        if (nerror++ < nerror_max)
          error() << "Node UniqueIds not consistent on item " << ItemPrinter(*inode) << " : " << (*new_node_uids)[inode];
    if (nerror > 0)
      fatal() << "Node UniqueIds not consistent (" << nerror << ")";
    if ((nerror=new_node_uids->checkIfSync(nerror_max))>0)
      fatal() << "Node uniqueIds not synchronized (" << nerror << ")";
    if ((nerror=new_face_uids->checkIfSync(nerror_max))>0)
      fatal() << "Face uniqueIds not synchronized (" << nerror << ")";
    if ((nerror=new_cell_uids->checkIfSync(nerror_max))>0)
      fatal() << "Cell uniqueIds not synchronized (" << nerror << ")";
  }

  SharedVariableNodeInt64 shared_node_uids(new_mesh->nodeFamily(),*node_uids);
  NodeGroup myAllNodes = new_mesh->allNodes();
  ENUMERATE_NODE(inode,myAllNodes) {
    if (shared_node_uids[inode] != inode->uniqueId())
      fatal() << "Bad shared Node variable data : " << ItemPrinter(*inode) << " " << shared_node_uids[inode];
  }

  // Not sync after addItems (step 3)
  std::set<Int64> do_not_check;
  if (m_global_iteration() == 3)
    ENUMERATE_ITEM(iitem,myNewParentItems)
      do_not_check.insert(iitem->uniqueId());

  if (check_variable) {
    Integer nerror2 = 0;
    ENUMERATE_CELL(icell,new_mesh->allCells()) {
      Cell cell = *icell;
      Item parent = cell.parent();
      if (do_not_check.find(cell.uniqueId().asInt64()) != do_not_check.end()) {
        (*new_cell_uids)[cell] = parent.uniqueId(); // mise à jour des non check
      } else {
        if ((*new_cell_uids)[cell] != parent.uniqueId() || cell.uniqueId() != parent.uniqueId()) {
          error() << "Inconsistent sub-mesh uniqueId on item " << ItemPrinter(cell)
                  << " vs variable uid " << (*new_cell_uids)[cell] << " vs parent uid " << parent.uniqueId();
          ++nerror2;
        }
      }
    }
    if (nerror2>0) fatal() << nerror2 << " errors";
  }
  new_mesh->checkValidMeshFull();

#if 0
  { // Comparaison des uids entre Item et Item parent à l'écran
    eItemKind kinds[] = { IK_Node, IK_Face, IK_Cell };
    Integer nbKind = sizeof(kinds)/sizeof(eItemKind);
    for(Integer i=0;i<nbKind;++i) {
      IItemFamily * family = new_mesh->itemFamily(kinds[i]);
      if (family->parentFamily()) {
        info() << "Checking uids for family " << family->name() << " vs parent family " << family->parentFamily()->name();
        ItemGroup myAllItems = family->allItems();
        ENUMERATE_ITEM(iitem,myAllItems) {
          info() << ItemPrinter(*iitem) << " " << ItemPrinter((*iitem).parent());
          
        }
      }
    }
  }
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SubMeshTestModule::
build()
{
  Connectivity c(mesh()->connectivity());
  if (options()->submeshKind() == IK_Face){
    info() << "Adding edge connectivity";
    c.enableConnectivity(Connectivity::CT_HasEdge);
  }
  // c.enableConnectivity(Connectivity::CT_EdgeConnectivity);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SubMeshTestModule::
init()
{
  info() << "SubMesh Test started";

//  ENUMERATE_EDGE (iedge,mesh()->allEdges()) {
//    info() << " Test edge " << FullItemPrinter(*iedge);
//  }

  // Génération d'un sous-maillage du genre demandé
  parentKind = options()->submeshKind(); 

  myParentFamily = mesh()->itemFamily(parentKind);
  myParentItems = myParentFamily->createGroup("MyItems");
  allParentItems = myParentFamily->allItems();
  myNewParentItems = myParentFamily->createGroup("MyNewItems");
  myOldParentItems = myParentFamily->createGroup("MyOldItems");

  if (parentKind == IK_Face) {

    IItemFamily * groupFamily = mesh()->faceFamily();
    Int32UniqueArray item_localids;
    ENUMERATE_FACE(iface,groupFamily->findGroup("XMIN"))
      item_localids.add(iface->localId());
    ENUMERATE_FACE(iface,groupFamily->findGroup("YMIN"))
      item_localids.add(iface->localId());
    ENUMERATE_FACE(iface,groupFamily->findGroup("ZMIN"))
      item_localids.add(iface->localId());
    myParentItems.addItems(item_localids);

    item_localids.clear();
    ENUMERATE_FACE(iface,groupFamily->findGroup("YMAX"))
      item_localids.add(iface->localId());
    ENUMERATE_FACE(iface,groupFamily->findGroup("YMIN"))
      item_localids.add(iface->localId());
    myOldParentItems.addItems(item_localids);
    
    item_localids.clear();
    ENUMERATE_FACE(iface,groupFamily->findGroup("ZMAX"))
      item_localids.add(iface->localId());
    myNewParentItems.addItems(item_localids);

  } else if (parentKind == IK_Cell) {

    IItemFamily * groupFamily = mesh()->faceFamily();
    std::set<Int64> restricted_uids;
    ENUMERATE_FACE(iface,groupFamily->findGroup("XMIN").own())
      restricted_uids.insert(iface->boundaryCell().uniqueId());
    ENUMERATE_FACE(iface,groupFamily->findGroup("YMIN").own())
      restricted_uids.insert(iface->boundaryCell().uniqueId());
    ENUMERATE_FACE(iface,groupFamily->findGroup("ZMIN").own())
      restricted_uids.insert(iface->boundaryCell().uniqueId());
    ENUMERATE_FACE(iface,groupFamily->findGroup("ZMAX").own())
      restricted_uids.insert(iface->boundaryCell().uniqueId());
    Int32UniqueArray item_localids;
    ENUMERATE_CELL(icell,ownCells()) {
      if (restricted_uids.find(icell->uniqueId()) == restricted_uids.end())
        item_localids.add(icell->localId());
    }
    // myParentItems.addItems(item_localids);
    myParentItems.setItems(allCells().view().localIds());

    info() << "XTestMesh " << m_global_iteration() << " INIT parent group size "
           << myParentItems.own().size() << " " << myParentItems.size();

    item_localids.clear();
    ENUMERATE_FACE(iface,groupFamily->findGroup("YMAX"))
      item_localids.add(iface->boundaryCell().localId());
//     ENUMERATE_FACE(iface,groupFamily->findGroup("YMIN"))
//       item_localids.add(iface->boundaryCell().localId());
    myOldParentItems.addItems(item_localids);
    
    item_localids.clear();
    ENUMERATE_FACE(iface,groupFamily->findGroup("ZMAX"))
      item_localids.add(iface->boundaryCell().localId());
    myNewParentItems.addItems(item_localids);
  }
  else
    fatal() << "Not implemented sub-mesh kind " << parentKind;

  info() << "MyParentItems : " << myParentItems.size() << " / " << allParentItems.size();
  ENUMERATE_ITEM(iitem,myParentItems){
    debug(Trace::High) << "Item to build : " << ItemPrinter(*iitem);
  }
  info() << "MyOldParentItems : " << myOldParentItems.size() << " / " << allParentItems.size();
  ENUMERATE_ITEM(iitem, myOldParentItems){
    debug(Trace::High) << "Item to delete : " << ItemPrinter(*iitem);
  }
  info() << "MyNewParentItems : " << myNewParentItems.size() << " / " << allParentItems.size();
  ENUMERATE_ITEM(iitem, myNewParentItems){
    debug(Trace::High) << "Item to add : " << ItemPrinter(*iitem);
  }
  
  _checkCreateOutputDir();
  IPostProcessorWriter* post_processor = options()->format();
  post_processor->setBaseDirectoryName(m_output_directory.path());

  m_global_deltat = 1.;

  // Données de contrôle
  node_uids = new VariableNodeInt64(Arcane::VariableBuildInfo(mesh(), "Uids", mesh()->nodeFamily()->name()));
  ENUMERATE_NODE(inode,mesh()->allNodes())
    (*node_uids)[inode] = inode->uniqueId();

  // Teste la connectivité aux arêtes
  if (options()->submeshKind() == IK_Face)
    _checkEdgeConnectivity();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SubMeshTestModule::
startInit()
{
  ;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SubMeshTestModule::
continueInit()
{
  ;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SubMeshTestModule::
_compute1CreateMesh() 
{
  // Création du sous-maillage
  new_mesh = subDomain()->mainFactory()->createSubMesh(mesh(),
                                                       myParentItems,
                                                       "TestMesh");
  // Statistiques sur sous-maillage
  MeshStats stats(traceMng(),new_mesh,subDomain()->parallelMng());
  stats.dumpStats();

  // Si le maillage est de dimension 1, vérifie que toutes les mailles
  // ont bien 2 faces.
  Int32 mesh_dimension = new_mesh->dimension();
  if (mesh_dimension==1){
    ENUMERATE_(Cell,icell,new_mesh -> allCells()) {
      Cell c = *icell;
      //info()<<"SUBMESH CELL : "<<icell->uniqueId()<<" nb faces = "<<c.nbFace();
      if (c.nbFace()!=2)
        ARCANE_FATAL("Bad number of faces for cell");
    }
  }

  if (check_variable){
    new_data = new VariableCellReal(Arcane::VariableBuildInfo(new_mesh, "Data", new_mesh->cellFamily()->name())); // , Arcane::IVariable::PNoDump|Arcane::IVariable::PNoNeedSync));
    new_cell_uids  = new VariableCellInt64(Arcane::VariableBuildInfo(new_mesh, "CellUids", new_mesh->cellFamily()->name()));
    new_face_uids  = new VariableFaceInt64(Arcane::VariableBuildInfo(new_mesh, "FaceUids", new_mesh->faceFamily()->name()));
    new_node_uids  = new VariableNodeInt64(Arcane::VariableBuildInfo(new_mesh, "NodeUids", new_mesh->nodeFamily()->name()));

    // Mise en place de données de contrôle
    ENUMERATE_CELL(icell,new_mesh->allCells())
      (*new_cell_uids)[icell] = icell->uniqueId();
    ENUMERATE_FACE(iface,new_mesh->allFaces())
      (*new_face_uids)[iface] = iface->uniqueId();
    ENUMERATE_NODE(inode,new_mesh->allNodes())
      (*new_node_uids)[inode] = inode->uniqueId();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SubMeshTestModule::
_compute2RemoveItems() 
{
  // Suppression d'items du support...
  info() << "Remove parent items from group " << myOldParentItems.name() << " " << myOldParentItems.size() << " / " << allParentItems.size();
  ENUMERATE_ITEM(iitem,myOldParentItems) debug(Trace::Highest) << "Removing " << iitem.index() << " " << ItemPrinter(*iitem);

  if (parentKind == IK_Cell) {
    // Suppression depuis le groupe support
    myParentItems.removeItems(myOldParentItems.view().localIds());
    // On peut aussi supprimer depuis le maillage support, mais ce maillage doit alors être correct (pas d'item orphelin)
    // Suppression réelle depuis le maillage
    // mesh()->modifier()->removeCells(myOldParentItems.view().localIds());
    // mesh()->modifier()->endUpdate();
  } else if (parentKind == IK_Face) {
    // Suppression depuis le groupe support
    myParentItems.removeItems(myOldParentItems.view().localIds());
  } else 
    fatal() << "Not implemented sub-mesh kind " << parentKind;

  // SdC add ghost rebuild layer in tests (usage in IFPEN applications)
  if (new_mesh)
    new_mesh->modifier()->endUpdate(true, true); // RELOCALISER LE CONCEPT DANS ARCANE
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SubMeshTestModule::
_compute3AddItems() 
{
  // Ajout de nouveaux items sur le groupe fondation du sous-maillage
  // induit un retaillage du sous-maillage
  info() << "Add parent items from group " << myNewParentItems.name() << " " << myNewParentItems.size() << " / " << allParentItems.size();
  myParentItems.addItems(myNewParentItems.view().localIds());

  if (new_mesh){
    new_mesh->modifier()->endUpdate(true,true); // RELOCALISER LE CONCEPT DANS ARCANE
      
    // Not sync after addItems (step 3)
    ItemVector parent2sub = MeshToMeshTransposer::transpose(mesh(),new_mesh,myNewParentItems.view(),true);
    if (check_variable) {
      ENUMERATE_CELL(icell,parent2sub) {
        (*new_cell_uids)[icell] = icell->uniqueId();
        for ( Face face : icell->faces()){
          (*new_face_uids)[face] = face.uniqueId();
        }
        for ( Node node : icell->nodes()){
          (*new_node_uids)[node] = node.uniqueId();
        }
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SubMeshTestModule::
_compute4TransposeItems() 
{
  if (!new_mesh)
    return;

  // Test de transposition
  ItemVector parent2sub = MeshToMeshTransposer::transpose(mesh(),new_mesh,allParentItems.view());
  //     info() << "Parent ParentItems";
  //     ENUMERATE_ITEM(iitem,ParentItems) info() << ItemPrinter(*iitem);
  //     info() << "Transposed ParentItems on TestMesh";
  //     ENUMERATE_ITEM(iitem,parent2sub) if (iitem.localId() != NULL_ITEM_LOCAL_ID) info() << ItemPrinter(*iitem);
  info() << "-----------------------------------------------------";
  ItemVector sub2parent = MeshToMeshTransposer::transpose(new_mesh,mesh(),new_mesh->allCells().view());
  //     info() << "TestMesh AllCells";
  //     ENUMERATE_ITEM(iitem,new_mesh->allCells()) info() << ItemPrinter(*iitem);
  //     info() << "Transposed AllCells to Parent Mesh";
  //     ENUMERATE_ITEM(iitem,sub2parent) info() << ItemPrinter(*iitem);
  //     info() << "-----------------------------------------------------";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SubMeshTestModule::
_compute5MoveItems() 
{
  // Test de migration
  VariableItemInt32& cells_new_owner = mesh()->toPrimaryMesh()->itemsNewOwner(IK_Cell);
  ENUMERATE_FACE(iface,allFaces()) {
    if (!iface->isOwn())
      for( CellLocalId icell : iface->cells())
        cells_new_owner[icell] = iface->owner();
  }
  info() << "Own cells before migration (" << ownCells().size() << " / " << allCells().size() << " )";
  //     ENUMERATE_CELL(icell,mesh()->ownCells()) info() << icell.index() << ": " << ItemPrinter(*icell);
  cells_new_owner.synchronize();
  Integer moved_cell_count = 0;
  ENUMERATE_CELL(icell,ownCells()) {
    if (cells_new_owner[icell] != icell->owner())
      {
        ++moved_cell_count;
        debug(Trace::Highest) << "Move cell " << ItemPrinter(*icell) << " to " << cells_new_owner[icell];
      }
  }

  if (new_mesh)
    {
      // Casse volontairement les données ghosts pour vérifier le bon transfert des originaux
      ENUMERATE_CELL(icell,new_mesh->allCells())
        if (!icell->isOwn())
          (*new_cell_uids)[icell] = NULL_ITEM_UNIQUE_ID;
      ENUMERATE_FACE(iface,new_mesh->allFaces())
        if (!iface->isOwn())
          (*new_face_uids)[iface] = NULL_ITEM_UNIQUE_ID;
      ENUMERATE_NODE(inode,new_mesh->allNodes())
        if (!inode->isOwn())
          (*new_node_uids)[inode] = NULL_ITEM_UNIQUE_ID;
    }      

  info() << "Own cells to move in migration : " <<  moved_cell_count;
  mesh()->utilities()->changeOwnersFromCells();

  mesh()->modifier()->setDynamic(true);
  mesh()->toPrimaryMesh()->exchangeItems();
  info() << "Own cells after migration (" << ownCells().size() << " / " << allCells().size() << " )";
  //     ENUMERATE_CELL(icell,mesh()->ownCells()) info() << icell.index() << ": " << ItemPrinter(*icell);

  if (new_mesh)
    new_mesh->modifier()->endUpdate(true,true); // RELOCALISER LE CONCEPT DANS ARCANE
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SubMeshTestModule::
_postProcessSubMesh()
{
  if (!new_mesh)
    return;

  info() << "Post-process sub-mesh " << myParentItems.size() << " / " << allParentItems.size();
  IPostProcessorWriter* post_processor = options()->format();
  times.add(m_global_time());
  post_processor->setTimes(times);
  post_processor->setMesh(new_mesh);

  if (parentKind == IK_Face) {
    info() << "Post-processor item kind " << parentKind;
    m_face_real_values.fill(0,allParentItems);
    m_face_real_values.fill(2,myParentItems);
  } else if (parentKind == IK_Cell) {
    info() << "Post-processor item kind " << parentKind;
    m_cell_real_values.fill(0,allParentItems);
    m_cell_real_values.fill(2,myParentItems);
    info() << "myParentItems : " << myParentItems.size();
  } else
    fatal() << "Not implemented sub-mesh kind " << parentKind;

  m_data.fill(0.);
  ENUMERATE_CELL(icell,allCells()) {
    m_data[icell] = icell->owner();
  }

  if (check_variable) {
    new_data->fill(0.);
    ENUMERATE_CELL(icell,new_mesh->allCells()) {
      (*new_data)[icell] = icell->owner();
    }
  }

  if (check_variable) {
    VariableList variables;
    variables.add(new_data->variable());
    post_processor->setVariables(variables);
    ItemGroupList groups;
    groups.add(new_mesh->allCells());
    post_processor->setGroups(groups);
    IVariableMng * vm = subDomain()->variableMng();
    vm->writePostProcessing(post_processor);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SubMeshTestModule::
compute()
{
  const Integer current_iteration = m_global_iteration();

  if (current_iteration == 1) {
    info() << "Step 1 - create mesh";
    _compute1CreateMesh();
  } else if (current_iteration == 2) {
    info() << "Step 2 - remove items";
    _compute2RemoveItems();
  } else if (current_iteration == 3) {
    info() << "Step 3 - add items";
    _compute3AddItems();
  } else if (current_iteration == 4) {
    info() << "Step 4 - transpose items";
    _compute4TransposeItems();
  } else if (current_iteration > 4) {
    info() << "Step >4 - move items";
    _compute5MoveItems();
  }

  _checkSubMeshIntegrity();
  _postProcessSubMesh();

  if (current_iteration>options()->nbIteration())
    subDomain()->timeLoopMng()->stopComputeLoop(true);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void SubMeshTestModule::
_checkEdgeConnectivity()
{
  UnstructuredMeshConnectivityView mc;
  mc.setMesh(mesh());

  auto edge_node = mc.edgeNode();
  auto edge_face = mc.edgeFace();
  auto edge_cell = mc.edgeCell();
  Int64 total_id = 0;
  ENUMERATE_(Edge,iedge,mesh()->allEdges()){
    Edge edge = *iedge;
    bool do_print = edge.localId()<12;
    if (do_print)
      info() << "EDGE i=" << edge.localId();
    for( NodeLocalId node : edge_node.nodes(edge) ){
      if (do_print)
        info() << "  NODE i=" << node.localId();
      total_id += node.localId();
    }
    for( FaceLocalId face : edge_face.faces(edge) ){
      if (do_print)
        info() << "  FACE i=" << face.localId();
      total_id += face.localId();
    }
    for( CellLocalId cell : edge_cell.cells(edge) ){
      if (do_print)
        info() << "  CELL i=" << cell.localId();
      total_id += cell.localId();
    }
  }
  info() << "TOTAL_ID=" << total_id;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace ArcaneTest

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
