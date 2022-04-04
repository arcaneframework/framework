﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshUnitTest.cc                                             (C) 2000-2021 */
/*                                                                           */
/* Service du test du maillage.                                              */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Array.h"
#include "arcane/utils/StringBuilder.h"
#include "arcane/utils/ScopedPtr.h"
#include "arcane/utils/List.h"
#include "arcane/utils/MD5HashAlgorithm.h"
#include "arcane/utils/ArithmeticException.h"
#include "arcane/utils/ValueChecker.h"

#include "arcane/BasicUnitTest.h"

#include "arcane/tests/ArcaneTestGlobal.h"
#include "arcane/tests/MeshUnitTest_axl.h"

#include "arcane/AbstractItemOperationByBasicType.h"
#include "arcane/IMeshWriter.h"
#include "arcane/IParallelMng.h"
#include "arcane/MeshUtils.h"
#include "arcane/IMesh.h"
#include "arcane/IMeshSubMeshTransition.h"
#include "arcane/IMeshModifier.h"
#include "arcane/ITiedInterface.h"
#include "arcane/IItemFamily.h"
#include "arcane/IItemConnectivityInfo.h"
#include "arcane/ItemPrinter.h"
#include "arcane/IMeshUtilities.h"
#include "arcane/IVariableMng.h"
#include "arcane/Directory.h"
#include "arcane/IVariableSynchronizer.h"
#include "arcane/VariableCollection.h"
#include "arcane/ServiceBuilder.h"
#include "arcane/IParallelReplication.h"
#include "arcane/IndexedItemConnectivityView.h"

#include "arcane/ServiceFinder2.h"
#include "arcane/SerializeBuffer.h"
#include "arcane/IMeshPartitioner.h"
#include "arcane/IMainFactory.h"
#include "arcane/IMeshModifier.h"
#include "arcane/Properties.h"
#include "arcane/Timer.h"

#include "arcane/ItemArrayEnumerator.h"
#include "arcane/ItemPairGroup.h"
#include "arcane/ItemPairEnumerator.h"
#include "arcane/ItemPairGroupBuilder.h"

#include "arcane/IPostProcessorWriter.h"

#include "arcane/ItemVectorView.h"

#include "arcane/GeometricUtilities.h"

#include "arcane/MeshVisitor.h"

#include "arcane/IMeshReader.h"
#include "arcane/IXmlDocumentHolder.h"
#include "arcane/IIOMng.h"
#include "arcane/MeshReaderMng.h"

#include "arcane/mesh/IncrementalItemConnectivity.h"

#include <set>

#ifdef ARCANE_HAS_CUSTOM_MESH_TOOLS
#include "neo/Mesh.h"
#endif

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
 * \brief Module de test du maillage
 */
class MeshUnitTest
: public ArcaneMeshUnitTestObject
{
public:

  class CountOperationByBasicType
  : public TraceAccessor, public AbstractItemOperationByBasicType
  {
  public:
    CountOperationByBasicType(ITraceMng* m) : TraceAccessor(m) {}
  public:
    virtual void applyVertex(ItemVectorView group)
    { info() << "NB Vertex = " << group.size(); }
    virtual void applyLine2(ItemVectorView group) 
    { info() << "NB Line2 = " << group.size(); }
    virtual void applyTriangle3(ItemVectorView group) 
    { info() << "NB Triangle3 = " << group.size(); }
    virtual void applyQuad4(ItemVectorView group) 
    { info() << "NB Quad4 = " << group.size(); }
    virtual void applyPentagon5(ItemVectorView group) 
    { info() << "NB Pentagon5 = " << group.size(); }
    virtual void applyHexagon6(ItemVectorView group) 
    { info() << "NB Hexagon6 = " << group.size(); }
    virtual void applyTetraedron4(ItemVectorView group) 
    { info() << "NB Tetraedron4 = " << group.size(); }
    virtual void applyPyramid5(ItemVectorView group) 
    { info() << "NB Pyramid5 = " << group.size(); }
    virtual void applyPentaedron6(ItemVectorView group) 
    { info() << "NB Pentaedron6 = " << group.size(); }
    virtual void applyHexaedron8(ItemVectorView group) 
    { info() << "NB Hexaedron8 = " << group.size(); }
    virtual void applyHeptaedron10(ItemVectorView group) 
    { info() << "NB Heptaedron10 = " << group.size(); }
    virtual void applyOctaedron12(ItemVectorView group) 
    { info() << "NB Octaedron12 = " << group.size(); }
    virtual void applyHemiHexa7(ItemVectorView group) 
    { info() << "NB HemiHexa7 = " << group.size(); }
    virtual void applyHemiHexa6(ItemVectorView group) 
    { info() << "NB HemiHexa6 = " << group.size(); }
    virtual void applyHemiHexa5(ItemVectorView group) 
    { info() << "NB HemiHexa5 = " << group.size(); }
    virtual void applyAntiWedgeLeft6(ItemVectorView group) 
    { info() << "NB AntiWedgeLeft6 = " << group.size(); }
    virtual void applyAntiWedgeRight6(ItemVectorView group) 
    { info() << "NB AntiWedgeRight6 = " << group.size(); }
    virtual void applyDiTetra5(ItemVectorView group) 
    { info() << "NB DiTetra5 = " << group.size(); }
    virtual void applyDualNode(ItemVectorView group) 
    { info() << "NB DualNode = " << group.size(); }
    virtual void applyDualEdge(ItemVectorView group) 
    { info() << "NB DualEdge = " << group.size(); }
    virtual void applyDualFace(ItemVectorView group) 
    { info() << "NB DualFace = " << group.size(); }
    virtual void applyDualCell(ItemVectorView group) 
    { info() << "NB DualCell = " << group.size(); }
    virtual void applyLink(ItemVectorView group) 
    { info() << "NB Link = " << group.size(); }
  };

public:

  MeshUnitTest(const ServiceBuildInfo& cb);
  ~MeshUnitTest();

 public:

  void initializeTest() override;
  void executeTest() override;

 private:

  void _dumpMesh();
  void _dumpTiedInterfaces();
  void _testGroups();
  void _dumpComputeFaceGroupNormal();
  void _dumpComputeNodeGroupDirection();
  void _testItemAdjency();
  void _testItemAdjency2();
  void _testItemAdjency3();
  void _testItemPartialAdjency();
  void _testVariableWriter();
  void _testItemArray();
  void _testProjection();
  void _testMD5();
  void _dumpConnections();
  void _partitionMesh(Int32 nb_part);
  void _testUsedVariables();
  void _dumpConnectivityInfos(IItemConnectivityInfo* cell_family,
                              IItemConnectivityInfo* face_family,
                              IItemConnectivityInfo* node_family);
  void _testSharedItems();
  void _testVisitors();
  template<typename ItemKind,typename SubItemKind>
  void _testItemAdjency(ItemGroupT<ItemKind> items,ItemGroupT<SubItemKind> subitems,
                        eItemKind link_kind);
  void _testAdditionalMeshes();
  void _testNullItem();
  void _testCustomMeshTools();
  void _testAdditionnalConnectivity();
  void _testShrinkGroups();
  void _testDeallocateMesh();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE_MESHUNITTEST(MeshUnitTest,MeshUnitTest);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshUnitTest::
MeshUnitTest(const ServiceBuildInfo& mb)
: ArcaneMeshUnitTestObject(mb)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshUnitTest::
~MeshUnitTest()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshUnitTest::
executeTest()
{
  CountOperationByBasicType op(traceMng());
  info() << "ItemTypeMng::singleton() = " << ItemTypeMng::singleton();
  info() << "Infos sur AllCells:";
  allCells().applyOperation(&op);
  info() << "Infos sur AllFaces:";
  allFaces().applyOperation(&op);
  info() << "Infos sur AllNodes:";
  allNodes().applyOperation(&op);
  if (options()->writeMesh())
    _dumpMesh();
  _testNullItem();
  _dumpTiedInterfaces();
  _dumpComputeFaceGroupNormal();
  _dumpComputeNodeGroupDirection();
  _testGroups();
  if (options()->testAdjency()){
    _testItemAdjency();
    _testItemAdjency2();
    _testItemAdjency3();
    _testItemPartialAdjency();
  }
  _dumpConnections();
  {
    info() << " ** ** CHECK UPDATE GHOST LAYER";
    mesh()->modifier()->setDynamic(true);
    mesh()->modifier()->updateGhostLayers();
    mesh()->toPrimaryMesh()->nodesCoordinates().checkIfSync(100);
  }
  if (options()->testVariableWriter())
    _testVariableWriter();
  _testItemArray();
  _testProjection();
  _testVisitors();
  _testSharedItems();
#if 0
  try{
    _testProjection();
  }
  catch(const ArithmeticException& ex)
  {
    error() << "ArithmeticException 1 catched!";
  }
  try{
    _testProjection();
  }
  catch(const ArithmeticException& ex)
  {
    error() << "ArithmeticException 2 catched!";
  }
#endif
  _testMD5();
  _testUsedVariables();
  _testAdditionalMeshes();
  _testCustomMeshTools();
  _testAdditionnalConnectivity();
  _testShrinkGroups();
  if (options()->testDeallocateMesh())
    _testDeallocateMesh();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshUnitTest::
initializeTest()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshUnitTest::
_testNullItem()
{
  Cell null_cell;
  info() << "NULL_CELL"
         << " nb_node=" << null_cell.nbNode()
         << " nb_edge=" << null_cell.nbEdge()
         << " nb_face=" << null_cell.nbFace()
  //<< " nb_hparent=" << null_cell.nbParent()
         << " nb_hchildren=" << null_cell.nbHChildren();
  Node null_node;
  info() << "NULL_NODE"
         << " nb_edge=" << null_node.nbEdge()
         << " nb_face=" << null_node.nbFace()
         << " nb_cell=" << null_node.nbCell();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshUnitTest::
_dumpMesh()
{
  ServiceBuilder<IMeshWriter> sbu(subDomain());
  auto mesh_io(sbu.createReference("Lima",SB_AllowNull));
  IParallelMng* pm = subDomain()->parallelMng();
  bool is_parallel = pm->isParallel();
  Integer sid = pm->commRank();
  StringBuilder sorted_file_name(options()->outputFile());
  sorted_file_name += "-sorted";
  if (is_parallel){
    sorted_file_name += "-";
    sorted_file_name += sid;
  }
  mesh_utils::writeMeshInfosSorted(mesh(),sorted_file_name);
  StringBuilder file_name_b(options()->outputFile());
  if (is_parallel){
    file_name_b += "-";
    file_name_b += sid;
  }
  String file_name(file_name_b.toString());
  mesh_utils::writeMeshInfos(mesh(),file_name);
  mesh_utils::writeMeshConnectivity(mesh(),file_name+".xml");
  if (mesh_io.get()){
    file_name = file_name + ".unf";
    mesh_io->writeMeshToFile(mesh(),file_name);
  }
  info() << "Local connectivity infos:";
  _dumpConnectivityInfos(mesh()->cellFamily()->localConnectivityInfos(),
                         mesh()->faceFamily()->localConnectivityInfos(),
                         mesh()->nodeFamily()->localConnectivityInfos());

  info() << "Global connectivity infos:";
  _dumpConnectivityInfos(mesh()->cellFamily()->globalConnectivityInfos(),
                         mesh()->faceFamily()->globalConnectivityInfos(),
                         mesh()->nodeFamily()->globalConnectivityInfos());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshUnitTest::
_dumpConnectivityInfos(IItemConnectivityInfo* cell_family,IItemConnectivityInfo* face_family,
                   IItemConnectivityInfo* node_family)
{
  info() << "max node per cell = " << cell_family->maxNodePerItem();
  info() << "max edge per cell = " << cell_family->maxEdgePerItem();
  info() << "max face per cell = " << cell_family->maxFacePerItem();
  info() << "max local edge per cell = " << cell_family->maxEdgeInItemTypeInfo();
  info() << "max local face per cell = " << cell_family->maxFaceInItemTypeInfo();

  info() << "max node per face = " << face_family->maxNodePerItem();
  info() << "max edge per face = " << face_family->maxEdgePerItem();

  info() << "max face per node = " << node_family->maxFacePerItem();
  info() << "max cell per node = " << node_family->maxCellPerItem();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshUnitTest::
_testUsedVariables()
{
  IItemFamily* cell_family = mesh()->cellFamily();
  info() << "Cell variables:";
  VariableList vars;
  cell_family->usedVariables(vars);
  for( VariableList::Enumerator i(vars); ++i; ){
    IVariable* var = *i;
    info() << "name=" << var->name(); 
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshUnitTest::
_dumpTiedInterfaces()
{
  bool has_tied_interface = mesh()->hasTiedInterface();
  if (!has_tied_interface)
    info() << "No tied interface in the mesh";
  TiedInterfaceCollection tied_interfaces(mesh()->tiedInterfaces());
  info() << "---------------------------------------------------------";
  info() << "Surfaces liées: n=" << tied_interfaces.count();
    
  Integer nb_error = 0;
  Integer max_print_error = 50;
  std::set<Int64> nodes_in_master_face;
  for( TiedInterfaceCollection::Enumerator i(tied_interfaces); ++i; ){
    ITiedInterface* interface = *i;
    FaceGroup slave_group = interface->slaveInterface();
    FaceGroup master_group = interface->masterInterface();
    info() << "Interface: Slave=" << slave_group.name()
           << " nb_face=" << slave_group.size();
    info() << "Interface: Master=" << master_group.name()
           << " nb_face=" << master_group.size();
    TiedInterfaceNodeList tied_nodes(interface->tiedNodes());
    TiedInterfaceFaceList tied_faces(interface->tiedFaces());
    ENUMERATE_FACE(iface,master_group){
      Face face = *iface;
      Int32 master_face_owner = face.owner();
      FaceVectorView slave_faces = face.slaveFaces();
      if (!face.isMasterFace()){
        ++nb_error;
        if (nb_error<max_print_error)
          error() << " Face uid=" << ItemPrinter(face) << " should have isMaster() true";
      }
      if (iface.index()<100000){
        nodes_in_master_face.clear();
        info() << "Master face uid=" << ItemPrinter(face)
               << " kind=" << face.kind()
               << " cell=" << ItemPrinter(face.cell(0));
        Int32 cell_face_owner = face.cell(0).owner();
        if (cell_face_owner!=master_face_owner){
          ++nb_error;
          if (nb_error<max_print_error)
            error() << "master_face and its cell do not have the same owner: face_owner=" << master_face_owner
                    << " cell_owner=" << cell_face_owner;
        }
        for( NodeEnumerator inode(face.nodes()); inode.hasNext(); ++inode )
          info() << "Master face node uid=" << inode->uniqueId();
        for( Integer zz=0, zs=tied_nodes[iface.index()].size(); zz<zs; ++zz ){
          const TiedNode& tn = tied_nodes[iface.index()][zz];
          nodes_in_master_face.insert(tn.node().uniqueId());
          if (zz<20){
            info() << " node_uid=" << tn.node().uniqueId()
                   << " iso=" << tn.isoCoordinates()
                   << " kind=" << tn.node().kind();
          }
        }
        for( NodeEnumerator inode(face.nodes()); inode.hasNext(); ++inode )
          if (nodes_in_master_face.find(inode->uniqueId())==nodes_in_master_face.end()){
            ++nb_error;
            if (nb_error<max_print_error)
              error() << "node in master face not in slave node list uid=" << inode->uniqueId();
          }
        Integer nb_tied = tied_faces[iface.index()].size();
        if (nb_tied!=slave_faces.size()){
          ++nb_error;
          if (nb_error<max_print_error)
            error() << "bad number of slave faces interne=" << slave_faces.size() << " struct=" << nb_tied;
        }
        for( Integer zz=0, zs=tied_faces[iface.index()].size(); zz<zs; ++zz ){
          const TiedFace& tf = tied_faces[iface.index()][zz];
          Face tied_slave_face = tf.face();
          if (!tied_slave_face.isSlaveFace()){
            ++nb_error;
            if (nb_error<max_print_error)
              error() << "slave face uid=" << ItemPrinter(tf.face()) << " should have isSlave() true";
          }
          if (tied_slave_face.masterFace()!=face){
            ++nb_error;
            if (nb_error<max_print_error)
              error() << "slave face uid=" << ItemPrinter(tf.face()) << " should have masterSlave() valid";
          }
          if (tied_slave_face!=slave_faces[zz]){
            ++nb_error;
            if (nb_error<max_print_error)
              error() << "bad number of slave faces internal=" << ItemPrinter(slave_faces[zz])
                      << " struct=" << ItemPrinter(tied_slave_face);
          }
          Int32 slave_face_owner = tf.face().owner();
          if (slave_face_owner!=master_face_owner){
            ++nb_error;
            if (nb_error<max_print_error)
              error() << "master_face and its slave_face do not have the same owner:"
                      << " master_face=" << ItemPrinter(face)
                      << " master_cell=" << ItemPrinter(face.cell(0))
                      << " slave_face=" << ItemPrinter(tf.face())
                      << " slave_cell=" << ItemPrinter(tf.face().cell(0));
          }
          if (zz<20){
            info() << " face_uid=" << tf.face().uniqueId() << " cell=" << ItemPrinter(tf.face().cell(0));

          }
        }
      }
    }
  }
  info() << "---------------------------------------------------------";
  if (nb_error!=0){
    ARCANE_FATAL("Errors in tied interface n={0}",nb_error);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshUnitTest::
_testGroups()
{
  IItemFamily* item_family = mesh()->cellFamily();
  CellGroup all_items = item_family->allItems();
  if (!all_items.checkIsSorted())
    throw FatalErrorException(A_FUNCINFO,"AllItems group is not sorted");

  Integer nb_item = all_items.size();
  CellGroup group = mesh()->cellFamily()->findGroup("TestGroup",true);
  {
    UniqueArray<Int32> items;
    for( Integer i=4, n=(nb_item/2); i<n; ++i )
      items.add(i);
    group.setItems(items);
    items.clear();
    // Ajoute l'entité de localId() à la fin du groupe.
    // Comme c'est celle avec le plus petit uniqueId(), normalement le groupe
    // ne doit ensuite plus être trié.
    items.add(0);
    group.addItems(items);
    if (group.size()>1)
      if (group.checkIsSorted())
        throw FatalErrorException(A_FUNCINFO,"Group should not be sorted");
  }
  {
    Int32UniqueArray items;
    Integer range = nb_item / 5;
    for( Integer i=0; i<range; ++i )
      items.add(i);
    group.setItems(items);
    items.clear();
    for( Integer i=range, is=range*2; i<is; ++i )
      items.add(i);
    group.addItems(items);
    items.clear();
    for( Integer i=0, is=range; i<is; ++i )
      items.add(i*2);
    group.removeItems(items);
  }
  
  info() << " GROUP " << group.size() << " addr=" << group.internal();
  NodeGroup nodes = group.nodeGroup();
  info() << " NB NODE in group " << nodes.size() << " addr=" << nodes.internal();
  const char* P_SORT = "sort-subitemitem-group";
  // Force le tri pour le groupe des faces de 'group' et vérifie si OK.
  bool is_sort = mesh()->properties()->getBoolWithDefault(P_SORT,false);
  mesh()->properties()->setBool(P_SORT,true);
  FaceGroup faces = group.faceGroup();
  info() << " NB FACE in group " << faces.size() << " addr=" << faces.internal();
  if (!faces.checkIsSorted()){
    ARCANE_FATAL("FaceGroup should be sorted!");
  }
  // Remet comme avant.
  mesh()->properties()->setBool(P_SORT,is_sort);
  CellGroup cells = nodes.cellGroup();
  info() << " NB CELL in nodes " << cells.size() << " addr=" << cells.internal();
  CellGroup cells2 = group.cellGroup();
  info() << " NB CELL2 in group " << cells2.size() << " addr=" << cells2.internal();

  {
    Int32UniqueArray items;
    items.add(nb_item-1);
    items.add(0);
    items.add(nb_item/2);
    group.setItems(items,true);
    Int64 last_uid = NULL_ITEM_UNIQUE_ID;
    ENUMERATE_ITEM(iitem,group){
      Item item = *iitem;
      if (item.uniqueId()<last_uid)
        fatal() << "Group is not sorted";
      last_uid = item.uniqueId();
    }
    bool is_sorted = group.checkIsSorted();
    if (!is_sorted)
      throw FatalErrorException(A_FUNCINFO,"Incorrect value for ItemGroup::checkIsSorted()");
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshUnitTest::
_dumpComputeFaceGroupNormal()
{
  Integer nb_compute = options()->computeNormal.size();
  for( Integer i=0; i<nb_compute; ++i ){
    FaceGroup surface = options()->computeNormal[i];
    info() << " COMPUTING NORMAL INFORMATION FOR THE SURFACE <" << surface.name() << ">";
    IMesh* mesh = surface.itemFamily()->mesh();
    IMeshUtilities* util = mesh->utilities();
    VariableNodeReal3& nodes_coord(mesh->toPrimaryMesh()->nodesCoordinates());
    Real3 value = util->computeNormal(surface,nodes_coord);
    info() << " NORMAL IS <" << value << ">";
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshUnitTest::
_dumpComputeNodeGroupDirection()
{
  Integer nb_compute = options()->computeDirection.size();
  for( Integer i=0; i<nb_compute; ++i ){
    NodeGroup group = options()->computeDirection[i];
    info() << " COMPUTING DIRECTION INFORMATION FOR GROUP <" << group.name() << ">";
    IMesh* mesh = group.itemFamily()->mesh();
    IMeshUtilities* util = mesh->utilities();
    VariableNodeReal3& nodes_coord(mesh->toPrimaryMesh()->nodesCoordinates());
    Real3 value = util->computeDirection(group,nodes_coord,0,0);
    info() << " DIRECTION IS <" << value << ">";
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshUnitTest::
_testItemAdjency2()
{
  FaceFaceGroup ad_list(allFaces(),outerFaces(),IK_Node);
  info() << " COMPUTE ITEM ADJENCY LIST2!";
  
  std::set<Integer> boundary_set;
  ENUMERATE_FACE(iface,outerFaces())
    boundary_set.insert(iface->localId());

  Int64 total_uid = 0;
  ENUMERATE_ITEMPAIR(Face,Face,iface,ad_list){
    //const Face& item = *iface;
    // info() << " ITEM uid=" << item.uniqueId() << " nb_sub=" << iface.nbSubItem();
    
    ENUMERATE_SUB_ITEM(Face,isubface,iface){
      const Face& subface = *isubface;
      total_uid += subface.uniqueId().asInt64();
      // info() << " SUBITEM #" << isubface.index() << " : " << subface.localId() << " is_boundary=" << (boundary_set.find(subface.localId()) != boundary_set.end());
      if (boundary_set.find(subface.localId()) == boundary_set.end())
        fatal() << "Non boundary face [" << subface.localId() << " ] found";
    }
  }
  info() << " TOTAL uid=" << total_uid;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshUnitTest::
_testItemAdjency3()
{
  // Teste les fonctors pour le calcul des infos des ItemPairGroup.
  // Le fonctor calcule les mailles voisines aux mailles par les faces
  // et compare le résultat avec le ItemPairGroup équivalent calculé par
  // Arcane.
  auto f = [](ItemPairGroupBuilder& builder)
    {
      const ItemPairGroup& pair_group = builder.group();
      const ItemGroup& items = pair_group.itemGroup();
      const ItemGroup& sub_items = pair_group.subItemGroup();

      // Marque toutes les entités qui n'ont pas le droit d'appartenir à
      // la liste des connectivités car elles ne sont pas dans \a sub_items;
      std::set<Int32> allowed_ids;
      ENUMERATE_CELL(iitem,sub_items) {
        allowed_ids.insert(iitem.itemLocalId());
      }

      Int32UniqueArray local_ids;
      local_ids.reserve(8);

      // Liste des entités déjà traitées pour la maille courante
      std::set<Int32> already_in_list;
      ENUMERATE_CELL(icell,items){
        Cell cell = *icell;
        local_ids.clear();
        Int32 current_local_id = icell.itemLocalId();
        already_in_list.clear();

        // Pour ne pas s'ajouter à sa propre liste de connectivité
        already_in_list.insert(current_local_id);

        for( FaceEnumerator iface(cell.faces()); iface.hasNext(); ++iface ){
          Face face = *iface;
          for( CellEnumerator isubcell(face.cells()); isubcell.hasNext(); ++isubcell ){
            const Int32 sub_local_id = isubcell.itemLocalId();
            // Vérifie qu'on est dans la liste des mailles autorisées et qu'on
            // n'a pas encore été traité.
            if (allowed_ids.find(sub_local_id)==allowed_ids.end())
              continue;
            if (already_in_list.find(sub_local_id)!=already_in_list.end())
              continue;
            // Cette maille doit être ajoutée. On la marque pour ne pas
            // la parcourir et on l'ajoute à la liste.
            already_in_list.insert(sub_local_id);
            local_ids.add(sub_local_id);
          }
        }
        builder.addNextItem(local_ids);
      }
    };

  CellCellGroup ad_list(allCells(),allCells(),functor::makePointer(f));
  info() << " CUSTOM COMPUTE ITEM ADJENCY LIST2!";
  ItemPairGroupT<Cell,Cell> ref_ad_list(allCells(),allCells(),IK_Face);

  Int32UniqueArray items;
  Int32UniqueArray sub_items;
  ENUMERATE_ITEMPAIR(Cell,Cell,iitem,ad_list){
    items.add(iitem.itemLocalId());
    ENUMERATE_SUB_ITEM(Cell,isubitem,iitem){
      sub_items.add(isubitem.itemLocalId());
    }
  }
  info() << "NB_ITEM=" << items.size() << " nb_sub_item=" << sub_items.size();
  // Vérifie que les listes sont les mêmes entre la notre fonctor et la référence.
  Int32UniqueArray ref_items;
  Int32UniqueArray ref_sub_items;
  ENUMERATE_ITEMPAIR(Cell,Cell,iitem,ref_ad_list){
    ref_items.add(iitem.itemLocalId());
    ENUMERATE_SUB_ITEM(Cell,isubitem,iitem){
      ref_sub_items.add(isubitem.itemLocalId());
    }
  }
  ValueChecker vc(A_FUNCINFO);
  vc.areEqual(items,ref_items,"bad items");
  vc.areEqual(sub_items,ref_sub_items,"bad sub items");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename ItemKind,typename SubItemKind> void MeshUnitTest::
_testItemAdjency(ItemGroupT<ItemKind> items,ItemGroupT<SubItemKind> sub_items,
                 eItemKind link_kind)
{
  ItemPairGroupT<ItemKind,SubItemKind> ad_list(items,sub_items,link_kind);
  info() << " COMPUTE ITEM ADJENCY LIST link_kind=" << link_kind
         << " items=" << items.name() << " sub_items=" << sub_items.name()
         << " nb_item=" << items.size() << " nb_sub_item=" << sub_items.size()
         << " dim=" << items.mesh()->dimension();
  
  Int64 total_uid = 0;
  Integer nb_item = 0;
  ENUMERATE_ITEMPAIR(ItemKind,SubItemKind,iitem,ad_list){
    ++nb_item;
    ENUMERATE_SUB_ITEM(SubItemKind,isubitem,iitem){
      Item subitem = *isubitem;
      total_uid += subitem.uniqueId().asInt64();
    }
  }
  if (nb_item!=items.size())
    ARCANE_FATAL("Bad number of items n={0} expected={1}",nb_item,items.size());

  // On ne connait pas la valeur du total mais ce ne doit pas être nul
  // sauf si le lien est les arêtes où dans certains cas 1D.
  info() << " TOTAL uid=" << total_uid;
  bool may_be_null = (link_kind==IK_Edge);
  if (items.mesh()->dimension()==1){
    may_be_null = may_be_null || (link_kind==IK_Face && items.itemKind()==IK_Node);
    may_be_null = may_be_null || (link_kind==IK_Node && items.itemKind()==IK_Face);
  }
  if (total_uid==0 && !may_be_null)
    ARCANE_FATAL("Null total");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshUnitTest::
_testItemAdjency()
{
  _testItemAdjency(allCells(),allCells(),IK_Node);
  _testItemAdjency(allCells(),allCells(),IK_Face);

  _testItemAdjency(allNodes(),allNodes(),IK_Cell);
  _testItemAdjency(allNodes(),allNodes(),IK_Face);
  _testItemAdjency(allNodes(),allNodes(),IK_Edge);

  _testItemAdjency(allFaces(),allCells(),IK_Node);

  _testItemAdjency(allFaces(),allFaces(),IK_Node);
  _testItemAdjency(allFaces(),allFaces(),IK_Edge);
  _testItemAdjency(allFaces(),allFaces(),IK_Cell);

  _testItemAdjency(allCells(),allFaces(),IK_Face);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshUnitTest::
_testItemPartialAdjency()
{
  info() << " COMPUTE ITEM PARTIAL ADJENCY LIST!";
  
  // Choisit une cellule au "milieu" des ownCells().
  CellGroup myCells = ownCells();
  Integer nb_my_cells = myCells.size();
  Cell center_cell = myCells.view()[nb_my_cells/2].toCell();

  Int32UniqueArray center_cell_list;
  center_cell_list.add(center_cell.localId());
  CellGroup center_cell_group = mesh()->cellFamily()->createGroup("CenterCell",center_cell_list);
  
  CellCellGroup center_neighbor_pairgroup(center_cell_group,allCells(),IK_Node);
  Int32UniqueArray center_neighbor_list;

  ENUMERATE_ITEMPAIR(Cell,Cell,icell,center_neighbor_pairgroup) {
    const Cell& item = *icell;
    info() << " CENTER ITEM uid=" << item.uniqueId() << " nb_sub=" << icell.nbSubItem();
    center_neighbor_list.add(item.localId());
    if (item != center_cell) fatal() << "Inconsistent center cell";
    ENUMERATE_SUB_ITEM(Cell,isubcell,icell){
      const Cell& subcell = *isubcell;
      center_neighbor_list.add(subcell.localId());
    }
  }
  
  CellGroup center_neighbor_group = mesh()->cellFamily()->createGroup("CenterNeighBor",center_neighbor_list);
  m_cell_flag.fill(1);                       // Cellules interdites toutes sauf ...
  m_cell_flag.fill(0,center_neighbor_group); // Cellules autorisées
  CellCellGroup ad_list(center_neighbor_group,center_neighbor_group,IK_Node);

  ENUMERATE_ITEMPAIR(Cell,Cell,icell,ad_list){
    //const Cell& item = *icell;
    // info() << " ITEM uid=" << item.uniqueId() << " nb_sub=" << icell.nbSubItem();

    ENUMERATE_SUB_ITEM(Cell,isubcell,icell){
      const Cell& subcell = *isubcell;
      if (m_cell_flag[subcell])
        fatal() << "Out of group subcell";
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshUnitTest::
_dumpConnections()
{
  IItemFamily* item_family = mesh()->cellFamily();
  IVariableSynchronizer* vsync = item_family->allItemsSynchronizer();
  if (!vsync)
    return;
  info() << "Sync info for family name=" << item_family->fullName();
  Int32ConstArrayView comm_ranks = vsync->communicatingRanks();
  Integer nb_comm_rank = comm_ranks.size();
  info() << "Nb communicating ranks=" << nb_comm_rank;
  for( Integer i=0; i<nb_comm_rank; ++i ){
    Int32ConstArrayView share_ids = vsync->sharedItems(i);
    Int32ConstArrayView ghost_ids = vsync->ghostItems(i);
    Integer nb_share = share_ids.size();
    Integer nb_ghost = ghost_ids.size();
    info() << "COMM_RANK I=" << i << " rank=" << comm_ranks[i]
           << " nb_share=" << nb_share
           << " nb_ghost=" << nb_ghost;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshUnitTest::
_testVariableWriter()
{
  info() << " TEST VARIABLE WRITER!";
  if (mesh()->parallelMng()->isThreadImplementation()){
    info() << "Disabling VariableWriter with hdf5 in shared memory mode because hdf5 is not thread-safe";
    return;
  }

  ServiceBuilder<IPostProcessorWriter> sbu(subDomain());
  auto s_writer(sbu.createReference("Hdf5VariableWriter",SB_AllowNull));
  IPostProcessorWriter* writer = s_writer.get();
  if (!writer){
    warning() << A_FUNCINFO << ": no writer";
		return;
	}
  info() << " V=" << writer;

  VariableFaceReal face_temperature(VariableBuildInfo(mesh(),"FaceTemperature"));
  VariableFaceReal face_pressure(VariableBuildInfo(mesh(),"FacePressure"));
  ENUMERATE_FACE(iface,allFaces()){
    const Face& face = *iface;
    face_temperature[iface] = 3.2 * Convert::toReal(face.uniqueId().asInt64());
    face_pressure[iface] = 2.4 * Convert::toReal(face.uniqueId().asInt64());
  }
  VariableList vars_to_write;
  vars_to_write.add(face_temperature.variable());
  vars_to_write.add(face_pressure.variable());

  writer->setVariables(vars_to_write);
  IParallelMng* pm = subDomain()->parallelMng();
  String dir_name = "ic";
  Int32 replication_rank = pm->replication()->replicationRank();
  if (replication_rank>0){
    dir_name = dir_name + String::fromNumber(replication_rank);
  }
  Directory out_dir(dir_name);
  if (pm->isMasterIO())
    out_dir.createDirectory();
  writer->setBaseDirectoryName(out_dir.path());

  RealUniqueArray times_to_write;
  times_to_write.add(1.0);
  writer->setTimes(times_to_write);

  IVariableMng* vm = subDomain()->variableMng();
  vm->writePostProcessing(writer);

  times_to_write.add(2.0);
  writer->setTimes(times_to_write);
  vm->writePostProcessing(writer);

  times_to_write.add(3.0);
  writer->setTimes(times_to_write);
  vm->writePostProcessing(writer);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshUnitTest::
_testVisitors()
{
  ValueChecker vc(A_FUNCINFO);

  info() << "TEST MeshVisitor";
  int nb_group = 0;
  auto func = [&](const ItemGroup& g)
  {
    info() << "VisitGroup name=" << g.name();
    ++nb_group;
  };
  IItemFamily* item_family = mesh()->cellFamily();

  // Vérifie que le nombre de passage dans le visiteur est égal au nombre
  // de groupes de la famille.
  info() << "TEST Visit CellFamily";
  meshvisitor::visitGroups(item_family,func);
  vc.areEqual(nb_group,item_family->groups().count(),"Bad number of group for cell family");
  info() << "Nb group=" << nb_group;

  info() << "TEST Visit IMesh";
  nb_group = 0;
  Integer nb_expected_group = 0;
  for( IItemFamilyCollection::Enumerator ifamily(mesh()->itemFamilies()); ++ifamily; ){
    nb_expected_group += (*ifamily)->groups().count();
  }
  meshvisitor::visitGroups(mesh(),func);
  vc.areEqual(nb_group,nb_expected_group,"Bad number of group for mesh");
  info() << "Nb group=" << nb_group;

  // Regarde le nombre de groupes triés.
  {
    int nb_sorted_group = 0;
    nb_group =0;
    info() << "Check sorted groups";
    auto check_sorted_func = [&](ItemGroup& g)
    {
      ++nb_group;
      if (g.checkIsSorted())
        ++nb_sorted_group;
      else
        info() << "group not sorted name=" << g.name();
    };
    meshvisitor::visitGroups(mesh(),check_sorted_func);
    info() << "Nb group=" << nb_group << " nb_sorted=" << nb_sorted_group;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#define ENUMERATE_CELLZ(name,array) \
  for( ItemEnumerator2T<Cell> name(array.enumerator()); name.hasNext(); ++name )

template<typename ItemType>
class ItemEnumerator2T
: public ItemEnumerator
{
 public:

  ItemEnumerator2T(const ItemInternalPtr* items,const Int32* local_ids,Integer n)
  : ItemEnumerator(items,local_ids,n) {}
  ItemEnumerator2T(const ItemInternalEnumerator& rhs)
  : ItemEnumerator(rhs) {}
  ItemEnumerator2T(const ItemEnumerator& rhs)
  : ItemEnumerator(rhs) {}
  ItemEnumerator2T(const ItemVectorViewT<ItemType>& rhs)
  : ItemEnumerator(rhs) {}

 public:

  ItemType operator*() const
    {
      return ItemType(m_items,m_local_ids[m_index]);
    }
  ItemType* operator->() const
  {
    return (ItemType*)&m_items[m_local_ids[m_index]];
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshUnitTest::
_testItemArray()
{
  Int32UniqueArray local_ids;
  IItemFamily* family = mesh()->cellFamily();
  ENUMERATE_CELL(icell,family->allItems()){
    local_ids.add((*icell).localId());
  }
  
  ItemVectorView v(family->itemsInternal(),local_ids);
  ItemVectorViewT<Cell> v2(v);
  Integer z = v2.size();
  info() << "NB CELL=" << z;
  for( Integer i=0; i<z; ++i ){
    Cell c = v2[i];
    info(6) << "CELL =" << ItemPrinter(c);
  }
  ENUMERATE_CELLZ(icell,v2){
    Cell c = *icell;
    info(6) << "CELL =" << ItemPrinter(c);
    for( NodeEnumerator inode(icell->nodes()); inode.hasNext(); ++inode ){
      info(7) << "NODE =" << ItemPrinter(*inode);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshUnitTest::
_testProjection()
{
  GeometricUtilities::QuadMapping mapping;
  Real3 point;
  Real3 uvw;
  bool v = false;
  Real3 a[4];
#if 0
  mapping.m_pos[0] = (1.09457822497375,1.36432427695779,-0.380286258927967);
  mapping.m_pos[1] = (1.20134150773065,1.27621011028963,-0.363546544084165);
  mapping.m_pos[2] = (1.21554570376586,1.30122221118225,-0.182671960715687);
  mapping.m_pos[3] = (1.10818506119729,1.39266571127366,-0.191070818885339);
  Real3 point(1.09457822497375,1.36432427695779,-0.380286258927967);
  Real3 uvw;
  bool v = mapping.cartesianToIso(point,uvw,traceMng());
  info() << "** VALUE = " << uvw;
#endif

  a[0] = Real3(1.09457822497375,1.36432427695779,-0.380286258927967);
  a[1] = Real3(1.20134150773065,1.27621011028963,-0.363546544084165);
  a[2] = Real3(1.21554570376586,1.30122221118225,-0.182671960715687);
  a[3] = Real3(1.10818506119729,1.39266571127366,-0.191070818885339);
  point = Real3(1.14802467452944,1.32021370596031,-0.371906240050006);
  for( Integer i=0; i<4; ++i )
    mapping.m_pos[i] = a[i];
  v = mapping.cartesianToIso2(point,uvw,traceMng());
  info() << "** VALUE = " << uvw << ' ' << v;

  a[0] = Real3(1.09457822497375,1.36432427695779,0.0);
  a[1] = Real3(1.20134150773065,1.27621011028963,0.0);
  a[2] = Real3(1.21554570376586,1.30122221118225,0.0);
  a[3] = Real3(1.10818506119729,1.39266571127366,0.0);
  point = Real3(1.14802467452944,1.32021370596031,0.0);
  for( Integer i=0; i<4; ++i )
    mapping.m_pos[i] = a[i];
  v = mapping.cartesianToIso(point,uvw,traceMng());
  info() << "** VALUE = " << uvw << ' ' << v;
  //Real a1 = 0.0;
  //Real a2 = 0.0;
  //Real a3 = 0.0;
  //StackTrace* vt = 0;
  //info() << vt->toString() << '\n';
  //a1 = a[0].x / a[0].z;
  //info() << " A1=" << a1;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshUnitTest::
_testMD5()
{
  info() << " MD5 TEST";
  Arcane::MD5HashAlgorithm md5;
  ByteUniqueArray output;
  ByteUniqueArray input;
  String s;
  String ref;

  // Calcul le md5 pour un tableau vide
  input.clear();
  output.clear();
  md5.computeHash(input,output);
  s = Convert::toHexaString(output);
  ref = "d41d8cd98f00b204e9800998ecf8427e";
  info() << " S=" << s;
  if (s!=ref)
    fatal() << "Bad value for empty array ref=" << ref << " new=" << s;

  input.clear();
  output.clear();
  input.add('a');
  md5.computeHash(input,output);
  s = Convert::toHexaString(output);
  ref = "0cc175b9c0f1b6a831c399e269772661";
  info() << " S=" << s;
  if (s!=ref)
    fatal() << "Bad value for simple array ref=" << ref << " new=" << s;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vérifie que les entités appartenant à plusieurs sous-domaine
 * sont bien marquées avec le flag II_Shared.
 */
void MeshUnitTest::
_testSharedItems()
{
  info() << "Checking if shared items are marked";
  // Devra contenir à combien de sous-domaine appartient la maille.
  VariableCellInt32 var_counter(VariableBuildInfo(mesh(),"CellCounter"));
  var_counter.fill(1);
  IItemFamily* cell_family = mesh()->cellFamily();
  cell_family->reduceFromGhostItems(var_counter.variable(),Parallel::ReduceSum);
  Int64 nb_shared = 0;
  ENUMERATE_ITEM(iitem,cell_family->allItems()){
    Item item = *iitem;
    Int32 counter = var_counter[iitem];
    if (counter>1)
      ++nb_shared;
    if (counter>1 && !item.isShared())
      ARCANE_FATAL("Item not marked shared item={0} counter={1} is_shared={2}",
                   ItemPrinter(item),counter,item.isShared());
    if (counter==1 && item.isShared())
      ARCANE_FATAL("Item marked shared item={0} counter={1} is_shared={2}",
                   ItemPrinter(item),counter,item.isShared());
  }
  info() << "NbSharedItesm=" << nb_shared;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshUnitTest::
_testAdditionalMeshes()
{
  // Test la lecture de maillages additionnels avec un IParallelMng
  // séquentiel même si on est en parallèle.
  ConstArrayView<String> additional_meshes = options()->additionalMesh.view();
  Integer nb_mesh = additional_meshes.size();
  if (nb_mesh==0)
    return;
  MeshReaderMng mesh_reader_mng(subDomain());
  Integer mesh_index = 0;
  for( String mesh_file_name : additional_meshes ){
    info() << "Reading mesh with file_name " << mesh_file_name;
    String mesh_name = String("TestSequentialMesh") + mesh_index;
    ++mesh_index;
    IMesh* mesh = mesh_reader_mng.readMesh(mesh_name,mesh_file_name);
    info() << "Success in reading mesh named '" << mesh->name() << "'";
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshUnitTest::
_testCustomMeshTools()
{
#ifdef ARCANE_HAS_CUSTOM_MESH_TOOLS
  // Test dépendance outillage externe pour maillage custom (ex polyédrique)
  Neo::Mesh mesh{"test_mesh"};
  info() << "Neo::Mesh{" << mesh.name() << "}";
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshUnitTest::
_testAdditionnalConnectivity()
{
  info() << A_FUNCINFO;
  // Créé une connectivité maille->face contenant pour chaque mailles la liste
  // des faces n'étant pas à la frontière: il s'agit donc des faces qui ont
  // deux mailles connectées.
  IItemFamily* cell_family = mesh()->cellFamily();
  IItemFamily* face_family = mesh()->faceFamily();
  CellGroup cells = cell_family->allItems();
  // NOTE: l'objet est automatiquement détruit par le maillage
  auto* cn = new mesh::IncrementalItemConnectivity(cell_family,face_family,"CellNoBoundaryFace");
  ENUMERATE_CELL(icell,cells){
    Cell cell = *icell;
    Integer nb_face = cell.nbFace();
    cn->notifySourceItemAdded(cell);
    for( Integer i=0; i<nb_face; ++i ){
      Face face = cell.face(i);
      if (face.nbCell()==2)
        cn->addConnectedItem(cell,face);
    }
  }

  IndexedCellFaceConnectivityView cn_view(cn->connectivityView());
  Int64 total_face_lid = 0;
  ENUMERATE_(Cell,icell,cells){
    for( FaceLocalId face : cn_view.faces(icell) )
      total_face_lid += face.localId();
  }
  info() << "TOTAL_NB_FACE = " << total_face_lid;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshUnitTest::
_testShrinkGroups()
{
  info() << A_FUNCINFO;
  mesh_utils::printMeshGroupsMemoryUsage(mesh(),1);
  mesh_utils::shrinkMeshGroups(mesh());
  mesh_utils::printMeshGroupsMemoryUsage(mesh(),1);
  Int64 total = mesh_utils::printMeshGroupsMemoryUsage(mesh(),0);
  info() << "TotalMemoryForGroups=" << total;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshUnitTest::
_testDeallocateMesh()
{
  info() << A_FUNCINFO;
  Integer nb_deallocate = 10;
  IPrimaryMesh* pmesh = mesh()->toPrimaryMesh();
  // TODO: Utiliser un service qui implémente IMeshBuilder au lieu de IMeshReader
  ServiceBuilder<IMeshReader> sbu(subDomain());
  String file_names[3] = { "tied_interface_1.vtk", "sphere_tied_1.vtk", "sphere_tied_2.vtk" };
  for( Integer i=0; i<nb_deallocate; ++i ){
    info() << "DEALLOCATE I=" << i;
    pmesh->deallocate();
    auto mesh_io(sbu.createReference("VtkLegacyMeshReader",SB_AllowNull));
    mesh_io->readMeshFromFile(pmesh,XmlNode{},file_names[i%3],String(),true);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace ArcaneTest

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
