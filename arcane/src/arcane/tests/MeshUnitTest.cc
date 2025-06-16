// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshUnitTest.cc                                             (C) 2000-2025 */
/*                                                                           */
/* Service de test du maillage.                                              */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/StringBuilder.h"
#include "arcane/utils/List.h"
#include "arcane/utils/ArithmeticException.h"
#include "arcane/utils/ValueChecker.h"
#include "arcane/utils/TestLogger.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/SHA3HashAlgorithm.h"

#include "arcane/core/BasicUnitTest.h"
#include "arcane/core/AbstractItemOperationByBasicType.h"
#include "arcane/core/IMeshWriter.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/MeshUtils.h"
#include "arcane/core/IMesh.h"
#include "arcane/core/IMeshSubMeshTransition.h"
#include "arcane/core/IMeshModifier.h"
#include "arcane/core/ITiedInterface.h"
#include "arcane/core/IItemFamily.h"
#include "arcane/core/ItemInfoListView.h"
#include "arcane/core/IItemConnectivityInfo.h"
#include "arcane/core/ItemPrinter.h"
#include "arcane/core/IMeshUtilities.h"
#include "arcane/core/IVariableMng.h"
#include "arcane/core/Directory.h"
#include "arcane/core/IVariableSynchronizer.h"
#include "arcane/core/VariableCollection.h"
#include "arcane/core/ServiceBuilder.h"
#include "arcane/core/IParallelReplication.h"
#include "arcane/core/IndexedItemConnectivityView.h"
#include "arcane/core/IIndexedIncrementalItemConnectivityMng.h"
#include "arcane/core/IIndexedIncrementalItemConnectivity.h"
#include "arcane/core/IIncrementalItemConnectivity.h"
#include "arcane/core/ServiceFinder2.h"
#include "arcane/core/SerializeBuffer.h"
#include "arcane/core/IMeshPartitioner.h"
#include "arcane/core/IMainFactory.h"
#include "arcane/core/Properties.h"
#include "arcane/core/ItemPairGroup.h"
#include "arcane/core/ItemPairEnumerator.h"
#include "arcane/core/ItemPairGroupBuilder.h"
#include "arcane/core/IPostProcessorWriter.h"
#include "arcane/core/ItemVectorView.h"
#include "arcane/core/ItemVector.h"
#include "arcane/core/GeometricUtilities.h"
#include "arcane/core/IMeshReader.h"
#include "arcane/core/IXmlDocumentHolder.h"
#include "arcane/core/IIOMng.h"
#include "arcane/core/MeshReaderMng.h"
#include "arcane/core/UnstructuredMeshConnectivity.h"
#include "arcane/core/MeshKind.h"
#include "arcane/core/MeshEvents.h"
#include "arcane/core/BlockIndexList.h"
#include "arcane/core/Connectivity.h"
#include "arcane/core/IMeshUniqueIdMng.h"

#include "arcane/tests/MeshUnitTest_axl.h"

#ifdef ARCANE_HAS_POLYHEDRAL_MESH_TOOLS
#include "neo/Mesh.h"
#endif

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

    explicit CountOperationByBasicType(ITraceMng* m)
    : TraceAccessor(m)
    {}

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

  explicit MeshUnitTest(const ServiceBuildInfo& cb);

 public:

  void buildInitializeTest() override;
  void executeTest() override;

 private:

  void _dumpMesh();
  void _dumpTiedInterfaces();
  void _testGroups();
  void _dumpComputeFaceGroupNormal();
  void _dumpComputeNodeGroupDirection();
  void _testItemAdjacency();
  void _testItemAdjacency2();
  void _testItemAdjacency3();
  void _testItemPartialAdjacency();
  void _testVariableWriter();
  void _testItemArray();
  void _testProjection();
  void _dumpConnections();
  void _partitionMesh(Int32 nb_part);
  void _testUsedVariables();
  void _dumpConnectivityInfos(IItemConnectivityInfo* cell_family,
                              IItemConnectivityInfo* face_family,
                              IItemConnectivityInfo* node_family);
  void _testSharedItems();
  void _testVisitors();
  template<typename ItemKind,typename SubItemKind>
  void _testItemAdjacency(ItemGroupT<ItemKind> items, ItemGroupT<SubItemKind> subitems,
                          eItemKind link_kind);
  void _testAdditionalMeshes();
  void _testNullItem();
  void _testCustomMeshTools();
  void _testAdditionnalConnectivity();
  void _testShrinkGroups();
  void _testDeallocateMesh();
  void _testUnstructuredConnectivities();
  void _testSortedNodeFaces();
  void _testFaces();
  void _testItemVectorView();
  void _logMeshInfos();
  void _testComputeLocalIdPattern();
  void _testGroupsAsBlocks();
  void _testCoherency();
  void _testFindOneItem();
  void _testEvents();
  void _testNodeNodeViaEdgeConnectivity();
  void _testBoundaryNodeNodeViaEdgeConnectivity();
  void _testComputeOwnersDirect();
  void _testLocalIdsFromConnectivity();
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

void MeshUnitTest::
buildInitializeTest()
{
  if (options()->createEdges()) {
    Connectivity c(mesh()->connectivity());
    c.enableConnectivity(Connectivity::CT_HasEdge);
  }
  if (!options()->compactMesh()) {
    Properties* p = mesh()->properties();
    p->setBool("compact",false);
    p->setBool("compact-after-allocate",false);
  }
  if (options()->generateUidFromNodesUid()) {
    mesh()->meshUniqueIdMng()->setUseNodeUniqueIdToGenerateEdgeAndFaceUniqueId(true);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshUnitTest::
executeTest()
{
  bool do_compute_owners_direct = options()->computeOwnersDirect();
  if (do_compute_owners_direct)
    _testComputeOwnersDirect();

  bool do_sort_faces_and_edges = options()->testSortNodeFacesAndEdges();
  if (do_sort_faces_and_edges) {
    mesh()->nodeFamily()->properties()->setBool("sort-connected-faces-edges",true);
    mesh()->modifier()->endUpdate();
  }

  CountOperationByBasicType op(traceMng());
  info() << "ItemTypeMng::singleton() = " << ItemTypeMng::singleton();
  info() << "Infos sur AllCells:";
  allCells().applyOperation(&op);
  info() << "Infos sur AllFaces:";
  allFaces().applyOperation(&op);
  info() << "Infos sur AllNodes:";
  allNodes().applyOperation(&op);
  _logMeshInfos();
  if (options()->writeMesh())
    _dumpMesh();
  _testNullItem();
  _dumpTiedInterfaces();
  _dumpComputeFaceGroupNormal();
  _dumpComputeNodeGroupDirection();
  if (do_sort_faces_and_edges)
    _testSortedNodeFaces();
  _testGroups();
  if (options()->testAdjency()){
    _testItemAdjacency();
    _testItemAdjacency2();
    _testItemAdjacency3();
    _testItemPartialAdjacency();
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
  _testItemVectorView();
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
  _testUsedVariables();
  _testAdditionalMeshes();
  _testCustomMeshTools();
  _testAdditionnalConnectivity();
  _testShrinkGroups();
  _testFaces();
  _testUnstructuredConnectivities();
  if (options()->testDeallocateMesh())
    _testDeallocateMesh();
  _testComputeLocalIdPattern();
  _testGroupsAsBlocks();
  _testCoherency();
  _testFindOneItem();
  _testEvents();
  if (options()->createEdges()) {
    // Appelle 2 fois la méthode pour vérifier que le recalcul est correct.
    _testNodeNodeViaEdgeConnectivity();
    _testNodeNodeViaEdgeConnectivity();
  }
  _testBoundaryNodeNodeViaEdgeConnectivity();
  if (options()->checkLocalIdsFromConnectivity())
    _testLocalIdsFromConnectivity();
}

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

  NodeGroup null_node_group;
  ENUMERATE_(Node,inode,null_node_group){
    Node node = *inode;
    info() << "NODE FOR null group: " << node.localId();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshUnitTest::
_logMeshInfos()
{
  TestLogger::stream() << "NbNode=" << mesh()->nodeFamily()->nbItem() << "\n";
  TestLogger::stream() << "NbEdge=" << mesh()->edgeFamily()->nbItem() << "\n";
  TestLogger::stream() << "NbFace=" << mesh()->faceFamily()->nbItem() << "\n";
  TestLogger::stream() << "NbCell=" << mesh()->cellFamily()->nbItem() << "\n";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshUnitTest::
_dumpMesh()
{
  ServiceBuilder<IMeshWriter> sbu(subDomain());
  String write_service_name = options()->writeMeshServiceName();
  auto mesh_io(sbu.createReference(write_service_name,SB_AllowNull));
  IParallelMng* pm = subDomain()->parallelMng();
  bool is_parallel = pm->isParallel();
  Int32 my_rank = pm->commRank();
  Directory base_path(subDomain()->exportDirectory());
  StringBuilder sorted_file_name(options()->outputFile());
  sorted_file_name += "-sorted";
  if (is_parallel){
    sorted_file_name += "-";
    sorted_file_name += my_rank;
  }
  mesh_utils::writeMeshInfosSorted(mesh(), base_path.file(sorted_file_name));
  StringBuilder file_name_b(options()->outputFile());
  if (is_parallel){
    file_name_b += "-";
    file_name_b += my_rank;
  }
  String file_name(file_name_b.toString());
  mesh_utils::writeMeshInfos(mesh(), base_path.file(file_name));
  String connectivity_file = base_path.file(file_name + ".xml");
  mesh_utils::writeMeshConnectivity(mesh(), connectivity_file);

  // Relit le fichier de connectivité et vérifie que la checksum est bonne
  String connectivity_checksum;
  if (pm->isParallel()) {
    Int32 nb_checksum = options()->connectivityFileChecksumParallel.size();
    if (nb_checksum > my_rank)
      connectivity_checksum = options()->connectivityFileChecksumParallel[my_rank];
  }
  else
    connectivity_checksum = options()->connectivityFileChecksum();
  if (!connectivity_checksum.null()) {
    SHA3_256HashAlgorithm hash_algo;
    UniqueArray<std::byte> bytes;
    if (platform::readAllFile(connectivity_file, false, bytes))
      ARCANE_FATAL("Can not read file '{0}'", connectivity_file);
    UniqueArray<Byte> hash_bytes;
    hash_algo.computeHash64(bytes, hash_bytes);
    String x = Convert::toHexaString(hash_bytes);
    info() << "SHA=" << x;
    if (x != connectivity_checksum)
      ARCANE_FATAL("Bad connectivity checksum x={0} expected={1}", x, connectivity_checksum);
  }

  if (mesh_io.get()){
    if (write_service_name=="Lima")
      file_name = file_name + ".unf";
    mesh_io->writeMeshToFile(mesh(), base_path.file(file_name));
  }

  IItemFamily* cell_family = mesh()->cellFamily();
  info() << "Local connectivity infos:";
  _dumpConnectivityInfos(cell_family->localConnectivityInfos(),
                         mesh()->faceFamily()->localConnectivityInfos(),
                         mesh()->nodeFamily()->localConnectivityInfos());

  info() << "Global connectivity infos:";
  _dumpConnectivityInfos(cell_family->globalConnectivityInfos(),
                         mesh()->faceFamily()->globalConnectivityInfos(),
                         mesh()->nodeFamily()->globalConnectivityInfos());

  MeshUtils::dumpSynchronizerTopologyJSON(cell_family->allItemsSynchronizer(), base_path.file("sync_topology.json"));
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
  std::set<String> var_names;
  for( VariableList::Enumerator i(vars); ++i; ){
    IVariable* var = *i;
    info() << "name=" << var->name();
    var_names.insert(var->name());
  }
  auto read_var_check_names = options()->checkReadProperty.view();
  for (auto read_var_check_name : read_var_check_names) {
    if (var_names.find(read_var_check_name) == var_names.end()) {
      ARCANE_FATAL("Error while reading mesh {0}. Variable {1} is not kept after reading.", mesh()->name(), read_var_check_name);
    }
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
        ENUMERATE_CONNECTED_(Node,inode,face,nodes()){
          info() << "Master face node uid=" << inode->uniqueId();
        }
        for( Integer zz=0, zs=tied_nodes[iface.index()].size(); zz<zs; ++zz ){
          const TiedNode& tn = tied_nodes[iface.index()][zz];
          nodes_in_master_face.insert(tn.node().uniqueId());
          if (zz<20){
            info() << " node_uid=" << tn.node().uniqueId()
                   << " iso=" << tn.isoCoordinates()
                   << " kind=" << tn.node().kind();
          }
        }
        ENUMERATE_CONNECTED_(Node,inode,face,nodes()){
          if (nodes_in_master_face.find(inode->uniqueId())==nodes_in_master_face.end()){
            ++nb_error;
            if (nb_error<max_print_error)
              error() << "node in master face not in slave node list uid=" << inode->uniqueId();
          }
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
  {
    // Vérifie que le groupe nul est trié
    CellGroup null_group;
    if (!null_group.checkIsSorted())
      ARCANE_FATAL("Null group is not sorted");
  }

  IItemFamily* item_family = mesh()->cellFamily();
  CellGroup all_items = item_family->allItems();
  if (!all_items.checkIsSorted())
    ARCANE_FATAL("AllItems group is not sorted");

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
        ARCANE_FATAL("Group should not be sorted");
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
_testItemAdjacency2()
{
  FaceFaceGroup ad_list(allFaces(),outerFaces(),IK_Node);
  info() << " COMPUTE ITEM ADJENCY LIST2!";
  
  std::set<Integer> boundary_set;
  ENUMERATE_FACE(iface,outerFaces())
    boundary_set.insert(iface->localId());

  // Test l'accès à la valeur via l'itérateur
  VariableFaceInt32 face_id(VariableBuildInfo(mesh(),"FaceId"));
  ENUMERATE_ITEMPAIR(Face,Face,iface,ad_list){
    face_id[iface] = iface.itemLocalId();
  }

  Int64 total_uid = 0;
  ENUMERATE_ITEMPAIR(Face,Face,iface,ad_list){
    if (face_id[iface]!=iface.itemLocalId())
      ARCANE_FATAL("Bad value for variable v={0} expected={1}",face_id[iface],iface.itemLocalId());
    
    ENUMERATE_SUB_ITEM(Face,isubface,iface){
      const Face& subface = *isubface;
      total_uid += subface.uniqueId().asInt64();
      // info() << " SUBITEM #" << isubface.index() << " : " << subface.localId()
      //<< " is_boundary=" << (boundary_set.find(subface.localId()) != boundary_set.end());
      if (boundary_set.find(subface.localId()) == boundary_set.end())
        fatal() << "Non boundary face [" << subface.localId() << " ] found";
    }
  }
  info() << " TOTAL uid=" << total_uid;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshUnitTest::
_testItemAdjacency3()
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
_testItemAdjacency(ItemGroupT<ItemKind> items, ItemGroupT<SubItemKind> subitems,
                   eItemKind link_kind)
{
  ItemPairGroupT<ItemKind, SubItemKind> ad_list(items, subitems, link_kind);
  info() << " COMPUTE ITEM ADJENCY LIST link_kind=" << link_kind
         << " items=" << items.name() << " sub_items=" << subitems.name()
         << " nb_item=" << items.size() << " nb_sub_item=" << subitems.size()
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
_testItemAdjacency()
{
  _testItemAdjacency(allCells(), allCells(), IK_Node);
  _testItemAdjacency(allCells(), allCells(), IK_Face);

  _testItemAdjacency(allNodes(), allNodes(), IK_Cell);
  _testItemAdjacency(allNodes(), allNodes(), IK_Face);
  _testItemAdjacency(allNodes(), allNodes(), IK_Edge);

  _testItemAdjacency(allFaces(), allCells(), IK_Node);

  _testItemAdjacency(allFaces(), allFaces(), IK_Node);
  _testItemAdjacency(allFaces(), allFaces(), IK_Edge);
  _testItemAdjacency(allFaces(), allFaces(), IK_Cell);

  _testItemAdjacency(allCells(), allFaces(), IK_Face);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshUnitTest::
_testItemPartialAdjacency()
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
  Directory out_dir(subDomain()->exportDirectory(), dir_name);
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
  MeshUtils::visitGroups(item_family, func);
  vc.areEqual(nb_group,item_family->groups().count(),"Bad number of group for cell family");
  info() << "Nb group=" << nb_group;

  info() << "TEST Visit IMesh";
  nb_group = 0;
  Integer nb_expected_group = 0;
  for( IItemFamilyCollection::Enumerator ifamily(mesh()->itemFamilies()); ++ifamily; ){
    nb_expected_group += (*ifamily)->groups().count();
  }
  MeshUtils::visitGroups(mesh(), func);
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
    MeshUtils::visitGroups(mesh(), check_sorted_func);
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
    return ItemType(ItemEnumerator::operator*());
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class MyCellClass
: public Cell
{
 public:
  //! Construit une référence à l'entité \a abase
  explicit MyCellClass(Item aitem) : Cell(aitem) {}
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
  
  // Pour tester le constructeur par défaut.
  [[maybe_unused]] ItemLocalIdToItemConverter empty_converter;

  ItemVectorView v(family,local_ids);
  ItemInfoListView cells_info_view(family);
  ItemLocalIdToItemConverter cells_local_id_converter(family);
  ItemVectorViewT<Cell> v2(v);
  Integer z = v2.size();
  info() << "NB CELL=" << z;
  for( Integer i=0; i<z; ++i ){
    Cell c = v2[i];
    Item c2 = cells_info_view[local_ids[i]];
    Item c3 = cells_local_id_converter[local_ids[i]];
    Item c4 = cells_local_id_converter[ItemLocalId(local_ids[i])];
    if (c2!=c3)
      ARCANE_FATAL("Bad same item item2={0} item3={1}", ItemPrinter(c2), ItemPrinter(c3));
    if (c2!=c4)
      ARCANE_FATAL("Bad same item item2={0} item4={1}", ItemPrinter(c2), ItemPrinter(c4));
    info(6) << "CELL =" << ItemPrinter(c);
    if (c2.uniqueId()!=c.uniqueId())
      ARCANE_FATAL("Not same uniqueId() (1) uid1={0} uid2={1}",c.uniqueId(),c2.uniqueId());
    if (cells_info_view.uniqueId(local_ids[i])!=c.uniqueId())
      ARCANE_FATAL("Not same uniqueId() (2) uid1={0} uid2={1}",cells_info_view.uniqueId(local_ids[i]),c.uniqueId());
    if (cells_info_view.owner(local_ids[i])!=c.owner())
      ARCANE_FATAL("Not same owner() owner1={0} owner2={1}",cells_info_view.owner(local_ids[i]),c.owner());
    if (cells_info_view.typeId(local_ids[i])!=c.type())
      ARCANE_FATAL("Not same typeId() type1={0} type2={1}",cells_info_view.typeId(local_ids[i]),c.type());
  }
  ENUMERATE_CELLZ(icell,v2){
    Cell c = *icell;
    info(6) << "CELL =" << ItemPrinter(c);
    for( NodeEnumerator inode(c.nodes()); inode.hasNext(); ++inode ){
      info(7) << "NODE =" << ItemPrinter(*inode);
    }
  }

  VariableCellInt32 var_counter(VariableBuildInfo(mesh(),"CellCounter"));
  Int64 total = 0;
  ENUMERATE_(MyCellClass,icell,v2){
    const MyCellClass& c = *icell;
    MyCellClass c2 = *icell;
    info(6) << "CELL =" << ItemPrinter(c);
    var_counter[icell] = c.localId();
    total += var_counter[c] + var_counter[c2];
    for( NodeEnumerator inode(c.nodes()); inode.hasNext(); ++inode ){
      info(7) << "NODE =" << ItemPrinter(*inode);
    }
  }
  info() << "TOTAL=" << total;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshUnitTest::
_testItemVectorView()
{
  ValueChecker vc(A_FUNCINFO);
  IItemFamily* cell_family = mesh()->cellFamily();
  CellGroup cells = cell_family->allItems();
  if (cells.size()<13)
    return;
  UniqueArray<Int32> local_ids;
  Int64 total_ref_uid = 0;
  ENUMERATE_(Cell,icell,cells){
    local_ids.add(icell.itemLocalId());
    total_ref_uid += icell.itemLocalId();
  }
  CellVector cell_vector(cell_family,local_ids);
  CellVectorView cell_vector_view(cell_vector);
  {
    Int64 total_uid_cell_vector = 0;
    ENUMERATE_(Cell,icell,cell_vector){
      total_uid_cell_vector += icell.itemLocalId();
    }
    vc.areEqual(total_ref_uid,total_uid_cell_vector,"SameCellVector");
  }
  {
    Int64 total_uid_cell_vector_view = 0;
    ENUMERATE_(Cell,icell,cell_vector_view){
      total_uid_cell_vector_view += icell.itemLocalId();
    }
    vc.areEqual(total_ref_uid,total_uid_cell_vector_view,"SameCellVectorView");
  }

  auto cell_iter = cell_vector_view.begin();
  auto cell_iter2 = cell_iter;
  ++cell_iter2;
  vc.areEqual(cell_vector[1],*cell_iter2,"SameCell1");
  auto cell_iter3 = cell_iter + 5;
  vc.areEqual(cell_vector[5],*cell_iter3,"SameCell2");
  auto cell_iter4 = cell_iter3 - 2;
  vc.areEqual(cell_vector[3],*cell_iter4,"SameCell3");
  auto diff1 = cell_iter4 - cell_iter3;
  CellVectorView::const_iterator::difference_type wanted_diff1 = -2;
  vc.areEqual(wanted_diff1,diff1,"SameDiff1");
  auto diff2 = cell_iter3 - cell_iter4;
  CellVectorView::const_iterator::difference_type wanted_diff2 = 2;
  vc.areEqual(wanted_diff2,diff2,"SameDiff2");

  auto cell_iter5 = std::find(cell_vector_view.begin(),cell_vector_view.end(),cell_vector[12]);
  auto diff3 = cell_iter5 - cell_vector_view.begin();
  CellVectorView::const_iterator::difference_type wanted_diff3 = 12;
  vc.areEqual(wanted_diff3,diff3,"SameDiff3");

  vc.areEqual(cell_vector[12],*cell_iter5,"SameCell4");

  {
    UniqueArray<Int32> int32_lid1;
    UniqueArray<Int32> int32_lid2;
    cells.view().fillLocalIds(int32_lid1);
    cells.view().indexes().fillLocalIds(int32_lid2);
    Int32 index = 0;
    ENUMERATE_(Cell,icell,cells){
      Cell cell = *icell;
      if (cell.localId()!=int32_lid1[index])
        ARCANE_FATAL("Bad 1 local id v={0} expected={1}",cell.localId(),int32_lid1[index]);
      if (cell.localId()!=int32_lid2[index])
        ARCANE_FATAL("Bad 2 local id v={0} expected={1}",cell.localId(),int32_lid2[index]);
      ++index;
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
    Int32 counter = var_counter[item.toCell()];
    if (counter>1)
      ++nb_shared;
    if (counter>1 && !item.isShared())
      ARCANE_FATAL("Item not marked shared item={0} counter={1} is_shared={2}",
                   ItemPrinter(item),counter,item.isShared());
    if (counter==1 && item.isShared())
      ARCANE_FATAL("Item marked shared item={0} counter={1} is_shared={2}",
                   ItemPrinter(item),counter,item.isShared());
  }
  info() << "NbSharedItems=" << nb_shared;
  TestLogger::stream() << "NbSharedItems=" << nb_shared << "\n";
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
#ifdef ARCANE_HAS_POLYHEDRAL_MESH_TOOLS
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
  ValueChecker vc(A_FUNCINFO);

  // Créé une connectivité maille <-> face contenant pour chaque maille la liste
  // des faces n'étant pas à la frontière : il s'agit donc des faces qui ont
  // deux mailles connectées.
  IItemFamily* cell_family = mesh()->cellFamily();
  IItemFamily* face_family = mesh()->faceFamily();
  CellGroup cells = cell_family->allItems();
  // NOTE: l'objet est automatiquement détruit par le maillage
  auto idx_cn = mesh()->indexedConnectivityMng()->findOrCreateConnectivity(cell_family,face_family,"CellNoBoundaryFace");
  auto* cn = idx_cn->connectivity();
  ENUMERATE_CELL(icell,cells){
    Cell cell = *icell;
    Integer nb_face = cell.nbFace();
    for( Integer i=0; i<nb_face; ++i ){
      Face face = cell.face(i);
      if (face.nbCell()==2)
        cn->addConnectedItem(cell,face);
    }
  }

  IndexedCellFaceConnectivityView cn_view(idx_cn->view());
  Int64 total_face_lid = 0;
  ENUMERATE_(Cell,icell,cells){
    for( FaceLocalId face : cn_view.faces(icell) )
      total_face_lid += face.localId();
    vc.areEqual(cn_view.faces(icell),cn_view.faceIds(icell),"SameArray");
    Int32 n = cn_view.nbFace(icell);
    vc.areEqual(n,cn_view.faceIds(icell).size(),"SameSize");
    for( Int32 i=0; i<n; ++i )
      vc.areEqual(cn_view.faceId(icell,i),cn_view.faces(icell)[i],"SameItem");
  }
  info() << "TOTAL_NB_FACE = " << total_face_lid;

  auto idx_cn2 = mesh()->indexedConnectivityMng()->findOrCreateConnectivity(cell_family,face_family,"CellNoBoundaryFace");
  auto idx_cn3 = mesh()->indexedConnectivityMng()->findConnectivity("CellNoBoundaryFace");

  if (idx_cn2->connectivity()!=cn)
    ARCANE_FATAL("Bad findOrCreateConnectivity");
  if (idx_cn3->connectivity()!=cn)
    ARCANE_FATAL("Bad findConnectivity");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshUnitTest::
_testUnstructuredConnectivities()
{
  ValueChecker vc(A_FUNCINFO);

  UnstructuredMeshConnectivityView connectivity_view;
  connectivity_view.setMesh(this->mesh());

  {
    // Teste Cell->Face
    IndexedCellFaceConnectivityView icv(connectivity_view.cellFace());
    ENUMERATE_(Cell,icell,allCells()){
      Cell cell = *icell;
      // Vérifie la cohérence entre les méthodes
      auto f1 = icv.faces(icell);
      auto f2 = icv.faceIds(icell);
      auto f3 = cell.faceIds();
      vc.areEqual(f1,f2,"SameFaceArray1");
      vc.areEqual(f1,f3,"SameFaceArray2");
      Int32 n = icv.nbFace(icell);
      vc.areEqual(n,icv.faceIds(icell).size(),"SameFaceSize");
      for( Int32 i=0; i<n; ++i )
        vc.areEqual(icv.faceId(icell,i),icv.faces(icell)[i],"SameFaceItem");
    }
  }

  {
    // Teste Cell->Node
    IndexedCellNodeConnectivityView icv(connectivity_view.cellNode());
    ENUMERATE_(Cell,icell,allCells()){
      Cell cell = *icell;
      // Vérifie la cohérence entre les méthodes
      auto f1 = icv.nodes(icell);
      auto f2 = icv.nodeIds(icell);
      auto f3 = cell.nodeIds();
      vc.areEqual(f1,f2,"SameNodeArray1");
      vc.areEqual(f1,f3,"SameNodeArray2");
      Int32 n = icv.nbNode(icell);
      vc.areEqual(n,icv.nodeIds(icell).size(),"SameNodeSize");
      for( Int32 i=0; i<n; ++i )
        vc.areEqual(icv.nodeId(icell,i),icv.nodes(icell)[i],"SameNodeItem");
    }
  }

  {
    // Teste Cell->Edge
    IndexedCellEdgeConnectivityView icv(connectivity_view.cellEdge());
    ENUMERATE_(Cell,icell,allCells()){
      Cell cell = *icell;
      // Vérifie la cohérence entre les méthodes
      auto f1 = icv.edges(icell);
      auto f2 = icv.edgeIds(icell);
      auto f3 = cell.edgeIds();
      vc.areEqual(f1,f2,"SameEdgeArray1");
      vc.areEqual(f1,f3,"SameEdgeArray2");
      Int32 n = icv.nbEdge(icell);
      vc.areEqual(n,icv.edgeIds(icell).size(),"SameEdgeSize");
      for( Int32 i=0; i<n; ++i )
        vc.areEqual(icv.edgeId(icell,i),icv.edges(icell)[i],"SameEdgeItem");
    }
  }

  {
    // Teste Node->Cell
    IndexedNodeCellConnectivityView icv(connectivity_view.nodeCell());
    ENUMERATE_(Node,inode,allNodes()){
      Node node = *inode;
      // Vérifie la cohérence entre les méthodes
      auto f1 = icv.cells(inode);
      auto f2 = icv.cellIds(inode);
      auto f3 = node.cellIds();
      vc.areEqual(f1,f2,"SameCellArray1");
      vc.areEqual(f1,f3,"SameCellArray2");
      Int32 n = icv.nbCell(inode);
      vc.areEqual(n,icv.cellIds(inode).size(),"SameCellSize");
      for( Int32 i=0; i<n; ++i )
        vc.areEqual(icv.cellId(inode,i),icv.cells(inode)[i],"SameCellItem");
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vérifie si les faces connectées aux noeuds sont triées.
 */
void MeshUnitTest::
_testSortedNodeFaces()
{
  ValueChecker vc(A_FUNCINFO);

  UnstructuredMeshConnectivityView connectivity_view;
  connectivity_view.setMesh(this->mesh());

  ENUMERATE_(Node,inode,allNodes()){
    Node node = *inode;
    Face previous_face;
    for( Face face : node.faces() ){
      if (!previous_face.null()){
        if (previous_face.uniqueId()>face.uniqueId())
          ARCANE_FATAL("Connected faces are not sorted node={0} previous_face={1} current_face={2}",
                       ItemPrinter(node), ItemPrinter(previous_face), ItemPrinter(face));
      }
      previous_face = face;
    }
  }
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
_testFaces()
{
  info() << A_FUNCINFO;
  ValueChecker vc(A_FUNCINFO);

  ENUMERATE_(Face,iface,allFaces()){
    Face face = *iface;
    Int32 flags = face.itemBase().flags();
    vc.areEqual(face.backCell().itemLocalId(),face.backCellId(),"BackCell");
    vc.areEqual(face.frontCell().itemLocalId(),face.frontCellId(),"FrontCell");
    vc.areEqual(face.isSubDomainBoundary(), ItemFlags::isSubDomainBoundary(flags), "IsSubDomainBoundary");
    vc.areEqual(!face.backCell().null(), ItemFlags::hasBackCell(flags), "HasBackCell");

    {
      Int32 back_cell_index = ItemFlags::backCellIndex(flags);
      if (back_cell_index < 0)
        vc.areEqual(face.backCell(), Cell(), "BackCellIndex 0");
      else
        vc.areEqual(face.backCell(), face.cell(back_cell_index), "BackCellIndex 1");
    }
    {
      Int32 front_cell_index = ItemFlags::frontCellIndex(flags);
      if (front_cell_index < 0)
        vc.areEqual(face.frontCell(), Cell(), "FrontCellIndex 0");
      else
        vc.areEqual(face.frontCell(), face.cell(front_cell_index), "FrontCellIndex 1");
    }
  }

  ENUMERATE_(Cell,icell,allCells()){
    Cell cell = *icell;
    for( Face face : cell.faces() ){
      vc.areEqual(face.oppositeCell(cell).itemLocalId(),face.oppositeCellId(cell),"OppositeCell1");
    }
    Int32 nb_face = cell.nbFace();
    FaceConnectedListViewType faces = cell.faces();
    for(Int32 i=0; i<nb_face; ++i ){
      vc.areEqual(faces[i].oppositeCell(cell).itemLocalId(),faces[i].oppositeCellId(cell),"OppositeCell2");
    }
    // Teste la compatibilité entre ItemVectorView et ItemConnectedListView.
    // Pour maintenant la compatibilité entre les versions de 3.8+ et antérieures
    // on doit pouvoir convertir un 'ItemConnectedListView' en un 'ItemVectorView'
    // et de même pour les itérateurs
    FaceVectorView faces_as_vector = cell.faces();
    FaceVectorView::const_iterator face_vector_begin2 = cell.faces().begin();
    FaceVectorView::const_iterator face_vector_begin1 = faces_as_vector.begin();
    auto face_begin1 = faces.begin();
    if (face_begin1!=face_vector_begin1)
      ARCANE_FATAL("Bad face1");
    if (face_vector_begin1!=face_vector_begin2)
      ARCANE_FATAL("Bad face2");
  }
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
  for (Integer i = 0; i < nb_deallocate; ++i) {
    info() << "DEALLOCATE I=" << i;
    pmesh->deallocate();
    auto mesh_io(sbu.createReference("VtkLegacyMeshReader", SB_AllowNull));
    mesh_io->readMeshFromFile(pmesh, XmlNode{}, file_names[i % 3], String(), true);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshUnitTest::
_testComputeLocalIdPattern()
{
  MeshUtils::computeConnectivityPatternOccurence(mesh());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshUnitTest::
_testGroupsAsBlocks()
{
  ITraceMng* tm = mesh()->traceMng();
  ValueChecker vc(A_FUNCINFO);
  auto xx = [&](const ItemGroup& group)
  {
    if (group.internal()->parent())
      return;
    Int32 nb_item = group.size();
    if (nb_item==0)
      return;
    BlockIndexList bli;
    BlockIndexListBuilder bli_builder(tm);
    bli_builder.setBlockSizeAsPowerOfTwo(5); // Bloc de 2^5 = 32
    bli_builder.build(bli,group.view().localIds(),group.name());
    UniqueArray<Int32> computed_values;
    bli.fillArray(computed_values);
    vc.areEqualArray(computed_values.constView(),group.view().localIds(),group.name());
  };
  MeshUtils::visitGroups(mesh(), xx);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshUnitTest::
_testCoherency()
{
  MeshKind mk = mesh()->meshKind();
  info() << "MeshKind amr=" << mk.meshAMRKind() << " structure=" << mk.meshStructure();
  bool is_amr = mesh()->isAmrActivated();
  if (is_amr && mk.meshAMRKind()==eMeshAMRKind::None)
    ARCANE_FATAL("AMR incoherence: is_amr==true but eMeshAMRKind == None");
  if (!is_amr && mk.meshAMRKind()!=eMeshAMRKind::None)
    ARCANE_FATAL("AMR incoherence: is_amr==false but eMeshAMRKind != None");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshUnitTest::
_testFindOneItem()
{
  UniqueArray<Int64> unique_ids;
  UniqueArray<Int32> local_ids;
  {
    Int32 index = 0;
    ENUMERATE_(Cell,icell,allCells()){
      if (index>100)
        break;
      Cell cell = *icell;
      unique_ids.add(cell.uniqueId());
      local_ids.add(cell.localId());
      ++index;
    }
  }

  IItemFamily* cell_family = mesh()->cellFamily();

  // Teste l'entité nulle
  {
    Int64 max_uid = 0;
    ENUMERATE_(Cell,icell,allCells()){
      Cell cell = *icell;
      Int64 uid = cell.uniqueId();
      max_uid = math::max(max_uid,uid);
    }

    Cell cell = MeshUtils::findOneItem(cell_family,max_uid+1);
    if (!cell.null())
      ARCANE_FATAL("Expected null cell but found cell={0}",ItemPrinter(cell));
  }

  for( Int32 i=0, n=unique_ids.size(); i<n; ++i ){
    Int64 uid = unique_ids[i];
    ItemUniqueId true_uid(uid);
    Cell cell1 = MeshUtils::findOneItem(cell_family,uid);
    if (cell1.null())
      ARCANE_FATAL("Unexpected null cell");
    if (cell1.localId()!=local_ids[i])
      ARCANE_FATAL("Bad local id cell1={0} expected_lid={1}",ItemPrinter(cell1),local_ids[i]);
    Cell cell2 = MeshUtils::findOneItem(cell_family,true_uid);
    if (cell1!=cell2)
      ARCANE_FATAL("Unexpected different cell cell1={0} cell2={1}",ItemPrinter(cell1),ItemPrinter(cell2));
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshUnitTest::
_testEvents()
{
  // Vérifie que les évènements 'BeginPrepareDump' et 'EndPrepareDump' sont
  // bien lancés.
  EventObserverPool pool;
  bool has_call_begin_prepare_dump = false;
  bool has_call_end_prepare_dump = false;
  auto f1 = [&](const MeshEventArgs&){ has_call_begin_prepare_dump = true; };
  auto f2 = [&](const MeshEventArgs&){ has_call_end_prepare_dump = true; };
  mesh()->eventObservable(eMeshEventType::BeginPrepareDump).attach(pool,f1);
  mesh()->eventObservable(eMeshEventType::EndPrepareDump).attach(pool,f2);
  mesh()->prepareForDump();
  if (!has_call_begin_prepare_dump)
    ARCANE_FATAL("Event BeginPrepareDump has not been called");
  if (!has_call_end_prepare_dump)
    ARCANE_FATAL("Event EndPrepareDump has not been called");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshUnitTest::
_testNodeNodeViaEdgeConnectivity()
{
  info() << "Test: _testNodeNodeViaEdgeConnectivity";
  auto x = Arcane::MeshUtils::computeNodeNodeViaEdgeConnectivity(mesh(), "NodeNodeViaEdge");
  IndexedNodeNodeConnectivityView nn_cv = x->view();

  // Tableau contenant la liste triée des nœuds connectés à un nœud.
  UniqueArray<Int32> ref_cx_nodes;

  ENUMERATE_ (Node, inode, ownNodes()) {
    Node node = *inode;
    Int32 nb_edge = node.nbEdge();
    Int32 nb_connectivity_node = nn_cv.nbNode(node);
    if (nb_edge != nb_connectivity_node)
      ARCANE_FATAL("Bad number of connected node uid={0} nb_edge={1} nb_connected={2}",
                   node.uniqueId(), nb_edge, nb_connectivity_node);
    ref_cx_nodes.resize(nb_edge);
    for (Int32 i = 0; i < nb_edge; ++i) {
      Edge edge = node.edge(i);
      Int32 connected_node_lid = (edge.node(0) == node) ? edge.node(1).localId() : edge.node(0).localId();
      ref_cx_nodes[i] = connected_node_lid;
    }
    std::sort(ref_cx_nodes.begin(), ref_cx_nodes.end());
    // Les nœuds de la connectivité 'nn_cv' sont bien triés par indice croissant des nœuds,
    // mais ce n'est pas forcément le cas de node.edges() car ces derniers sont triés par
    // uniqueId() des arêtes. On passe par un tableau temporaire qu'on trie pour tester les
    // valeurs.
    for (Int32 i = 0; i < nb_edge; ++i) {
      Int32 ref_cx_node_lid = ref_cx_nodes[i];
      Int32 cx_node_lid = nn_cv.nodeId(node, i);
      if (ref_cx_node_lid != cx_node_lid)
        ARCANE_FATAL("Bad connected node uid={0} index={1} ref={2} current={3}",
                     node.uniqueId(), i, ref_cx_node_lid, cx_node_lid);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshUnitTest::
_testBoundaryNodeNodeViaEdgeConnectivity()
{
  info() << "CREATE Boundary Edge Mesh";
  auto bx = Arcane::MeshUtils::computeBoundaryNodeNodeViaEdgeConnectivity(mesh(), "BoundaryNodeNodeViaEdge");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshUnitTest::
_testComputeOwnersDirect()
{
  info() << "Test: _testComputeOwnersDirect()";
  mesh()->modifier()->setDynamic(true);
  mesh()->modifier()->updateGhostLayers();
  mesh()->utilities()->computeAndSetOwnersForNodes();
  mesh()->utilities()->computeAndSetOwnersForFaces();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshUnitTest::
_testLocalIdsFromConnectivity()
{
  info() << "Testing LocalIdsFromConnectivity";
  UniqueArray<Int64> faces_connectivity;
  UniqueArray<Int32> faces_nb_node;
  UniqueArray<Int32> faces_local_id;
  ENUMERATE_ (Cell, icell, mesh()->allCells()) {
    for (Face face : icell->faces()) {
      Int32 nb_node = face.nbNode();
      faces_nb_node.add(nb_node);
      for (Node node : face.nodes())
        faces_connectivity.add(node.uniqueId());
    }
  }
  faces_local_id.resize(faces_nb_node.size());
  mesh()->utilities()->localIdsFromConnectivity(IK_Face, faces_nb_node, faces_connectivity, faces_local_id, false);

  // Vérifie que les indices locaux des faces sont corrects
  Int32 index = 0;
  ENUMERATE_ (Cell, icell, mesh()->allCells()) {
    for (Face face : icell->faces()) {
      if (face.localId() != faces_local_id[index])
        ARCANE_FATAL("Bad value face={0} expected_local_id={1}", ItemPrinter(face), faces_local_id[index]);
      ++index;
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace ArcaneTest

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
