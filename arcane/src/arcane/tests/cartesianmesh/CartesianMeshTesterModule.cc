// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CartesianMeshTesterModule.cc                                (C) 2000-2026 */
/*                                                                           */
/* Module de test du gestionnaire de maillages cartésiens.                   */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/CheckedConvert.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/StringBuilder.h"

#include "arcane/core/MeshUtils.h"
#include "arcane/core/MathUtils.h"

#include "arcane/core/ITimeLoopMng.h"
#include "arcane/core/ITimeLoopService.h"
#include "arcane/core/ITimeLoop.h"
#include "arcane/core/TimeLoopEntryPointInfo.h"
#include "arcane/core/IMesh.h"
#include "arcane/core/IItemFamily.h"
#include "arcane/core/ItemPrinter.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/IMeshWriter.h"
#include "arcane/core/MeshKind.h"

#include "arcane/core/ICaseDocument.h"
#include "arcane/core/IInitialPartitioner.h"
#include "arcane/core/IMesh.h"
#include "arcane/core/IItemFamily.h"
#include "arcane/core/IMeshModifier.h"
#include "arcane/core/IMeshUtilities.h"
#include "arcane/core/ServiceBuilder.h"
#include "arcane/core/ServiceFactory.h"
#include "arcane/core/IMeshPartitionerBase.h"
#include "arcane/core/BasicService.h"
#include "arcane/core/MeshReaderMng.h"
#include "arcane/core/IGridMeshPartitioner.h"
#include "arcane/core/ICartesianMeshGenerationInfo.h"
#include "arcane/core/AbstractItemOperationByBasicType.h"

#include "arcane/core/Connectivity.h"

#include "arcane/cartesianmesh/CartesianMeshCoarsening.h"
#include "arcane/cartesianmesh/CartesianMeshCoarsening2.h"

#include "arcane/cartesianmesh/CartesianMeshUtils.h"
#include "arcane/cartesianmesh/ICartesianMesh.h"
#include "arcane/cartesianmesh/CellDirectionMng.h"
#include "arcane/cartesianmesh/NodeDirectionMng.h"
#include "arcane/cartesianmesh/CartesianConnectivity.h"

#include "arcane/cartesianmesh/ICartesianMeshPatch.h"

#include "arcane/tests/ArcaneTestGlobal.h"
#include "arcane/tests/cartesianmesh/CartesianMeshTester_axl.h"
#include "arcane/tests/cartesianmesh/CartesianMeshTestUtils.h"
#include "arcane/tests/cartesianmesh/CartesianMeshV2TestUtils.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace ArcaneTest
{

using namespace Arcane;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Module de test pour les infos sur les maillages cartésiens.
 */
class CartesianMeshTesterModule
: public ArcaneCartesianMeshTesterObject
{
  class CountByBasicType
  : public AbstractItemOperationByBasicType
  {
   public:

    void applyQuad4(ItemVectorView group) override { m_nb_quad4 += group.size(); }
    void applyHexaedron8(ItemVectorView group) override { m_nb_hexa8 += group.size(); }

   public:

    Int32 m_nb_quad4 = 0;
    Int32 m_nb_hexa8 = 0;
  };

 public:

  explicit CartesianMeshTesterModule(const ModuleBuildInfo& mbi);
  ~CartesianMeshTesterModule();

 public:

  static void staticInitialize(ISubDomain* sd);

 public:
  
  void buildInit() override;
  void compute() override;
  void init() override;

 private:

  VariableCellReal m_density;
  VariableCellReal m_old_density;
  VariableCellReal3 m_cell_center;
  VariableFaceReal3 m_face_center;
  VariableNodeReal m_node_density; 
  VariableFaceInt64 m_faces_uid;
  ICartesianMesh* m_cartesian_mesh;
  IInitialPartitioner* m_initial_partitioner;
  Ref<CartesianMeshTestUtils> m_utils;
  Ref<CartesianMeshV2TestUtils> m_utils_v2;

 private:

  void _compute1();
  void _compute2();
  void _sample(ICartesianMesh* cartesian_mesh);
  void _testXmlInfos();
  void _testGridPartitioning();
  void _printCartesianMeshInfos();
  void _checkFaceUniqueIdsAreContiguous();
  void _checkNearlyEqual(Real3 a,Real3 b,const String& message);
  void _testCoarsening();
  void _checkSpecificApplyOperator();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class CartesianMeshPartitionerService
: public BasicService
, public IMeshPartitionerBase
{
 public:
  explicit CartesianMeshPartitionerService(const ServiceBuildInfo& sbi)
  : BasicService(sbi){}
 public:
  void build() override {}
  void notifyEndPartition() override {}
  IPrimaryMesh* primaryMesh() override { return mesh()->toPrimaryMesh(); }
  void partitionMesh(bool initial_partition) override
  {
    if (!initial_partition)
      return;
    IMesh* mesh = this->mesh();
    IParallelMng* pm = mesh->parallelMng();
    Int32 nb_rank = pm->commSize();
    IItemFamily* cell_family = mesh->cellFamily();
    VariableItemInt32& cells_new_owner = cell_family->itemsNewOwner();
    ItemGroup own_cells = cell_family->allItems().own();
    Int64 nb_cell = own_cells.size();
    Int64 nb_bloc = nb_rank * 3;
    Int64 cell_index = 0;
    info() << "Partitioning with 'CartesianMeshPartitionerService' nb_rank=" << nb_rank;
    ENUMERATE_CELL(icell,mesh->ownCells()){
      Cell cell = *icell;
      // Utilise des Int64 plutôt que des Int32 pour être sur de ne pas déborder.
      Int64 new_owner = ((cell_index * nb_bloc) / nb_cell) % nb_rank;
      cells_new_owner[cell] = CheckedConvert::toInt32(new_owner);
      ++cell_index;
    }
    cells_new_owner.synchronize();
    mesh->utilities()->changeOwnersFromCells();
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE(CartesianMeshPartitionerService,
                        Arcane::ServiceProperty("CartesianMeshPartitionerTester",Arcane::ST_SubDomain),
                        ARCANE_SERVICE_INTERFACE(IMeshPartitionerBase));

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class CartesianMeshInitialPartitioner
: public TraceAccessor
, public IInitialPartitioner
{
 public:

  CartesianMeshInitialPartitioner(ISubDomain* sd)
  : TraceAccessor(sd->traceMng()), m_sub_domain(sd)
  {
  }
  void build() override {}
  void partitionAndDistributeMeshes(ConstArrayView<IMesh*> meshes) override
  {
    for( IMesh* mesh : meshes ){
      info() << "Partitioning mesh name=" << mesh->name();
      _doPartition(mesh);
    }
  }
 private:
  void _doPartition(IMesh* mesh)
  {
    ServiceBuilder<IMeshPartitionerBase> sbuilder(m_sub_domain);
    String service_name = "CartesianMeshPartitionerTester";
    auto mesh_partitioner(sbuilder.createReference(service_name,mesh));

    bool is_dynamic = mesh->isDynamic();
    mesh->modifier()->setDynamic(true);
    mesh->utilities()->partitionAndExchangeMeshWithReplication(mesh_partitioner.get(),true);
    mesh->modifier()->setDynamic(is_dynamic);
  }
 private:
  ISubDomain* m_sub_domain;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartesianMeshTesterModule::
CartesianMeshTesterModule(const ModuleBuildInfo& mbi)
: ArcaneCartesianMeshTesterObject(mbi)
, m_density(VariableBuildInfo(this,"Density"))
, m_old_density(VariableBuildInfo(this,"OldDensity"))
, m_cell_center(VariableBuildInfo(this,"CellCenter"))
, m_face_center(VariableBuildInfo(this,"FaceCenter"))
, m_node_density(VariableBuildInfo(this,"NodeDensity"))
, m_faces_uid(VariableBuildInfo(this,"CartesianMeshTesterNodeUid"))
, m_cartesian_mesh(nullptr)
, m_initial_partitioner(nullptr)
{
  // Regarde s'il faut tester le partitionnement
  if (!platform::getEnvironmentVariable("TEST_PARTITIONING").null()){
    ISubDomain* sd = mbi.subDomain();
    m_initial_partitioner = new CartesianMeshInitialPartitioner(sd);
    info() << "SETTING INITIAL PARTITIONER";
    // NOTE: le sous-domaine prend possession du partitionneur. Il ne faut
    // donc pas le détruire.
    sd->setInitialPartitioner(m_initial_partitioner);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartesianMeshTesterModule::
~CartesianMeshTesterModule()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshTesterModule::
staticInitialize(ISubDomain* sd)
{
  String time_loop_name("CartesianMeshTestLoop");

  ITimeLoopMng* tlm = sd->timeLoopMng();
  ITimeLoop* time_loop = tlm->createTimeLoop(time_loop_name);

  {
    List<TimeLoopEntryPointInfo> clist;
    clist.add(TimeLoopEntryPointInfo("CartesianMeshTester.buildInit"));
    time_loop->setEntryPoints(ITimeLoop::WBuild,clist);
  }

  {
    List<TimeLoopEntryPointInfo> clist;
    clist.add(TimeLoopEntryPointInfo("CartesianMeshTester.init"));
    time_loop->setEntryPoints(ITimeLoop::WInit,clist);
  }

  {
    List<TimeLoopEntryPointInfo> clist;
    clist.add(TimeLoopEntryPointInfo("CartesianMeshTester.compute"));
    time_loop->setEntryPoints(ITimeLoop::WComputeLoop,clist);
  }

  {
    StringList clist;
    clist.add("CartesianMeshTester");
    time_loop->setRequiredModulesName(clist);
    clist.clear();
    clist.add("ArcanePostProcessing");
    clist.add("ArcaneCheckpoint");
    time_loop->setOptionalModulesName(clist);
  }

  tlm->registerTimeLoop(time_loop);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshTesterModule::
buildInit()
{
  bool has_edge = options()->hasEdges();
  info() << "Adding edge connectivity?=" << has_edge;
  if (has_edge){
    Connectivity c(mesh()->connectivity());
    c.enableConnectivity(Connectivity::CT_HasEdge);
  }
  
  m_global_deltat.assign(1.0);

  IItemFamily* cell_family = defaultMesh()->cellFamily();
  cell_family->createGroup("CELL0");
  cell_family->createGroup("CELL1");
  cell_family->createGroup("CELL2");
  IItemFamily* face_family = defaultMesh()->faceFamily();
  face_family->createGroup("FACE0");
  face_family->createGroup("FACE1");
  face_family->createGroup("FACE2");
  face_family->createGroup("FACE3");
  face_family->createGroup("FACE4");
  face_family->createGroup("FACE5");

  face_family->createGroup("AllFacesDirection0");
  face_family->createGroup("AllFacesDirection1");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshTesterModule::
init()
{
  IMesh* mesh = defaultMesh();

  IItemFamily* cell_family = mesh->cellFamily();
  Int32UniqueArray ids(1);
  ids[0] = 0;
  cell_family->createGroup("CELL0",ids,true);
  ids[0] = 1;
  cell_family->createGroup("CELL1",ids,true);
  ids[0] = 2;
  cell_family->createGroup("CELL2",ids,true);
  IItemFamily* face_family = defaultMesh()->faceFamily();
  ids[0] = 0;
  face_family->createGroup("FACE0",ids,true);
  ids[0] = 1;
  face_family->createGroup("FACE1",ids,true);
  ids[0] = 2;
  face_family->createGroup("FACE2",ids,true);
  ids[0] = 3;
  face_family->createGroup("FACE3",ids,true);
  ids[0] = 4;
  face_family->createGroup("FACE4",ids,true);
  ids[0] = 5;
  face_family->createGroup("FACE5",ids,true);

  // Calcule le centre des mailles
  {
    VariableNodeReal3& nodes_coord = mesh->nodesCoordinates();
    ENUMERATE_CELL(icell,allCells()){
      Cell cell = *icell;
      Real3 center;
      for( NodeLocalId inode : cell.nodeIds() )
        center += nodes_coord[inode];
      center /= cell.nbNode();
      m_cell_center[icell] = center;
    }
  }

  // Calcule le centre des faces
  {
    VariableNodeReal3& nodes_coord = mesh->nodesCoordinates();
    ENUMERATE_FACE(iface,allFaces()){
      Face face = *iface;
      Real3 center;
      for( NodeLocalId inode : face.nodeIds() )
        center += nodes_coord[inode];
      center /= face.nbNode();
      m_face_center[iface] = center;
    }
  }

  m_cartesian_mesh = ICartesianMesh::getReference(mesh);
  m_cartesian_mesh->computeDirections();

  m_utils = makeRef(new CartesianMeshTestUtils(m_cartesian_mesh,acceleratorMng()));
  m_utils_v2 = makeRef(new CartesianMeshV2TestUtils(m_cartesian_mesh));

  // Initialise la densité.
  // On met une densité de 1.0 à l'intérieur
  // et on ajoute une densité de 5.0 pour chaque direction dans les
  // mailles de bord.
  m_density.fill(1.0);
  Integer nb_dir = defaultMesh()->dimension();
  for( Integer idir=0; idir<nb_dir; ++idir){
    CellDirectionMng cdm(m_cartesian_mesh->cellDirection(idir));
    Integer nb_boundary1 = 0;
    Integer nb_boundary2 = 0;
    ENUMERATE_CELL(icell,cdm.innerCells()){
      DirCell cc(cdm.cell(*icell));
      Cell next = cc.next();
      Cell prev = cc.previous();
      if (next.null() || prev.null()){
        // Maille au bord. J'ajoute de la densité.
        // Ne devrait pas arriver car on est sur les innerCells()
        ++nb_boundary1;
        m_density[icell] += 5.0;
      }
    }
    // Parcours les mailles frontières pour la direction
    ENUMERATE_CELL(icell,cdm.outerCells()){
      DirCell cc(cdm[icell]);
      if (icell.index()<5)
        info() << "CELL: cell=" << ItemPrinter(*icell)
               << " next=" << ItemPrinter(cc.next())
               << " previous=" << ItemPrinter(cc.previous());
      // Maille au bord. J'ajoute de la densité.
      ++nb_boundary2;
      m_density[icell] += 5.0;
    }

    info() << "NB_BOUNDARY1=" << nb_boundary1 << " NB_BOUNDARY2=" << nb_boundary2;
  }

  m_utils->testAll(false);
  m_utils_v2->testAll();
  _checkFaceUniqueIdsAreContiguous();
  _testXmlInfos();
  _testGridPartitioning();
  _printCartesianMeshInfos();
  _checkSpecificApplyOperator();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshTesterModule::
_testCoarsening()
{
  Int32 coarse_version = options()->coarseCartesianMesh();
  IMesh* mesh = m_cartesian_mesh->mesh();
  Int32 mesh_dim = mesh->dimension();
  const Int32 coarse_factor = 1 << mesh_dim;
  if (coarse_version==1){
    info() << "Test CartesianCoarsening V1";
    Ref<CartesianMeshCoarsening> coarser = m_cartesian_mesh->createCartesianMeshCoarsening();
    IItemFamily* cell_family = mesh->cellFamily();
    CellInfoListView cells(cell_family);
    coarser->createCoarseCells();
    Int32 index = 0;
    for( Int32 cell_lid : coarser->coarseCells()){
      Cell cell = cells[cell_lid];
      info() << "Test1: CoarseCell= " << ItemPrinter(cell);
      ConstArrayView<Int32> sub_cells(coarser->refinedCells(index));
      ++index;
      for( Int32 sub_lid : sub_cells )
        info() << "SubCell=" << ItemPrinter(cells[sub_lid]);
    }
    coarser->removeRefinedCells();
  }

  if (coarse_version==2){
    info() << "Test CartesianCoarsening V2";
    const Int32 nb_orig_cell = ownCells().size();
    Ref<CartesianMeshCoarsening2> coarser = CartesianMeshUtils::createCartesianMeshCoarsening2(m_cartesian_mesh);
    coarser->createCoarseCells();
    ENUMERATE_(Cell,icell,allCells()){
      Cell cell = *icell;
      if (cell.level()!=0)
        continue;
      info() << "Test2: CoarseCell= " << ItemPrinter(cell);
      for( Int32 i=0, n=cell.nbHChildren(); i<n; ++i ){
        info() << "SubCell=" << ItemPrinter(cell.hChild(i));
      }
    }
    Int32 nb_patch = m_cartesian_mesh->nbPatch();
    info() << "NB_PATCH=" << nb_patch;
    for( Int32 i=0; i<nb_patch; ++i ){
      ICartesianMeshPatch* p = m_cartesian_mesh->patch(i);
      info() << "Patch i=" << i << " nb_cell=" << p->cells().size();
    }
    coarser->removeRefinedCells();
    // Le nombre de mailles doit être égal au nombre d'origine
    // divisé par \a coarse_factor.
    const Int32 nb_final_cell = ownCells().size();
    info() << "nb_orig_cell=" << nb_orig_cell << " nb_final_cell=" << nb_final_cell
           << " coarse_factor=" << coarse_factor;
    const Int32 nb_computed = nb_final_cell * coarse_factor;
    if (nb_computed != nb_orig_cell)
      ARCANE_FATAL("Bad number of cells orig={0} computed={1}", nb_orig_cell, nb_computed);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshTesterModule::
compute()
{
  if (m_global_iteration()==1)
    _testCoarsening();

  _compute1();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshTesterModule::
_compute1()
{
  // Pour test, on parcours les N directions
  // et pour chaque maille, on modifie sa densité
  // par la formule new_density = (density+density_next+density_prev) / 3.0.
  
  // Effectue l'operation en deux fois. Une premiere sur les
  // mailles internes, et une deuxieme sur les mailles externes.
  // Du coup, il faut passer par une variable intermediaire (m_old_density)
  // mais on evite un test dans la boucle principale
  Integer nb_dir = defaultMesh()->dimension();
  for( Integer idir=0; idir<nb_dir; ++idir){
    m_old_density.copy(m_density);
    CellDirectionMng cdm(m_cartesian_mesh->cellDirection(idir));
    // Travail sur les mailles internes
    ENUMERATE_CELL(icell,cdm.innerCells()){
      DirCell cc(cdm.cell(*icell));
      Cell next = cc.next();
      Cell prev = cc.previous();
      Real d = m_old_density[icell] + m_old_density[next] + m_old_density[prev];
      m_density[icell] = d / 3.0;
    }
    // Travail sur les mailles externes
    // Test si la maille avant ou apres est nulle.
    ENUMERATE_CELL(icell,cdm.outerCells()){
      DirCell cc(cdm[icell]);
      Cell next = cc.next();
      Cell prev = cc.previous();
      Real d = m_old_density[icell];
      Integer n = 1;
      if (!next.null()){
        d += m_old_density[next];
        ++n;
      }
      if (!prev.null()){
        d += m_old_density[prev];
        ++n;
      }
      m_density[icell] = d / n;
    }
  }

  {
    Int64 to_add = m_global_iteration();
    ENUMERATE_(Face,iface,ownFaces()){
      Int64 uid = iface->uniqueId();
      m_faces_uid[iface] = uid + to_add;
    }
    m_faces_uid.synchronize();
    ENUMERATE_(Face,iface,allFaces()){
      Face face(*iface);
      Int64 uid = face.uniqueId();
      Int64 expected_value = uid + to_add;
      if (expected_value!=m_faces_uid[iface])
        ARCANE_FATAL("Bad FaceUid face={0} expected={1} value={2}",ItemPrinter(face),expected_value,m_faces_uid[iface]);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshTesterModule::
_compute2()
{
  // Pour test, on parcours les N directions
  // et pour chaque maille, on modifie sa densité
  // par la formule new_density = (density+density_next+density_prev) / 3.0.

  // A noter que cette methode ne donne pas le meme comportement que
  // _compute1() car les mailles de bord et internes sont mises à jour
  // dans un ordre différent.
  Integer nb_dir = defaultMesh()->dimension();
  for( Integer idir=0; idir<nb_dir; ++idir){
    CellDirectionMng cdm(m_cartesian_mesh->cellDirection(idir));
    // Travail sur toutes les mailles
    ENUMERATE_CELL(icell,cdm.allCells()){
      DirCell cc(cdm[icell]);
      Cell next = cc.next();
      Cell prev = cc.previous();
      Real d = m_density[icell];
      Integer n = 1;
      if (!next.null()){
        d += m_density[next];
        ++n;
      }
      if (!prev.null()){
        d += m_density[prev];
        ++n;
      }
      m_density[icell] = d / n;
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshTesterModule::
_sample(ICartesianMesh* cartesian_mesh)
{
  //! [SampleNodeToCell]
  CartesianConnectivity cc = cartesian_mesh->connectivity();
  ENUMERATE_NODE(inode,allNodes()){
    Node n = *inode;
    Cell c1 = cc.upperLeft(n); // Maille en haut à gauche
    Cell c2 = cc.upperRight(n); // Maille en haut à droite
    Cell c3 = cc.lowerRight(n); // Maille en bas à droite
    Cell c4 = cc.lowerLeft(n); // Maille en bas à gauche
    info(6) << " C1=" << ItemPrinter(c1) << " C2=" << ItemPrinter(c2)
            << " C3=" << ItemPrinter(c3) << " C4=" << ItemPrinter(c4);
  }
  //! [SampleNodeToCell]
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshTesterModule::
_checkFaceUniqueIdsAreContiguous()
{
  if (!options()->checkContiguousFaceUniqueIds())
    return;
  info() << "Test " << A_FUNCINFO;
  // Parcours les faces et vérifie que le uniqueId() de chaque face n'est
  // pas supérieur au nombre total de face.
  Int64 total_nb_face = allFaces().own().size();
  total_nb_face = parallelMng()->reduce(Parallel::ReduceSum,total_nb_face);
  info() << "TotalNbFace=" << total_nb_face;
  ENUMERATE_(Face,iface,allFaces()){
    Face face = *iface;
    if (face.uniqueId()>=total_nb_face)
      ARCANE_FATAL("FaceUniqueId is too big: uid={0} total_nb_face={1}",
                   face.uniqueId(),total_nb_face);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshTesterModule::
_testXmlInfos()
{
  info() << "PRINT Xml infos for <cartesian> mesh generator";
  ICaseDocument* cd = subDomain()->caseDocument();
  XmlNodeList mesh_elements = cd->meshElements();
  if (mesh_elements.size()==0)
    return;
  XmlNode mesh_generator_element = mesh_elements[0].child("meshgenerator");
  // Si nul, cela signifie qu'on n'utilise pas le 'meshgenerator'.
  if (mesh_generator_element.null()){
    info() << "No element <meshgenerator> found";
    return;
  }
  XmlNode cartesian_node = mesh_generator_element.child("cartesian");
  if (cartesian_node.null()){
    info() << "No element <cartesian> found";
    return;
  }
  XmlNode origine_node = cartesian_node.child("origine");
  XmlNode nsd_node = cartesian_node.child("nsd");

  // Récupère et affiche les infos pour <lx>.
  XmlNodeList lx_node_list = cartesian_node.children("lx");
  info() << "NB_X=" << lx_node_list.size();
  for( XmlNode lx_node : lx_node_list ){
    Real lx_value = lx_node.valueAsReal(true);
    Integer nx_value = lx_node.attr("nx",true).valueAsInteger(true);
    Real px_value = lx_node.attr("prx").valueAsReal(true);
    info() << "V=" << lx_value << " nx=" << nx_value << " px=" << px_value;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshTesterModule::
_testGridPartitioning()
{
  if (!options()->unstructuredMeshFile.isPresent())
    return;
  // NOTE: On utilise explicitement le namespace Arcane
  // pour que la documentation générée par doxygen génère les
  // liens correctement.

  //![SampleGridMeshPartitioner]
  // file_name est le nom du fichier de maillage non structuré

  Arcane::String file_name = options()->unstructuredMeshFile();
  info() << "UnstructuredMeshFileName=" << file_name;

  Arcane::ISubDomain* sd = subDomain();
  Arcane::ICartesianMesh* cartesian_mesh = m_cartesian_mesh;
  Arcane::IMesh* current_mesh = cartesian_mesh->mesh();
  Arcane::IParallelMng* pm = current_mesh->parallelMng();

  Arcane::MeshReaderMng reader_mng(sd);
  Arcane::IMesh* new_mesh = reader_mng.readMesh("UnstructuredMesh2",file_name,pm);
  info() << "MESH=" << new_mesh;

  // Création du service de partitionnement
  Arcane::ServiceBuilder<Arcane::IGridMeshPartitioner> sbuilder(sd);
  auto partitioner_ref = sbuilder.createReference("SimpleGridMeshPartitioner",new_mesh);
  Arcane::IGridMeshPartitioner* partitioner = partitioner_ref.get();

  // Positionne les coordonnées de notre sous-domaine dans la grille
  Int32 sd_x = cartesian_mesh->cellDirection(MD_DirX).subDomainOffset();
  Int32 sd_y = cartesian_mesh->cellDirection(MD_DirY).subDomainOffset();
  Int32 sd_z = cartesian_mesh->cellDirection(MD_DirZ).subDomainOffset();
  partitioner->setPartIndex(sd_x,sd_y,sd_z);

  // Positionne la bounding box de notre sous-domaine.
  // Pour cela, parcours uniquement nos noeuds et prend les coordonnées min et max
  Real max_value = FloatInfo<Real>::maxValue();
  Real min_value = -max_value;
  Arcane::Real3 min_box(max_value,max_value,max_value);
  Arcane::Real3 max_box(min_value,min_value,min_value);
  VariableNodeReal3& nodes_coord = current_mesh->nodesCoordinates();
  ENUMERATE_(Cell,icell,current_mesh->ownCells()){
    Cell cell{*icell};
    for( Node node : cell.nodes() ){
      Real3 coord = nodes_coord[node];
      min_box = math::min(min_box,coord);
      max_box = math::max(max_box,coord);
    }
  }
  partitioner->setBoundingBox(min_box,max_box);

  // Applique le partitionnement
  partitioner->applyMeshPartitioning(new_mesh);
  //![SampleGridMeshPartitioner]

  // Maintenant, écrit le fichier du maillage non structuré et de notre partie
  // cartésienne.
  const bool is_debug = false;
  if (is_debug){
    ServiceBuilder<IMeshWriter> sbuilder2(sd);
    auto mesh_writer = sbuilder2.createReference("VtkLegacyMeshWriter",SB_Collective);
    {
      StringBuilder fname = "cut_mesh_";
      fname += pm->commRank();
      fname += ".vtk";
      mesh_writer->writeMeshToFile(new_mesh,fname);
    }
    {
      StringBuilder fname = "my_mesh_";
      fname += pm->commRank();
      fname += ".vtk";
      mesh_writer->writeMeshToFile(current_mesh,fname);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshTesterModule::
_checkNearlyEqual(Real3 a,Real3 b,const String& message)
{
  info() << "A=" << a;
  info() << "B=" << b;
  if (!math::isNearlyEqual(a.x,b.x))
    ARCANE_FATAL("Bad value X expected={0} value={1} message={2}",a.x,b.x,message);
  if (!math::isNearlyEqual(a.y,b.y))
    ARCANE_FATAL("Bad value Y expected={0} value={1} message={2}",a.y,b.y,message);
  if (!math::isNearlyEqual(a.z,b.z))
    ARCANE_FATAL("Bad value Z expected={0} value={1} message={2}",a.z,b.z,message);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshTesterModule::
_printCartesianMeshInfos()
{
  auto* cartesian_info = ICartesianMeshGenerationInfo::getReference(defaultMesh(),false);
  if (!cartesian_info)
    ARCANE_FATAL("No cartesian info");

  info() << "Test: _printCartesianMeshInfos()";
  info() << " Origin=" << cartesian_info->globalOrigin();
  info() << " Length=" << cartesian_info->globalLength();

  if (options()->expectedMeshOrigin.isPresent())
    _checkNearlyEqual(cartesian_info->globalOrigin(),options()->expectedMeshOrigin(),"Origin");
  if (options()->expectedMeshLength.isPresent())
    _checkNearlyEqual(cartesian_info->globalLength(),options()->expectedMeshLength(),"Length");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshTesterModule::
_checkSpecificApplyOperator()
{
  CountByBasicType op;
  CellGroup all_cells = allCells();
  Int32 nb_item = all_cells.size();
  Int32 dim = mesh()->dimension();
  all_cells.applyOperation(&op);
  eMeshStructure mk = mesh()->meshKind().meshStructure();
  info() << "MeshStructure=" << mk;
  if (mk != eMeshStructure::Cartesian)
    ARCANE_FATAL("Invalid mesh structure v={0} (expected 'Cartesian')", mk);
  if (dim == 3) {
    if (nb_item != op.m_nb_hexa8)
      ARCANE_FATAL("Bad number of Hexa8 n={0} expected={1}", op.m_nb_hexa8, nb_item);
  }
  else if (dim == 2) {
    if (nb_item != op.m_nb_quad4)
      ARCANE_FATAL("Bad number of Quad8 n={0} expected={1}", op.m_nb_quad4, nb_item);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_MODULE_CARTESIANMESHTESTER(CartesianMeshTesterModule);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace ArcaneTest

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
