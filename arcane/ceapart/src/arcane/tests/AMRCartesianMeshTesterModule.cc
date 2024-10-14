// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AMRCartesianMeshTesterModule.cc                             (C) 2000-2024 */
/*                                                                           */
/* Module de test du gestionnaire de maillages cartésiens AMR.               */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/CheckedConvert.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/Real2.h"
#include "arcane/utils/MD5HashAlgorithm.h"

#include "arcane/core/MeshUtils.h"
#include "arcane/core/MeshKind.h"
#include "arcane/core/Directory.h"

#include "arcane/core/ITimeLoopMng.h"
#include "arcane/core/ITimeLoopService.h"
#include "arcane/core/ITimeLoop.h"
#include "arcane/core/TimeLoopEntryPointInfo.h"
#include "arcane/core/IMesh.h"
#include "arcane/core/IItemFamily.h"
#include "arcane/core/ItemPrinter.h"
#include "arcane/core/IParallelMng.h"

#include "arcane/core/IMesh.h"
#include "arcane/core/IItemFamily.h"
#include "arcane/core/IMeshModifier.h"
#include "arcane/core/IMeshUtilities.h"
#include "arcane/core/ServiceBuilder.h"
#include "arcane/core/ServiceFactory.h"
#include "arcane/core/MeshStats.h"
#include "arcane/core/IPostProcessorWriter.h"
#include "arcane/core/IVariableMng.h"
#include "arcane/core/SimpleSVGMeshExporter.h"

#include "arcane/cartesianmesh/ICartesianMesh.h"
#include "arcane/cartesianmesh/CellDirectionMng.h"
#include "arcane/cartesianmesh/FaceDirectionMng.h"
#include "arcane/cartesianmesh/NodeDirectionMng.h"
#include "arcane/cartesianmesh/CartesianConnectivity.h"
#include "arcane/cartesianmesh/CartesianMeshRenumberingInfo.h"
#include "arcane/cartesianmesh/ICartesianMeshPatch.h"
#include "arcane/cartesianmesh/CartesianMeshUtils.h"
#include "arcane/cartesianmesh/CartesianMeshCoarsening2.h"
#include "arcane/cartesianmesh/CartesianMeshPatchListView.h"

#include "arcane/tests/ArcaneTestGlobal.h"
#include "arcane/tests/AMRCartesianMeshTester_axl.h"
#include "arcane/tests/CartesianMeshTestUtils.h"

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
class AMRCartesianMeshTesterModule
: public ArcaneAMRCartesianMeshTesterObject
{
 public:

  explicit AMRCartesianMeshTesterModule(const ModuleBuildInfo& mbi);
  ~AMRCartesianMeshTesterModule();

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
  ICartesianMesh* m_cartesian_mesh;
  Ref<CartesianMeshTestUtils> m_utils;
  UniqueArray<VariableCellReal*> m_cell_patch_variables;
  Int32 m_nb_expected_patch = 0;

 private:

  void _compute1();
  void _compute2();
  void _initAMR();
  void _coarsePatch();
  void _computeSubCellDensity(Cell cell);
  void _computeCenters();
  void _processPatches();
  void _writePostProcessing();
  void _checkUniqueIds();
  void _testDirections();
  void _cellsInPatch(Real3 position, Real3 length, bool is_3d, Int32 level, UniqueArray<Int32>& cells_in_patch);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AMRCartesianMeshTesterModule::
AMRCartesianMeshTesterModule(const ModuleBuildInfo& mbi)
: ArcaneAMRCartesianMeshTesterObject(mbi)
, m_density(VariableBuildInfo(this,"Density"))
, m_old_density(VariableBuildInfo(this,"OldDensity"))
, m_cell_center(VariableBuildInfo(this,"CellCenter"))
, m_face_center(VariableBuildInfo(this,"FaceCenter"))
, m_node_density(VariableBuildInfo(this,"NodeDensity"))
, m_cartesian_mesh(nullptr)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AMRCartesianMeshTesterModule::
~AMRCartesianMeshTesterModule()
{
  for (VariableCellReal* v : m_cell_patch_variables)
    delete v;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AMRCartesianMeshTesterModule::
staticInitialize(ISubDomain* sd)
{
  String time_loop_name("AMRCartesianMeshTestLoop");

  ITimeLoopMng* tlm = sd->timeLoopMng();
  ITimeLoop* time_loop = tlm->createTimeLoop(time_loop_name);

  {
    List<TimeLoopEntryPointInfo> clist;
    clist.add(TimeLoopEntryPointInfo("AMRCartesianMeshTester.buildInit"));
    time_loop->setEntryPoints(ITimeLoop::WBuild,clist);
  }

  {
    List<TimeLoopEntryPointInfo> clist;
    clist.add(TimeLoopEntryPointInfo("AMRCartesianMeshTester.init"));
    time_loop->setEntryPoints(ITimeLoop::WInit,clist);
  }

  {
    List<TimeLoopEntryPointInfo> clist;
    clist.add(TimeLoopEntryPointInfo("AMRCartesianMeshTester.compute"));
    time_loop->setEntryPoints(ITimeLoop::WComputeLoop,clist);
  }

  {
    StringList clist;
    clist.add("AMRCartesianMeshTester");
    time_loop->setRequiredModulesName(clist);
    clist.clear();
    clist.add("ArcanePostProcessing");
    clist.add("ArcaneCheckpoint");
    clist.add("ArcaneLoadBalance");
    time_loop->setOptionalModulesName(clist);
  }

  tlm->registerTimeLoop(time_loop);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AMRCartesianMeshTesterModule::
buildInit()
{
  if (subDomain()->isContinue())
    return;

  m_global_deltat.assign(1.0);

  IItemFamily* cell_family = defaultMesh()->cellFamily();
  cell_family->createGroup("CELL0");
  cell_family->createGroup("CELL1");
  cell_family->createGroup("CELL2");
  cell_family->createGroup("AMRPatchCells0");
  cell_family->createGroup("AMRPatchCells1");
  cell_family->createGroup("AMRPatchCells2");
  cell_family->createGroup("AMRPatchCells3");
  cell_family->createGroup("AMRPatchCells4");
  cell_family->createGroup("AMRPatchCells5");

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

void AMRCartesianMeshTesterModule::
init()
{
  info() << "AMR Init";

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

  m_cartesian_mesh = ICartesianMesh::getReference(mesh);
  m_utils = makeRef(new CartesianMeshTestUtils(m_cartesian_mesh,acceleratorMng()));

  if (!subDomain()->isContinue())
    _initAMR();

  _computeCenters();

  _coarsePatch();

  const bool do_coarse_at_init = options()->coarseAtInit();

  const Integer dimension = defaultMesh()->dimension();
  if (dimension==2)
    m_nb_expected_patch = 1 + options()->refinement2d().size();
  else if (dimension==3)
    m_nb_expected_patch = 1 + options()->refinement3d().size();

  // Si on dé-raffine à l'init, on aura un patch de plus
  if (do_coarse_at_init)
    ++m_nb_expected_patch;

  if (subDomain()->isContinue())
    m_cartesian_mesh->recreateFromDump();
  else{
    m_cartesian_mesh->computeDirections();
    CartesianMeshRenumberingInfo renumbering_info;
    renumbering_info.setRenumberPatchMethod(options()->renumberPatchMethod());
    renumbering_info.setSortAfterRenumbering(true);
    if (options()->coarseAtInit())
      renumbering_info.setParentPatch(m_cartesian_mesh->amrPatch(1));
    m_cartesian_mesh->renumberItemsUniqueId(renumbering_info);
    _checkUniqueIds();
    _processPatches();
    info() << "MaxUid for mesh=" << MeshUtils::getMaxItemUniqueIdCollective(m_cartesian_mesh->mesh());
  }

  // Initialise la densité.
  // On met une densité de 1.0 à l'intérieur
  // et on ajoute une densité de 5.0 pour chaque direction dans les
  // mailles de bord.
  m_density.fill(1.0);
  for( Integer idir=0, nb_dir=dimension; idir<nb_dir; ++idir){
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
  bool is_amr = m_nb_expected_patch!=1;
  if (options()->verbosityLevel()==0)
    m_utils->setNbPrint(5);
  m_utils->testAll(is_amr);
  _writePostProcessing();
  _testDirections();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AMRCartesianMeshTesterModule::
_checkUniqueIds()
{
  IMesh* mesh = m_cartesian_mesh->mesh();
  bool print_hash = true;
  MD5HashAlgorithm hash_algo;
  MeshUtils::checkUniqueIdsHashCollective(mesh->nodeFamily(),&hash_algo,
                                          options()->nodesUidHash(), print_hash);
  MeshUtils::checkUniqueIdsHashCollective(mesh->faceFamily(),&hash_algo,
                                          options()->facesUidHash(), print_hash);
  MeshUtils::checkUniqueIdsHashCollective(mesh->cellFamily(),&hash_algo,
                                          options()->cellsUidHash(), print_hash);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AMRCartesianMeshTesterModule::
_processPatches()
{
  const bool do_check = true;
  const bool is_verbose = options()->verbosityLevel()>=1;

  const Int32 dimension = defaultMesh()->dimension();
  // Vérifie qu'il y a autant de patchs que d'options raffinement dans
  // le jeu de données (en comptant le patch 0 qui est le maillage cartésien).
  // Cela permet de vérifier que les appels successifs
  // à computeDirections() n'ajoutent pas de patchs.
  Integer nb_expected_patch = m_nb_expected_patch;
    
  Integer nb_patch = m_cartesian_mesh->nbPatch();
  if (nb_expected_patch!=nb_patch)
    ARCANE_FATAL("Bad number of patchs expected={0} value={1}",nb_expected_patch,nb_patch);

  IParallelMng* pm = parallelMng();
  Int32 comm_rank = pm->commRank();
  Int32 comm_size = pm->commSize();

  UniqueArray<Int32> nb_cells_expected(options()->expectedNumberOfCellsInPatchs);
  if (nb_cells_expected.size()!=nb_patch)
    ARCANE_FATAL("Bad size ({0}, expected={1}) for option '{2}'",
                 nb_cells_expected.size(),nb_patch,options()->expectedNumberOfCellsInPatchs.name());

  // Nombre de mailles fantômes attendu. Utilisé uniquement en parallèle
  bool has_expected_ghost_cells = options()->expectedNumberOfGhostCellsInPatchs.isPresent();
  if (!pm->isParallel())
    has_expected_ghost_cells = false;

  UniqueArray<Int32> nb_ghost_cells_expected(options()->expectedNumberOfGhostCellsInPatchs);
  if (has_expected_ghost_cells && (nb_ghost_cells_expected.size()!=nb_patch))
    ARCANE_FATAL("Bad size ({0}, expected={1}) for option '{2}'",
                 nb_ghost_cells_expected.size(), nb_patch, options()->expectedNumberOfGhostCellsInPatchs.name());
  // Affiche les informations sur les patchs
  for( Integer i=0; i<nb_patch; ++i ){
    ICartesianMeshPatch* p = m_cartesian_mesh->patch(i);
    CellGroup patch_cells(p->cells());
    info() << "Patch cell_group=" << patch_cells.name() << " nb_cell=" << patch_cells.size();
    VariableCellReal* cellv = new VariableCellReal(VariableBuildInfo(defaultMesh(),String("CellPatch")+i));
    m_cell_patch_variables.add(cellv);
    cellv->fill(0.0);
    ENUMERATE_CELL(icell,patch_cells){
      (*cellv)[icell] = 2.0;
    }

    CellGroup patch_own_cell = patch_cells.own();
    UniqueArray<Int64> own_cells_uid;
    ENUMERATE_(Cell,icell,patch_own_cell){
      Cell cell{*icell};
      if (is_verbose)
        info() << "Patch i=" << i << " cell=" << ItemPrinter(*icell);
      own_cells_uid.add(cell.uniqueId());
    }
    // Affiche la liste globales des uniqueId() des mailles.
    {
      UniqueArray<Int64> global_cells_uid;
      pm->allGatherVariable(own_cells_uid,global_cells_uid);
      std::sort(global_cells_uid.begin(),global_cells_uid.end());
      Integer nb_global_uid = global_cells_uid.size();
      info() << "GlobalUids Patch=" << i << " NB=" << nb_global_uid
             << " expected=" << nb_cells_expected[i];
      // Vérifie que le nombre de mailles par patch est le bon.
      if (do_check && nb_cells_expected[i]!=nb_global_uid)
        ARCANE_FATAL("Bad number of cells for patch I={0} N={1} expected={2}",
                     i,nb_global_uid,nb_cells_expected[i]);
      if (is_verbose)
        for( Integer c=0; c<nb_global_uid; ++c )
          info() << "GlobalUid Patch=" << i << " I=" << c << " cell_uid=" << global_cells_uid[c];
    }
    // Teste le nombre de mailles fantômes
    if (has_expected_ghost_cells){
      Int32 local_nb_ghost_cell = patch_cells.size() - patch_own_cell.size();
      Int32 total = pm->reduce(Parallel::ReduceSum,local_nb_ghost_cell);
      pinfo() << "NbGhostCells my_rank=" << comm_rank << " local=" << local_nb_ghost_cell << " total=" << total;
      if (total!=nb_ghost_cells_expected[i])
        ARCANE_FATAL("Bad number of ghost cells for patch I={0} N={1} expected={2}",
                     i,total,nb_ghost_cells_expected[i]);
    }

    // Exporte le patch au format SVG
    if (dimension==2 && options()->dumpSvg()){
      String filename = String::format("Patch{0}-{1}-{2}.svg",i,comm_rank,comm_size);
      Directory directory = subDomain()->exportDirectory();
      String full_filename = directory.file(filename);
      std::ofstream ofile(full_filename.localstr());
      SimpleSVGMeshExporter exporter(ofile);
      exporter.write(patch_cells);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AMRCartesianMeshTesterModule::
_computeCenters()
{
  IMesh* mesh = defaultMesh();

  // Calcule le centre des mailles
  {
    VariableNodeReal3& nodes_coord = mesh->nodesCoordinates();
    ENUMERATE_CELL(icell,allCells()){
      Cell cell = *icell;
      Real3 center;
      for( NodeLocalId inode : cell.nodes() )
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
      for( NodeLocalId inode : face.nodes() )
        center += nodes_coord[inode];
      center /= face.nbNode();
      m_face_center[iface] = center;
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AMRCartesianMeshTesterModule::
_initAMR()
{
  // Regarde si on dé-raffine le maillage initial
  if (options()->coarseAtInit()){
    // Il faut que les directions aient été calculées avant d'appeler le dé-raffinement
    m_cartesian_mesh->computeDirections();

    info() << "Doint initial coarsening";

    if (m_cartesian_mesh->mesh()->meshKind().meshAMRKind() == eMeshAMRKind::PatchCartesianMeshOnly) {
      debug() << "Coarse with specific coarser (for cartesian mesh only)";
      Ref<ICartesianMeshAMRPatchMng> coarser = CartesianMeshUtils::cartesianMeshAMRPatchMng(m_cartesian_mesh);
      coarser->coarse();
    }
    else {
      Ref<CartesianMeshCoarsening2> coarser = CartesianMeshUtils::createCartesianMeshCoarsening2(m_cartesian_mesh);
      coarser->createCoarseCells();
    }

    CartesianMeshPatchListView patches = m_cartesian_mesh->patches();
    Int32 nb_patch = patches.size();
    {
      Int32 index = 0;
      info() << "NB_PATCH=" << nb_patch;
      for( CartesianPatch p : patches){
        info() << "Patch i=" << index << " nb_cell=" << p.cells().size();
        ++index;
      }
    }
  }
  // Parcours les mailles actives et ajoute dans la liste des mailles
  // à raffiner celles qui sont contenues dans le boîte englobante
  // spécifiée dans le jeu de données.
  Int32 dim = defaultMesh()->dimension();
  if (dim==2){
    for( auto& x : options()->refinement2d() ){    
      m_cartesian_mesh->refinePatch2D(x->position(),x->length());
      m_cartesian_mesh->computeDirections();
    }
  }
  if (dim==3){
    for( auto& x : options()->refinement3d() ){    
      m_cartesian_mesh->refinePatch3D(x->position(),x->length());
      m_cartesian_mesh->computeDirections();
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AMRCartesianMeshTesterModule::
_coarsePatch()
{
  Int32 dim = defaultMesh()->dimension();

  if (dim == 2) {
    //UniqueArray<Int32> cells_in_patchs;
    for (auto& x : options()->coarseZone2d()) {
      // _cellsInPatch(Real3(x->position()), Real3(x->length()), false, x->level(), cells_in_patchs);
      // defaultMesh()->modifier()->flagCellToCoarsen(cells_in_patchs);
      // defaultMesh()->modifier()->coarsenItemsV2();
      // cells_in_patchs.clear();
      m_cartesian_mesh->coarsePatch2D(x->position(), x->length());
      m_cartesian_mesh->computeDirections();
    }
  }
  if (dim == 3) {
    // UniqueArray<Int32> cells_in_patchs;
    for (auto& x : options()->coarseZone3d()) {
      // _cellsInPatch(x->position(), x->length(), true, x->level(), cells_in_patchs);
      // defaultMesh()->modifier()->flagCellToCoarsen(cells_in_patchs);
      // defaultMesh()->modifier()->coarsenItemsV2();
      // cells_in_patchs.clear();
      m_cartesian_mesh->coarsePatch3D(x->position(), x->length());
      m_cartesian_mesh->computeDirections();
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AMRCartesianMeshTesterModule::
compute()
{
  _compute1();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Calcule la densité d'une maille AMR.
 */
void AMRCartesianMeshTesterModule::
_computeSubCellDensity(Cell cell)
{
  Int32 nb_children = cell.nbHChildren();
  if (nb_children==0)
    return;
  // Pour les mailles AMR, la densité est la moyenne des noeuds qui la compose.
  for( Int32 j=0; j<nb_children; ++j ) {
    Real sub_density = 0.0;
    Cell sub_cell = cell.hChild(j);
    Integer sub_cell_nb_node = sub_cell.nbNode();
    for( Integer k=0; k<sub_cell_nb_node; ++k )
      sub_density += m_node_density[sub_cell.node(k)];
    sub_density /= (Real)sub_cell_nb_node;
    m_density[sub_cell] =sub_density;
    _computeSubCellDensity(sub_cell);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AMRCartesianMeshTesterModule::
_compute1()
{
  // Pour test, on parcours les N directions
  // et pour chaque maille, on modifie sa densité
  // par la formule new_density = (density+density_next+density_prev) / 3.0.
  
  // Effectue l'operation en deux fois. Une premiere sur les
  // mailles internes, et une deuxieme sur les mailles externes.
  // Du coup, il faut passer par une variable intermediaire (m_old_density)
  // mais on evite un test dans la boucle principale
  IMesh* mesh = defaultMesh();
  Integer nb_dir = mesh->dimension();
  for( Integer idir=0; idir<nb_dir; ++idir){
    m_old_density.copy(m_density);
    CellDirectionMng cdm(m_cartesian_mesh->cellDirection(idir));
    // Travail sur les mailles internes
    info() << "Direction=" << idir << " cells=" << cdm.innerCells().name()
           << " n=" << cdm.innerCells().size();
    ENUMERATE_CELL(icell,cdm.innerCells()){
      Cell cell = *icell;
      DirCell cc(cdm.cell(cell));
      Cell next = cc.next();
      Cell prev = cc.previous();
      Real d = m_old_density[icell] + m_old_density[next] + m_old_density[prev];
      m_density[icell] = d / 3.0;
      _computeSubCellDensity(cell);
    }
    // Travail sur les mailles externes
    // Test si la maille avant ou apres est nulle.
    ENUMERATE_CELL(icell,cdm.outerCells()){
      Cell cell = *icell;
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
      _computeSubCellDensity(cell);
    }
  }
  // Modifie la densité aux noeuds.
  // Elle sera égale à la moyenne des densités des mailles entourant ce noeud
  ENUMERATE_NODE(inode,mesh->allNodes()){
    Node node = *inode;
    Integer nb_cell = node.nbCell();
    Real density = 0.0;
    for( Integer i=0; i<nb_cell; ++i )
      density += m_density[node.cell(i)];
    density /= (Real)nb_cell;
    m_node_density[inode] = density;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AMRCartesianMeshTesterModule::
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

void AMRCartesianMeshTesterModule::
_writePostProcessing()
{
  info() << "Post-process AMR";
  IPostProcessorWriter* post_processor = options()->postProcessor();
  Directory output_directory = Directory(subDomain()->exportDirectory(),"amrtestpost1");
  output_directory.createDirectory();
  info() << "Creating output dir '" << output_directory.path() << "' for export";
  UniqueArray<Real> times;
  times.add(m_global_time());
  post_processor->setTimes(times);
  post_processor->setMesh(defaultMesh());
  post_processor->setBaseDirectoryName(output_directory.path());

  VariableList variables;
  //variables.add(m_density.variable());
  //variables.add(m_node_density.variable());
  for( VariableCellReal* v : m_cell_patch_variables )
    variables.add(v->variable());
  post_processor->setVariables(variables);
  ItemGroupList groups;
  groups.add(allCells());
  for( CartesianPatch p : m_cartesian_mesh->patches() )
    groups.add(p.cells());
  post_processor->setGroups(groups);
  IVariableMng* vm = subDomain()->variableMng();
  vm->writePostProcessing(post_processor);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AMRCartesianMeshTesterModule::
_testDirections()
{
  Integer nb_patch = m_cartesian_mesh->nbPatch();
  Integer nb_dir = m_cartesian_mesh->mesh()->dimension();
  NodeDirectionMng node_dm2;
  for( Integer ipatch=0; ipatch<nb_patch; ++ipatch ){
    ICartesianMeshPatch* p = m_cartesian_mesh->patch(ipatch);
    for( Integer idir=0; idir<nb_dir; ++idir ){
      NodeDirectionMng node_dm(p->nodeDirection(idir));
      node_dm2 = p->nodeDirection(idir);
      NodeGroup dm_all_nodes = node_dm.allNodes();
      ENUMERATE_NODE(inode,dm_all_nodes){
        DirNode dir_node(node_dm[inode]);
        DirNode dir_node2(node_dm2[inode]);
        Node prev_node = dir_node.previous();
        Node next_node = dir_node.next();
        Node prev_node2 = dir_node2.previous();
        Node next_node2 = dir_node2.next();
        m_utils->checkSameId(prev_node,prev_node2);
        m_utils->checkSameId(next_node,next_node2);
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AMRCartesianMeshTesterModule::
_cellsInPatch(Real3 position, Real3 length, bool is_3d, Int32 level, UniqueArray<Int32>& cells_in_patch)
{
  // Parcours les mailles actives et ajoute dans la liste des mailles
  // à raffiner celles qui sont contenues dans le boîte englobante
  // spécifiée dans le jeu de données.
  Real3 min_pos = position;
  Real3 max_pos = min_pos + length;
  ENUMERATE_ (Cell, icell, mesh()->allCells()) {
    if ((icell->level() == level) || (level == -1 && icell->nbHChildren() == 0)) {
      Real3 center = m_cell_center[icell];
      bool is_inside_x = center.x > min_pos.x && center.x < max_pos.x;
      bool is_inside_y = center.y > min_pos.y && center.y < max_pos.y;
      bool is_inside_z = (center.z > min_pos.z && center.z < max_pos.z) || !is_3d;
      if (is_inside_x && is_inside_y && is_inside_z) {
        cells_in_patch.add(icell.itemLocalId());
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_MODULE_AMRCARTESIANMESHTESTER(AMRCartesianMeshTesterModule);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace ArcaneTest

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
