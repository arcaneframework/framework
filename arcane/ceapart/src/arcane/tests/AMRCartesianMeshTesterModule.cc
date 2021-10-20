// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AMRCartesianMeshTesterModule.cc                             (C) 2000-2021 */
/*                                                                           */
/* Module de test du gestionnaire de maillages cartésiens AMR.               */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/CheckedConvert.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/Real2.h"

#include "arcane/MeshUtils.h"
#include "arcane/Directory.h"

#include "arcane/ITimeLoopMng.h"
#include "arcane/ITimeLoopService.h"
#include "arcane/ITimeLoop.h"
#include "arcane/TimeLoopEntryPointInfo.h"
#include "arcane/IMesh.h"
#include "arcane/IItemFamily.h"
#include "arcane/ItemPrinter.h"
#include "arcane/IParallelMng.h"

#include "arcane/IMesh.h"
#include "arcane/IItemFamily.h"
#include "arcane/IMeshModifier.h"
#include "arcane/IMeshUtilities.h"
#include "arcane/ServiceBuilder.h"
#include "arcane/ServiceFactory.h"
#include "arcane/MeshStats.h"
#include "arcane/IPostProcessorWriter.h"
#include "arcane/IVariableMng.h"
#include "arcane/SimpleSVGMeshExporter.h"

#include "arcane/cea/ICartesianMesh.h"
#include "arcane/cea/CellDirectionMng.h"
#include "arcane/cea/FaceDirectionMng.h"
#include "arcane/cea/NodeDirectionMng.h"
#include "arcane/cea/CartesianConnectivity.h"
#include "arcane/cea/ICartesianMeshPatch.h"

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
 private:

  void _compute1();
  void _compute2();
  void _initAMR();
  void _computeSubCellDensity(Cell cell);
  void _computeCenters();
  void _processPatches();
  void _writePostProcessing();
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
  m_utils = makeRef(new CartesianMeshTestUtils(m_cartesian_mesh));

  if (!subDomain()->isContinue())
    _initAMR();

  _computeCenters();

  if (subDomain()->isContinue())
    m_cartesian_mesh->recreateFromDump();
  else{
    m_cartesian_mesh->computeDirections();
    m_cartesian_mesh->renumberItemsUniqueIdInPatchs();
    _processPatches();
  }

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
      //info() << "CELL: cell=" << ItemPrinter(*icell);
      // Maille au bord. J'ajoute de la densité.
      ++nb_boundary2;
      m_density[icell] += 5.0;
    }

    info() << "NB_BOUNDARY1=" << nb_boundary1 << " NB_BOUNDARY2=" << nb_boundary2;
  }
  m_utils->testAll();
  _writePostProcessing();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AMRCartesianMeshTesterModule::
_processPatches()
{
  // Vérifie qu'il y a autant de patchs que d'options raffinement dans
  // le jeu de données (en comptant le patch 0 qui es le maillage cartésien).
  // Cela permet de vérifier que les appels successifs
  // à computeDirections() n'ajoutent pas de patchs.
  Integer nb_expected_patch = 1 + options()->refinement().size();
  Integer nb_patch = m_cartesian_mesh->nbPatch();
  if (nb_expected_patch!=nb_patch)
    ARCANE_FATAL("Bad number of patchs expected={0} value={1}",nb_expected_patch,nb_patch);

  IParallelMng* pm = parallelMng();
  Int32 comm_rank = pm->commRank();
  Int32 comm_size = pm->commSize();

  UniqueArray<Int32> nb_cells_expected(options()->expectedNumberOfCellsInPatchs);
  if (nb_cells_expected.size()!=nb_patch)
    ARCANE_FATAL("Bad size for option '{0}'",options()->expectedNumberOfCellsInPatchs.name());

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
      // Si disponible, vérifie que le nombre de mailles par patch est le bon.
      if (nb_cells_expected[i]!=nb_global_uid)
        ARCANE_FATAL("Bad number of cells for patch I={0} N={1} expected={2}",
                     i,nb_cells_expected[i],nb_global_uid);

      for( Integer c=0; c<nb_global_uid; ++c )
        info() << "GlobalUid Patch=" << i << " I=" << c << " cell_uid=" << global_cells_uid[c];
    }

    // Exporte le patch au format SVG
    {
      String filename = String::format("Patch{0}-{1}-{2}.svg",i,comm_rank,comm_size);
      Directory directory = subDomain()->exportDirectory();
      String full_filename = directory.file(filename);
      ofstream ofile(full_filename.localstr());
      SimpleSVGMeshExporter exporter(ofile);
      exporter.write(patch_own_cell);
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
      for( NodeEnumerator inode(cell.nodes()); inode.hasNext(); ++inode )
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
      for( NodeEnumerator inode(face.nodes()); inode.hasNext(); ++inode )
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
  // Parcours les mailles actives et ajoute dans la liste des mailles
  // à raffiner celles qui sont contenues dans le boîte englobante
  // spécifiée dans le jeu de données.
  for( auto& x : options()->refinement() ){    
    m_cartesian_mesh->refinePatch2D(x->position(),x->length());
    m_cartesian_mesh->computeDirections();
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
  {
    Integer nb_patch = m_cartesian_mesh->nbPatch();
    for( Integer i=0; i<nb_patch; ++i ){
      ICartesianMeshPatch* p = m_cartesian_mesh->patch(i);
      groups.add(p->cells());
    }
  }
  post_processor->setGroups(groups);
  IVariableMng* vm = subDomain()->variableMng();
  vm->writePostProcessing(post_processor);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_MODULE_AMRCARTESIANMESHTESTER(AMRCartesianMeshTesterModule);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace ArcaneTest

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
