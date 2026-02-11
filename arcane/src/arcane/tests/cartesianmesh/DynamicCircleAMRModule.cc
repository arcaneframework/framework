// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DynamicCircleAMRModule.cc                                   (C) 2000-2026 */
/*                                                                           */
/* Module de test de l'AMR type 3. Juste un cercle qui tourne.               */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ITimeLoopMng.h"
#include "arcane/core/ITimeLoop.h"
#include "arcane/core/TimeLoopEntryPointInfo.h"
#include "arcane/core/ICartesianMeshGenerationInfo.h"
#include "arcane/core/IMesh.h"
#include "arcane/core/Directory.h"
#include "arcane/core/IPostProcessorWriter.h"
#include "arcane/core/IVariableMng.h"
#include "arcane/core/IParallelMng.h"

#include "arcane/cartesianmesh/ICartesianMesh.h"
#include "arcane/cartesianmesh/CartesianMeshAMRMng.h"
#include "arcane/cartesianmesh/CartesianMeshNumberingMng.h"
#include "arcane/cartesianmesh/CartesianPatch.h"
#include "arcane/cartesianmesh/CellDirectionMng.h"
#include "arcane/cartesianmesh/NodeDirectionMng.h"
#include "arcane/cartesianmesh/SimpleHTMLMeshAMRPatchExporter.h"

#include "arcane/tests/cartesianmesh/DynamicCircleAMR_axl.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace ArcaneTest
{

using namespace Arcane;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class DynamicCircleAMRModule
: public ArcaneDynamicCircleAMRObject
{

 public:

  explicit DynamicCircleAMRModule(const ModuleBuildInfo& mbi);
  ~DynamicCircleAMRModule() override = default;

 public:

  static void staticInitialize(ISubDomain* sd);

 public:

  void init() override;
  void compute() override;
  void computeDistance(CartesianPatch& patch, const Real3& center_large_circle);
  void computeValue(CartesianPatch& patch);
  Integer computeRefine(CartesianPatch& patch);
  void syncUp(Integer level_down, VariableCellReal& var);
  void syncDown(Integer level_down, VariableCellReal& var);
  void postProcessing();

 private:

  void _svgOutput(const String& name);

 private:

  Real3 m_center{};
  Real m_radius = 0;
  Real m_radius_large_circle = 0;
  Real m_circle_width = 0;
  bool m_change_radius = true;
  UniqueArray<Real> times;
  ICartesianMesh* m_cartesian_mesh = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

DynamicCircleAMRModule::
DynamicCircleAMRModule(const ModuleBuildInfo& mbi)
: ArcaneDynamicCircleAMRObject(mbi)
{}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicCircleAMRModule::
staticInitialize(ISubDomain* sd)
{
  String time_loop_name("DynamicCircleAMRLoop");

  ITimeLoopMng* tlm = sd->timeLoopMng();
  ITimeLoop* time_loop = tlm->createTimeLoop(time_loop_name);

  {
    List<TimeLoopEntryPointInfo> clist;
    clist.add(TimeLoopEntryPointInfo("DynamicCircleAMR.init"));
    time_loop->setEntryPoints(ITimeLoop::WInit,clist);
  }

  {
    List<TimeLoopEntryPointInfo> clist;
    clist.add(TimeLoopEntryPointInfo("DynamicCircleAMR.compute"));
    time_loop->setEntryPoints(ITimeLoop::WComputeLoop,clist);
  }

  {
    StringList clist;
    clist.add("DynamicCircleAMR");
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

void DynamicCircleAMRModule::
init()
{
  info() << "Module SayHello INIT";

  const auto* m_generation_info = ICartesianMeshGenerationInfo::getReference(mesh(), true);
  m_global_deltat = 1;

  // Le principe est que l'on va avoir un grand cercle (sphère en 3D) avec son
  // contour qui va être raffiné.
  // Ce cercle bougera et aura comme orbite un petit cercle ayant pour centre
  // le centre du maillage.

  m_center = m_generation_info->globalLength() / 2;
  m_radius = math::normL2(m_center / 10);
  m_radius_large_circle = m_radius * 5;
  m_circle_width = m_radius * 1.3;

  info() << "Global length : " << m_generation_info->globalLength();
  info() << "Global center : " << m_center;
  info() << "Radius of small circle : " << m_radius;
  info() << "Radius of large circle : " << m_radius_large_circle;
  info() << "Large circle width : " << m_circle_width;

  m_cartesian_mesh = ICartesianMesh::getReference(mesh());
  CartesianMeshAMRMng amr_mng(m_cartesian_mesh);
  // On demande au manageur AMR deux couches de mailles de recouvrement pour
  // le niveau le plus haut.
  amr_mng.setOverlapLayerSizeTopLevel(2);
  m_cartesian_mesh->computeDirections();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicCircleAMRModule::
compute()
{
  info() << "Module SayHello COMPUTE";

  CartesianMeshAMRMng amr_mng(m_cartesian_mesh);

  m_refine_tracer.fill(0);
  m_amr.fill(0);
  m_distance.fill(0);

  // On fait simple pour le centre du grand cercle.
  Real3 center_large_circle = m_center + Real3{ std::cos(globalIteration()), std::sin(globalIteration()), 0 } * m_radius;
  info() << "Large center : " << center_large_circle;

  // On change le taille du grand cercle à chaque itération. On la diminue
  // jusqu'à un min puis on l'augmente jusqu'à un max, &c.
  if (!m_change_radius && m_radius_large_circle > m_radius * 5) {
    m_change_radius = true;
  }
  else if (m_change_radius && m_radius_large_circle < m_radius) {
    m_change_radius = false;
  }
  m_radius_large_circle += (m_change_radius ? -1 : 1);

  // L'adaptation du maillage se fait en trois phases :
  //
  // D'abord, on initialise l'adaptation en donnant le nombre maximum de
  // niveaux dont on aura besoin. Ce maximum permet de calculer le nombre de
  // couches de mailles de recouvrement pour chaque niveau. Si ce nombre de
  // niveaux n'est pas atteint, le nombre de couches devra être ajusté lors de
  // la troisième phase (quelques calculs en plus).
  //
  // Le deuxième argument est le niveau auquel on commence l'adaptation.
  // Si, lors d'une précédente itération, on a créé un niveau que l'on
  // souhaite conserver, on peut le choisir ici. Les patchs de ce niveau ne
  // seront pas effacés, ainsi que les patchs des niveaux inférieurs. Les
  // patchs des niveaux supérieurs seront effacés pour être recréés dans la
  // seconde phase.
  // Il est important de noter que ce sont les patchs qui sont supprimés dans
  // cette première phase, pas les mailles de ces patchs. Les mailles (et les
  // différents items autours), si elles ne sont plus dans aucun patch à
  // l'issue de la seconde phase, seront supprimées dans la troisième phase.
  // La conséquence est que, si une maille à vu son patch être supprimé, mais
  // a retrouvé un patch lors de la seconde phase, les variables qui lui sont
  // associées ne seront pas réinitialisées.
  // Enfin, il faut noter qu'une maille "InPatch" peut devenir une
  // maille "Overlap", et inversement.
  amr_mng.beginAdaptMesh(options()->getNbLevelsMax(), 0);
  for (Integer level_to_adapt = 0; level_to_adapt < options()->getNbLevelsMax() - 1; ++level_to_adapt) {
    Integer nb_cell_refine = 0;
    // TODO : Methode pour itérer sur tous les patchs d'un niveau.
    for (Integer p = 0; p < m_cartesian_mesh->nbPatch(); ++p) {
      auto patch = m_cartesian_mesh->amrPatch(p);
      if (patch.level() == level_to_adapt) {
        // Ici, on va attribuer le flag "II_Refine" à toutes les mailles qui
        // faudra raffiner.
        computeDistance(patch, center_large_circle);
        computeValue(patch);
        // Voir méthode computeRefine()...
        nb_cell_refine += computeRefine(patch);
      }
    }
    if (nb_cell_refine == 0) {
      break;
    }
    // Deuxième phase. Avant d'appeler cette méthode, les mailles des patchs
    // du niveau "level_to_adapt" qui doivent être raffinées doivent avoir le
    // flag "II_Refine".
    // Le premier argument est le niveau à adapter. L'adaptation se fait
    // niveau par niveau, un par un, du plus bas vers le plus haut.
    // Il est possible de "recommencer" l'adaptation en appelant cette méthode
    // avec un niveau à adapter inférieur à l'appel précédent. Dans ce cas,
    // les patchs de niveau supérieur à "level_to_adapt" seront supprimés
    // (comme lors de la première phase).
    // Le second argument permet de faire planter le programme si l'appel est
    // inutile (c'est-à-dire s'il n'y a pas de mailles "II_Refine" ou si
    // level_to_adapt est supérieur au précedent appel +1 (ce qui implique
    // qu'il n'y a pas de mailles "II_Refine")).
    // Ici, mettre ce paramètre à "false" peut permettre de retirer la
    // variable "nb_cell_refine", au prix de plus de calculs inutiles.
    // Une fois cette méthode appelée, les patchs créés sont utilisables
    // normalement (leurs directions sont calculées).
    amr_mng.adaptLevel(level_to_adapt, true);

    // Pas utile pour l'instant.
    syncUp(level_to_adapt, m_amr);
  }
  // if (globalIteration() == 2) {
  //   _svgOutput("aaaa");
  // }

  // Enfin, la dernière phase.
  // Cette phase va d'abord ajuster le nombre de couches de mailles de
  // recouvrement de chaque patch dans le cas où le nombre de niveaux maximum
  // donné lors de la première phase n'est pas atteint.
  // Puis, elle va supprimer toutes les mailles qui n'ont ni le flag
  // "II_InPatch", ni le flag "II_Overlap".
  amr_mng.endAdaptMesh();

  ENUMERATE_ (Cell, icell, allCells()) {
    m_celluid[icell] = icell->uniqueId();
  }

  // On calcule les valeurs sur le dernier niveau de raffinement.
  for (Integer p = 0; p < m_cartesian_mesh->nbPatch(); ++p) {
    auto patch = m_cartesian_mesh->amrPatch(p);
    if (patch.level() == options()->getNbLevelsMax() - 1) {
      computeDistance(patch, center_large_circle);
      computeValue(patch);
    }
  }

  postProcessing();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// On calcule la distance de chaque noeud par rapport au centre du grand
// cercle.
void DynamicCircleAMRModule::
computeDistance(CartesianPatch& patch, const Real3& center_large_circle)
{
  constexpr bool alternative_pattern = false;
  VariableNodeReal3& node_coords = mesh()->nodesCoordinates();

  NodeDirectionMng ndm_x{ patch.nodeDirection(MD_DirX) };
  ENUMERATE_ (Node, inode, ndm_x.inPatchNodes()) {
    if constexpr (alternative_pattern) {
      Real node_dist = math::normL2(node_coords[inode] - center_large_circle);
      Real value = std::max(m_radius_large_circle - node_dist, 0.);
      m_distance[inode] = value;
    }
    else {
      Real node_dist = math::normL2(node_coords[inode] - center_large_circle);
      // On donne un dégradé au bord du cercle.
      node_dist = m_circle_width - std::abs(node_dist - m_radius_large_circle);
      Real value = std::max(node_dist, 0.);
      m_distance[inode] = value;
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicCircleAMRModule::
computeValue(CartesianPatch& patch)
{
  CellDirectionMng cdm_x{ patch.cellDirection(MD_DirX) };
  ENUMERATE_ (Cell, icell, cdm_x.inPatchCells()) {
    Real moy = 0;
    for (Node node : icell->nodes()) {
      if (std::isnan(m_distance[node])) {
        ARCANE_FATAL("Nan detected -- NodeUID : {0} -- CellUID : {1}", node.uniqueId(), icell->uniqueId());
      }
      moy += m_distance[node];
    }
    moy /= icell->nbNode();
    m_amr[icell] = moy;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer DynamicCircleAMRModule::
computeRefine(CartesianPatch& patch)
{
  CartesianMeshNumberingMng numbering(m_cartesian_mesh);
  // Real ref = 0.04;
  // Real ref_level = ref * pow(10, patch.level());

  // -2 car le dernier niveau que l'on va adapter est celui sous le niveau le
  // plus haut (qui sera créé et qui sera le niveau m_nb_levels-1).
  Real ref_level = (m_circle_width * 0.5) / pow(10, options()->getNbLevelsMax() - patch.level() - 2);

  // m_circle_width est la valeur max (voir computeDistance()).
  // On raffine les mailles 50% au-dessus de la valeur max.
  // Real ref_level = m_circle_width * 0.5;

  info() << "Ref level : " << ref_level << " -- Patch level : " << patch.level();

  Integer nb_cell_refine = 0;

  // Ici, on va attribuer le flags "II_Refine" à des mailles.
  // Plusieurs choses à noter.
  // D'abord, il est possible d'utiliser les directions des patchs car la
  // méthode adaptLevel() (deuxième phase de l'adaptation du maillage) calcule
  // les directions de tous les patchs nouvellement créés ; pas besoin
  // d'attendre endAdaptMesh().
  // Enfin, le flag "II_Refine" ne peut être attribué qu'à des mailles ayant
  // le flag "InPatch". Il est impossible de raffiner des mailles purement de
  // recouvrement (les mailles ayant le flag "II_Overlap" ET n'ayant pas le
  // flag "InPatch").
  // Ces mailles pouvant être raffinées sont regroupées dans le groupe
  // "inPatchCells()".
  CellDirectionMng cdm_x{ patch.cellDirection(MD_DirX) };
  ENUMERATE_ (Cell, icell, cdm_x.inPatchCells()) {
    //debug() << "m_amr[icell] : " << m_amr[icell];
    if (m_amr[icell] > ref_level) {
      icell->mutableItemBase().addFlags(ItemFlags::II_Refine);
      debug() << "Need refine CellUID : " << icell->uniqueId()
              << " -- Pos : " << numbering.cellUniqueIdToCoord(*icell);
      m_refine_tracer[icell] = 1;
      nb_cell_refine++;
    }
    else {
      m_refine_tracer[icell] = 0;
    }
  }
  info() << "nb_cell_refine : " << nb_cell_refine;
  return nb_cell_refine;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicCircleAMRModule::
syncUp(Integer level_down, VariableCellReal& var)
{
  ENUMERATE_ (Cell, icell, mesh()->allLevelCells(level_down)) {
    if (icell->hasHChildren()) {
      Real value = var[icell];
      for (Integer i = 0; i < icell->nbHChildren(); ++i) {
        Cell child = icell->hChild(i);
        var[child] = value;
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicCircleAMRModule::
syncDown(Integer level_down, VariableCellReal& var)
{
  ENUMERATE_ (Cell, icell, mesh()->allLevelCells(level_down)) {
    if (icell->hasHChildren()) {
      Real value = 0;
      for (Integer i = 0; i < icell->nbHChildren(); ++i) {
        Cell child = icell->hChild(i);
        value += var[child];
      }
      value /= icell->nbHChildren();
      var[icell] = value;
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicCircleAMRModule::
postProcessing()
{
  info() << "Post-process AMR";
  IPostProcessorWriter* post_processor = options()->postProcessor();
  Directory output_directory = Directory(subDomain()->exportDirectory(), "amrtestpost1");
  output_directory.createDirectory();
  info() << "Creating output dir '" << output_directory.path() << "' for export";
  times.add(m_global_time());

  VariableCellInteger patch_cell(VariableBuildInfo(mesh(), "Patch"));

  post_processor->setTimes(times);
  post_processor->setBaseDirectoryName(output_directory.path());

  ItemGroupList groups;
  for (Integer p = 0; p < m_cartesian_mesh->nbPatch(); ++p) {
    auto patch = m_cartesian_mesh->amrPatch(p);
    //groups.add(patch.cells());
    groups.add(patch.inPatchCells());
    ENUMERATE_ (Cell, icell, patch.inPatchCells()) {
      patch_cell[icell] = patch.index();
    }
  }

  post_processor->setGroups(groups);

  IVariableMng* vm = subDomain()->variableMng();

  vm->writePostProcessing(post_processor);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicCircleAMRModule::
_svgOutput(const String& name)
{
  const Int32 dimension = defaultMesh()->dimension();
  if (dimension != 2) {
    return;
  }

  SimpleHTMLMeshAMRPatchExporter amr_exporter;

  const Int32 nb_patch = m_cartesian_mesh->nbPatch();
  for (Integer i = 0; i < nb_patch; ++i) {
    amr_exporter.addPatch(m_cartesian_mesh->amrPatch(i));
  }

  IParallelMng* pm = parallelMng();
  Int32 comm_rank = pm->commRank();
  Int32 comm_size = pm->commSize();

  // Exporte le patch au format SVG
  String amr_filename = String::format("MeshPatch{0}-{1}-{2}.html", name, comm_rank, comm_size);
  String amr_full_filename = subDomain()->exportDirectory().file(amr_filename);
  std::ofstream amr_ofile(amr_full_filename.localstr());
  amr_exporter.write(amr_ofile);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_MODULE_DYNAMICCIRCLEAMR(DynamicCircleAMRModule);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace ArcaneTest

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
