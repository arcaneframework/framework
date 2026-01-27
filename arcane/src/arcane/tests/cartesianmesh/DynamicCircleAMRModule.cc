// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DynamicCircleAMRModule.cc                                (C) 2000-2024 */
/*                                                                           */
/* Module de test du gestionnaire de maillages cartésiens.                   */
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
  void computeRefine(CartesianPatch& patch);
  void syncUp(Integer level_down, VariableCellReal& var);
  void syncDown(Integer level_down, VariableCellReal& var);
  void postProcessing();

 private:

  void _svgOutput(const String& name);

 private:

  Real3 m_center{};
  Real m_radius = 0;
  Real m_radius_large_circle = 0;
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

  m_center = m_generation_info->globalLength() / 2;
  m_radius = math::normL2(m_center / 10);
  m_radius_large_circle = m_radius * 5;

  info() << "Global length : " << m_generation_info->globalLength();
  info() << "Global center : " << m_center;
  info() << "Global radius : " << m_radius;
  info() << "Large radius : " << m_radius_large_circle;

  m_cartesian_mesh = ICartesianMesh::getReference(mesh());
  CartesianMeshAMRMng amr_mng(m_cartesian_mesh);
  amr_mng.setOverlapLayerSizeTopLevel(2);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicCircleAMRModule::
compute()
{
  info() << "Module SayHello COMPUTE";

  CartesianMeshAMRMng amr_mng(m_cartesian_mesh);
  m_cartesian_mesh->computeDirections();

  m_refine_tracer.fill(0);
  m_amr.fill(0);
  m_distance.fill(0);

  Real3 center_large_circle = m_center + Real3{ std::cos(globalIteration()), std::sin(globalIteration()), 0 } * m_radius;
  info() << "Large center : " << center_large_circle;

  if (!m_change_radius && m_radius_large_circle > m_radius * 5) {
    m_change_radius = true;
  }
  else if (m_change_radius && m_radius_large_circle < m_radius) {
    m_change_radius = false;
  }

  m_radius_large_circle += (m_change_radius ? -1 : 1);

  constexpr Int32 nb_levels = 3;

  amr_mng.beginAdaptMesh(nb_levels, 0);
  for (Integer l = 0; l < nb_levels - 1; ++l) {
    for (Integer p = 0; p < m_cartesian_mesh->nbPatch(); ++p) {
      auto patch = m_cartesian_mesh->amrPatch(p);
      if (patch.level() == l) {
        computeDistance(patch, center_large_circle);
        computeValue(patch);
        computeRefine(patch);
      }
    }
    amr_mng.adaptLevel(l);
    syncUp(l, m_amr);
  }
  // if (globalIteration() == 2) {
  //   _svgOutput("aaaa");
  // }
  amr_mng.endAdaptMesh();

  ENUMERATE_ (Cell, icell, allCells()) {
    m_celluid[icell] = icell->uniqueId();
  }

  for (Integer p = 0; p < m_cartesian_mesh->nbPatch(); ++p) {
    auto patch = m_cartesian_mesh->amrPatch(p);
    if (patch.level() == nb_levels - 1) {
      computeDistance(patch, center_large_circle);
    }
  }

  postProcessing();

  if (globalIteration() > options()->getNSteps())
    subDomain()->timeLoopMng()->stopComputeLoop(true);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicCircleAMRModule::
computeDistance(CartesianPatch& patch, const Real3& center_large_circle)
{
  VariableNodeReal3& node_coords = mesh()->nodesCoordinates();

  // CellGroup in_patch_cells = patch.inPatchCells();
  // ENUMERATE_ (Cell, icell, in_patch_cells) {
  //   Real3 bary{};
  //   for (Node node : icell->nodes()) {
  //     bary += node_coords[node];
  //   }
  //   bary /= icell->nbNode();
  //
  //   // Real node_dist = math::normL2(bary - center_large_circle);
  //   // Real value = std::max(m_radius_large_circle - node_dist, 0.);
  //
  //   Real node_dist = math::normL2(bary - center_large_circle);
  //   node_dist = m_radius - std::abs(node_dist - m_radius_large_circle);
  //   Real value = std::max(node_dist, 0.);
  //
  //   m_amr[icell] = value;
  // }

  NodeDirectionMng ndm_x{ patch.nodeDirection(MD_DirX) };
  ENUMERATE_ (Node, inode, ndm_x.inPatchNodes()) {
    // Real node_dist = math::normL2(bary - center_large_circle);
    // Real value = std::max(m_radius_large_circle - node_dist, 0.);
    Real node_dist = math::normL2(node_coords[inode] - center_large_circle);
    node_dist = m_radius - std::abs(node_dist - m_radius_large_circle);
    Real value = std::max(node_dist, 0.);
    m_distance[inode] = value;
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
      // if (m_distance[node] == 0) {
      //   ARCANE_FATAL("0 detected -- NodeUID : {0} -- CellUID : {1}", node.uniqueId(), icell->uniqueId());
      // }
      moy += m_distance[node];
    }
    moy /= icell->nbNode();
    m_amr[icell] = moy;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DynamicCircleAMRModule::
computeRefine(CartesianPatch& patch)
{
  Integer nb_cell_refine = 0;

  Real ref = 0.05;
  Real ref_level = ref * pow(10, patch.level());
  info() << "Ref level : " << ref_level;

  CellDirectionMng cdm_x{ patch.cellDirection(MD_DirX) };
  ENUMERATE_ (Cell, icell, cdm_x.inPatchCells()) {
    //info() << "m_amr[icell] : " << m_amr[icell];
    if (m_amr[icell] > ref_level) {
      icell->mutableItemBase().addFlags(ItemFlags::II_Refine);
      m_refine_tracer[icell] = 1;
      nb_cell_refine++;
    }
    else {
      m_refine_tracer[icell] = 0;
    }
  }
  info() << "nb_cell_refine : " << nb_cell_refine;
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
  post_processor->setTimes(times);
  post_processor->setBaseDirectoryName(output_directory.path());

  ItemGroupList groups;
  // groups.add(allCells());
  for (Integer p = 0; p < m_cartesian_mesh->nbPatch(); ++p) {
    auto patch = m_cartesian_mesh->amrPatch(p);
    //groups.add(patch.cells());
    groups.add(patch.inPatchCells());
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
