// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DynamicCircleAMRModule.cc                                   (C) 2000-2026 */
/*                                                                           */
/* AMR test module type 3. Just a rotating circle.                           */
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

  // The principle is that we will have a large circle (sphere in 3D) with its
  // contour that will be refined.
  // This circle will move and its orbit will be a small circle centered
  // at the mesh center.

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
  // We ask the AMR manager for two overlapping mesh layers for
  // the highest level.
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

  // We keep it simple for the center of the large circle.
  Real3 center_large_circle = m_center + Real3{ std::cos(globalIteration()), std::sin(globalIteration()), 0 } * m_radius;
  info() << "Large center : " << center_large_circle;

  // We change the size of the large circle at each iteration. We decrease it
  // until a minimum is reached, then we increase it until a maximum is reached, etc.
  if (!m_change_radius && m_radius_large_circle > m_radius * 5) {
    m_change_radius = true;
  }
  else if (m_change_radius && m_radius_large_circle < m_radius) {
    m_change_radius = false;
  }
  m_radius_large_circle += (m_change_radius ? -1 : 1);

  // Mesh adaptation is done in three phases:
  //
  // First, we initialize the adaptation by providing the maximum number of
  // levels we will need. This maximum allows calculating the number of
  // overlapping mesh layers for each level. If this number of levels is not
  // reached, the number of layers must be adjusted during the third phase
  // (some extra calculations).
  //
  // The second argument is the level at which adaptation starts.
  // If, during a previous iteration, we created a level that we wish to keep,
  // we can choose it here. The patches at this level will not be deleted, nor
  // will the patches of lower levels. The patches of higher levels will be
  // deleted to be recreated in the second phase.
  // It is important to note that it is the patches that are deleted in this
  // first phase, not the meshes of these patches. The meshes (and the various
  // items around them), if they are no longer in any patch at the end of the
  // second phase, will be deleted in the third phase.
  // The consequence is that if a mesh had its patch deleted, but found a patch
  // again during the second phase, the variables associated with it will not
  // be reset.
  // Finally, it must be noted that an "InPatch" mesh can become an "Overlap"
  // mesh, and vice versa.
  amr_mng.beginAdaptMesh(options()->getNbLevelsMax(), 0);
  for (Integer level_to_adapt = 0; level_to_adapt < options()->getNbLevelsMax() - 1; ++level_to_adapt) {
    Integer nb_cell_refine = 0;
    // TODO : Method to iterate over all patches of a level.
    for (Integer p = 0; p < m_cartesian_mesh->nbPatch(); ++p) {
      auto patch = m_cartesian_mesh->amrPatch(p);
      if (patch.level() == level_to_adapt) {
        // Here, we will assign the "II_Refine" flag to all meshes that
        // need to be refined.
        computeDistance(patch, center_large_circle);
        computeValue(patch);
        // See computeRefine() method...
        nb_cell_refine += computeRefine(patch);
      }
    }
    if (nb_cell_refine == 0) {
      break;
    }
    // Second phase. Before calling this method, the meshes of the patches
    // at the "level_to_adapt" that must be refined must have the
    // "II_Refine" flag.
    // The first argument is the level to adapt. Adaptation is done
    // level by level, one by one, from the lowest to the highest.
    // It is possible to "restart" the adaptation by calling this method
    // with a level to adapt lower than the previous call. In this case,
    // patches at levels higher than "level_to_adapt" will be deleted
    // (as in the first phase).
    // The second argument allows the program to crash if the call is
    // unnecessary (i.e., if there are no "II_Refine" meshes or if
    // level_to_adapt is higher than the previous call + 1 (which implies
    // there are no "II_Refine" meshes)).
    // Here, setting this parameter to "false" may allow the removal of the
    // "nb_cell_refine" variable, at the cost of more unnecessary calculations.
    // Once this method is called, the created patches are usable
    // normally (their directions are calculated).
    amr_mng.adaptLevel(level_to_adapt, true);

    // Not useful for now.
    syncUp(level_to_adapt, m_amr);
  }
  // if (globalIteration() == 2) {
  //   _svgOutput("aaaa");
  // }

  // Finally, the last phase.
  // This phase will first adjust the number of overlapping mesh layers
  // for each patch in case the maximum number of levels given during the
  // first phase was not reached.
  // Then, it will delete all meshes that have neither the "II_InPatch" flag
  // nor the "II_Overlap" flag.
  amr_mng.endAdaptMesh();

  ENUMERATE_ (Cell, icell, allCells()) {
    m_celluid[icell] = icell->uniqueId();
  }

  // We calculate the values on the last refinement level.
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

// We calculate the distance of each node relative to the center of the large
// circle.
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
      // We give a gradient to the edge of the circle.
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

  // -2 because the last level we will adapt is the one below the
  // highest level (which will be created and will be level m_nb_levels-1).
  Real ref_level = (m_circle_width * 0.5) / pow(10, options()->getNbLevelsMax() - patch.level() - 2);

  // m_circle_width is the max value (see computeDistance()).
  // We refine meshes 50% above the max value.
  // Real ref_level = m_circle_width * 0.5;

  info() << "Ref level : " << ref_level << " -- Patch level : " << patch.level();

  Integer nb_cell_refine = 0;

  // Here, we will assign the "II_Refine" flags to meshes.
  // Several things to note.
  // First, it is possible to use the patch directions because the
  // adaptLevel() method (second phase of mesh adaptation) calculates
  // the directions of all newly created patches; there is no need
  // to wait for endAdaptMesh().
  // Finally, the "II_Refine" flag can only be assigned to meshes having
  // the "InPatch" flag. It is impossible to refine purely overlap meshes
  // (meshes having the "II_Overlap" flag AND not having the "InPatch" flag).
  // These meshes that can be refined are grouped in the
  // "inPatchCells()" group.
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

  // Export the patch in SVG format
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
