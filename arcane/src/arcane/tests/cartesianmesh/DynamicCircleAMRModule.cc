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

#include "arcane/cartesianmesh/ICartesianMesh.h"
#include "arcane/cartesianmesh/CartesianMeshAMRMng.h"
#include "arcane/cartesianmesh/CartesianPatch.h"

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

 private:

  Real3 m_center{};
  Real m_radius = 0;
  Real m_radius_large_circle = 0;
  bool m_change_radius = true;
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
  m_loop_sum = 0;

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

  Real3 center_large_circle = m_center + Real3{ std::cos(globalIteration()), std::sin(globalIteration()), 0 } * m_radius;
  info() << "Large center : " << center_large_circle;

  if (!m_change_radius && m_radius_large_circle > m_radius * 5) {
    m_change_radius = true;
  }
  else if (m_change_radius && m_radius_large_circle < m_radius) {
    m_change_radius = false;
  }

  m_radius_large_circle += (m_change_radius ? -1 : 1);

  VariableNodeReal3& node_coords = mesh()->nodesCoordinates();

  amr_mng.beginAdaptMesh(2, 0);
  for (Integer l = 0; l < 1; ++l) {
    for (Integer p = 0; p < m_cartesian_mesh->nbPatch(); ++p) {
      auto patch = m_cartesian_mesh->amrPatch(p);
      if (patch.level() == l) {
        ENUMERATE_ (Cell, icell, patch.inPatchCells()) {
          bool sup = false;
          bool inf = false;
          for (Node node : icell->nodes()) {
            Real node_dist = math::normL2(node_coords[node] - center_large_circle);
            if (node_dist < m_radius_large_circle)
              inf = true;
            else
              sup = true;
          }
          if (sup && inf)
            icell->mutableItemBase().addFlags(ItemFlags::II_Refine);
        }
      }
    }
    amr_mng.adaptLevel(l);
  }
  amr_mng.endAdaptMesh();


  ENUMERATE_ (Cell, icell, allCells()) {
    bool sup = false;
    bool inf = false;
    for (Node node : icell->nodes()) {
      Real node_dist = math::normL2(node_coords[node] - center_large_circle);
      if (node_dist < m_radius_large_circle)
        inf = true;
      else
        sup = true;
    }
    if (sup && inf)
      m_amr[icell] = 1;
    else
      m_amr[icell] = 0;
  }


  m_loop_sum = m_loop_sum() + m_global_iteration();
  if (globalIteration() > options()->getNSteps())
    subDomain()->timeLoopMng()->stopComputeLoop(true);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_MODULE_DYNAMICCIRCLEAMR(DynamicCircleAMRModule);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace ArcaneTest

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
