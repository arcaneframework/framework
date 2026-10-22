// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshSubdividerTesterModule.cc                                     (C) 2000-2026 */
/*                                                                           */
/* AMR type 3 test module.                                                   */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ITimeLoopMng.h"
#include "arcane/core/ITimeLoop.h"
#include "arcane/core/TimeLoopEntryPointInfo.h"
#include "arcane/core/IMesh.h"
#include "arcane/core/Directory.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/MachineShMemWinVariable.h"
#include "arcane/core/ParallelMngUtils.h"

#include "arcane/cartesianmesh/ICartesianMesh.h"
#include "arcane/cartesianmesh/CartesianMeshAMRMng.h"
#include "arcane/cartesianmesh/CartesianMeshNumberingMng.h"
#include "arcane/cartesianmesh/CartesianPatch.h"
#include "arcane/cartesianmesh/CellDirectionMng.h"
#include "arcane/cartesianmesh/FaceDirectionMng.h"
#include "arcane/cartesianmesh/NodeDirectionMng.h"
#include "arcane/cartesianmesh/SimpleHTMLMeshAMRPatchExporter.h"

#include "arcane/core/IUnitTest.h"
#include "arcane/tests/meshsubdivider/MeshSubdividerTester_axl.h"
#include "arccore/base/FatalErrorException.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace ArcaneTest
{
using namespace Arcane;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class MeshSubdividerTesterModule
: public ArcaneMeshSubdividerTesterObject
{
 public:

  explicit MeshSubdividerTesterModule(const ModuleBuildInfo& mbi);
  ~MeshSubdividerTesterModule() override = default;

 public:

  static void staticInitialize(ISubDomain* sd);

 public:
  void build() override;
  void init() override;
  void compute() override;
  void exit() override;

  void computeVolume();
  //---------------------------------------------------------------------------
  // 2D : aire d'un triangle 3D
  //---------------------------------------------------------------------------
  Real computeTriangleVolume(Real3 n0, Real3 n1, Real3 n2);

  //---------------------------------------------------------------------------
  // 2D : aire d'un quadrilatère plan convexe
  // Découpage en (n0,n1,n2) et (n0,n2,n3)
  //---------------------------------------------------------------------------

  Real computeQuadVolume(Real3 n0, Real3 n1, Real3 n2, Real3 n3);

  //---------------------------------------------------------------------------
  // 3D : volume d'un tétraèdre
  //---------------------------------------------------------------------------

  Real
  computeTetraVolume(Real3 n0, Real3 n1, Real3 n2, Real3 n3);

  //---------------------------------------------------------------------------
  // 3D : volume d'un hexaèdre
  //
  // Numérotation VTK :
  //       7-------6
  //      /|      /|
  //     4-------5 |
  //     | 3-----|-2
  //     |/      |/
  //     0-------1
  //
  // Décomposition suivant la diagonale (n0,n6).
  // Les sommets de chaque tétraèdre sont ordonnés pour conserver
  // une orientation cohérente.
  //---------------------------------------------------------------------------

  Real
  computeHexaVolume(
  Real3 n0, Real3 n1, Real3 n2, Real3 n3,
  Real3 n4, Real3 n5, Real3 n6, Real3 n7);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshSubdividerTesterModule::
MeshSubdividerTesterModule(const ModuleBuildInfo& mbi)
: ArcaneMeshSubdividerTesterObject(mbi)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshSubdividerTesterModule::
staticInitialize(ISubDomain* sd)
{
  String time_loop_name("MeshSubdividerTesterLoop");

  ITimeLoopMng* tlm = sd->timeLoopMng();
  ITimeLoop* time_loop = tlm->createTimeLoop(time_loop_name);
  {
    List<TimeLoopEntryPointInfo> clist;
    clist.add(TimeLoopEntryPointInfo("MeshSubdividerTester.build"));
    time_loop->setEntryPoints(ITimeLoop::WBuild, clist);
  }
  {
    List<TimeLoopEntryPointInfo> clist;
    clist.add(TimeLoopEntryPointInfo("MeshSubdividerTester.init"));
    time_loop->setEntryPoints(ITimeLoop::WInit, clist);
  }

  {
    List<TimeLoopEntryPointInfo> clist;
    clist.add(TimeLoopEntryPointInfo("MeshSubdividerTester.compute"));
    time_loop->setEntryPoints(ITimeLoop::WComputeLoop, clist);
  }
  {
    List<TimeLoopEntryPointInfo> clist;
    clist.add(TimeLoopEntryPointInfo("MeshSubdividerTester.exit"));
    time_loop->setEntryPoints(ITimeLoop::WExit, clist);
  }
  {
    StringList clist;
    clist.add("MeshSubdividerTester");
    clist.add("ArcanePostProcessing");
    time_loop->setRequiredModulesName(clist);
    clist.clear();

    clist.add("ArcaneCheckpoint");
    time_loop->setOptionalModulesName(clist);

  }
  tlm->registerTimeLoop(time_loop);
}

void MeshSubdividerTesterModule::build()
{
  info() << "MeshSubdividerTesterModule::build";
  for (IUnitTest* service : options()->test)
    service->buildInitializeTest();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshSubdividerTesterModule::
init()
{
  info() << "MeshSubdividerTesterModule::init";
  for (IUnitTest* service : options()->test)
    service->initializeTest();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshSubdividerTesterModule::
compute()
{
  info() << "MeshSubdividerTesterModule::compute";
  for (IUnitTest* service : options()->test)
    service->executeTest();

  m_arcane_cell_rank.fill(0);
  m_arcane_cell_uniqueid.fill(0);
  ENUMERATE_CELL(icell,mesh()->allCells() )
  {
    m_arcane_cell_rank[icell] = icell->owner();
    m_arcane_cell_uniqueid[icell] = icell->uniqueId().asInt64();
  }
  computeVolume();
}

void MeshSubdividerTesterModule::exit()
{
  info() << "MeshSubdividerTesterModule::exit";
  for (IUnitTest* service : options()->test)
    service->finalizeTest();
}

void MeshSubdividerTesterModule::computeVolume()
{
  info() << "MeshSubdividerTesterModule::computeVolume";

  bool negative_volume_found = false;
  const VariableNodeReal3& coords = mesh()->nodesCoordinates();
  ENUMERATE_CELL (icell, mesh()->allCells()) {
    Cell cell = *icell;

    Real3 nodes[8];
    const Integer nb_node = cell.nbNode();

    if (nb_node > 8)
      ARCANE_FATAL("Trop de noeuds pour cette cellule : {0}", nb_node);

    for (Integer i = 0; i < nb_node; ++i)
      nodes[i] = coords[cell.node(i)];

    Real volume = 0.;

    switch (cell.type()) {
    case IT_Triangle3:
      volume = computeTriangleVolume(
      nodes[0], nodes[1], nodes[2]);
      break;

    case IT_Quad4:
      volume = computeQuadVolume(
      nodes[0], nodes[1], nodes[2], nodes[3]);
      break;

    case IT_Tetraedron4:
      volume = computeTetraVolume(
      nodes[0], nodes[1], nodes[2], nodes[3]);
      break;

    case IT_Hexaedron8:
      volume = computeHexaVolume(
      nodes[0], nodes[1], nodes[2], nodes[3],
      nodes[4], nodes[5], nodes[6], nodes[7]);
      break;

    default:
      ARCANE_FATAL("Type de cellule non supporté : {0}",
                   cell.type());
    }
    if (volume < 0.0) {
      error()
      << "Volume négatif, cell uid=" << cell.uniqueId()
      << ", volume=" << volume;
      negative_volume_found = true;
      ARCANE_FATAL("Volume négatif pour la cellule {0}", cell.uniqueId());
    }
  }
  if (negative_volume_found)
    ARCANE_FATAL("Volume négatif");for (IUnitTest* service : options()->test)
    service->finalizeTest();
}

Real MeshSubdividerTesterModule::computeTriangleVolume(Real3 n0, Real3 n1, Real3 n2)
{
  return math::normeR3(math::vecMul(n1 - n0, n2 - n0)) / 2.0;
}

Real MeshSubdividerTesterModule::computeQuadVolume(Real3 n0, Real3 n1, Real3 n2, Real3 n3)
{
  Real volume1 = computeTriangleVolume(n0, n1, n2);
  Real volume2 = computeTriangleVolume(n0, n2, n3);

  return volume1 + volume2;
}

Real MeshSubdividerTesterModule::computeTetraVolume(Real3 n0, Real3 n1, Real3 n2, Real3 n3)
{
  // math::abs
  return (
         math::mixteMul(n1 - n0, n2 - n0, n3 - n0)) /
  6.0;
}

Real MeshSubdividerTesterModule::computeHexaVolume(Real3 n0, Real3 n1, Real3 n2, Real3 n3, Real3 n4, Real3 n5,
                                                   Real3 n6,
                                                   Real3 n7)
{
  // Volumes orientés des six tétraèdres.
  Real v0 = math::mixteMul(n6 - n0, n1 - n0, n2 - n0) / 6.0;
  Real v1 = math::mixteMul(n6 - n0, n2 - n0, n3 - n0) / 6.0;

  Real v2 = math::mixteMul(n1 - n0, n6 - n0, n5 - n0) / 6.0;
  Real v3 = math::mixteMul(n5 - n0, n6 - n0, n4 - n0) / 6.0;

  Real v4 = math::mixteMul(n4 - n0, n6 - n0, n7 - n0) / 6.0;
  Real v5 = math::mixteMul(n7 - n0, n6 - n0, n3 - n0) / 6.0;
  //math::abs
  return (v0 + v1 + v2 + v3 + v4 + v5);
}

ARCANE_REGISTER_MODULE_MESHSUBDIVIDERTESTER(MeshSubdividerTesterModule);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
} // End namespace ArcaneTest

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
