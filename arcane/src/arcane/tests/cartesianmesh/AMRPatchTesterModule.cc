// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AMRPatchTesterModule.cc                                (C) 2000-2024 */
/*                                                                           */
/* Module de test du gestionnaire de maillages cartésiens.                   */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ITimeLoopMng.h"
#include "arcane/core/ITimeLoop.h"
#include "arcane/core/TimeLoopEntryPointInfo.h"
#include "arcane/core/IMesh.h"
#include "arcane/core/Directory.h"
#include "arcane/core/IParallelMng.h"

#include "arcane/cartesianmesh/ICartesianMesh.h"
#include "arcane/cartesianmesh/CartesianMeshAMRMng.h"
#include "arcane/cartesianmesh/CartesianMeshNumberingMng.h"
#include "arcane/cartesianmesh/CartesianPatch.h"
#include "arcane/cartesianmesh/CellDirectionMng.h"
#include "arcane/cartesianmesh/FaceDirectionMng.h"
#include "arcane/cartesianmesh/NodeDirectionMng.h"
#include "arcane/cartesianmesh/SimpleHTMLMeshAMRPatchExporter.h"

#include "arcane/tests/cartesianmesh/AMRPatchTester_axl.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace ArcaneTest
{

using namespace Arcane;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class AMRPatchTesterModule
: public ArcaneAMRPatchTesterObject
{

 public:

  explicit AMRPatchTesterModule(const ModuleBuildInfo& mbi);
  ~AMRPatchTesterModule() override = default;

 public:

  static void staticInitialize(ISubDomain* sd);

 public:

  void init() override;
  void compute() override;
  void _reset();
  void _test1();
  void _test1_1();
  void _test1_2();
  void testMarkCellsToRefine(Integer level);
  void _svgOutput(const String& name);

 private:

  ICartesianMesh* m_cartesian_mesh = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AMRPatchTesterModule::
AMRPatchTesterModule(const ModuleBuildInfo& mbi)
: ArcaneAMRPatchTesterObject(mbi)
{}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AMRPatchTesterModule::
staticInitialize(ISubDomain* sd)
{
  String time_loop_name("AMRPatchTesterLoop");

  ITimeLoopMng* tlm = sd->timeLoopMng();
  ITimeLoop* time_loop = tlm->createTimeLoop(time_loop_name);

  {
    List<TimeLoopEntryPointInfo> clist;
    clist.add(TimeLoopEntryPointInfo("AMRPatchTester.init"));
    time_loop->setEntryPoints(ITimeLoop::WInit, clist);
  }

  {
    List<TimeLoopEntryPointInfo> clist;
    clist.add(TimeLoopEntryPointInfo("AMRPatchTester.compute"));
    time_loop->setEntryPoints(ITimeLoop::WComputeLoop, clist);
  }

  {
    StringList clist;
    clist.add("AMRPatchTester");
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

void AMRPatchTesterModule::
init()
{

  m_cartesian_mesh = ICartesianMesh::getReference(mesh());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AMRPatchTesterModule::
compute()
{
  _test1();
  _reset();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * Ce n'est pas un vrai reset total (l'objet CartesianPatchGroup n'est pas
 * remis à zéro).
 */
void AMRPatchTesterModule::
_reset()
{
  m_cartesian_mesh->computeDirections();

  CartesianMeshAMRMng amr_mng(m_cartesian_mesh);
  amr_mng.beginAdaptMesh(1, 0);
  amr_mng.endAdaptMesh();

  m_cartesian_mesh->computeDirections();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AMRPatchTesterModule::
_test1()
{
  m_cartesian_mesh->computeDirections();
  CartesianMeshAMRMng amr_mng(m_cartesian_mesh);

  {
    _svgOutput("WithoutRefine");

    _test1_1();
    _test1_2();
  }
  {
    amr_mng.beginAdaptMesh(2, 0);
    testMarkCellsToRefine(0);
    amr_mng.adaptLevel(0);
    amr_mng.endAdaptMesh();

    _svgOutput("2Levels0First");

    _test1_1();
    _test1_2();
  }
  {
    amr_mng.beginAdaptMesh(3, 0);
    testMarkCellsToRefine(0);
    amr_mng.adaptLevel(0);
    amr_mng.endAdaptMesh();

    _svgOutput("2Levels3Max0First");

    _test1_1();
    _test1_2();
  }
  {
    amr_mng.beginAdaptMesh(3, 0);
    testMarkCellsToRefine(0);
    amr_mng.adaptLevel(0);
    testMarkCellsToRefine(1);
    amr_mng.adaptLevel(1);
    amr_mng.endAdaptMesh();

    _svgOutput("3Levels0First");

    _test1_1();
    _test1_2();
  }
  {
    amr_mng.beginAdaptMesh(3, 1);
    testMarkCellsToRefine(1);
    amr_mng.adaptLevel(1);
    amr_mng.endAdaptMesh();

    _svgOutput("3Levels1First");

    _test1_1();
    _test1_2();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AMRPatchTesterModule::
_test1_1()
{
  Integer dimension = mesh()->dimension();
  VariableNodeInt16 var_test(VariableBuildInfo(mesh(), "VarTest"));
  var_test.fill(-1);

  for (Integer p = 0; p < m_cartesian_mesh->nbPatch(); ++p) {
    auto patch = m_cartesian_mesh->amrPatch(p);
    {
      constexpr Int16 val = 1;
      NodeDirectionMng ndm_x{ patch.nodeDirection(MD_DirX) };
      ENUMERATE_ (Node, inode, ndm_x.inPatchNodes()) {
        var_test[inode] = val;
      }

      CellDirectionMng cdm_x{ patch.cellDirection(MD_DirX) };
      ENUMERATE_ (Cell, icell, cdm_x.inPatchCells()) {
        for (Node node : icell->nodes()) {
          if (var_test[node] != val) {
            ARCANE_FATAL("A node of InPatchCell is not InPatchNode -- NodeUID : {0} -- CellUID : {1} -- Val : {2} -- PatchIndex : {3}",
              node.uniqueId(), icell->uniqueId(), var_test[node], patch.index());
          }
        }
      }
      CellDirectionMng cdm_y{ patch.cellDirection(MD_DirY) };
      ENUMERATE_ (Cell, icell, cdm_y.inPatchCells()) {
        for (Node node : icell->nodes()) {
          if (var_test[node] != val) {
            ARCANE_FATAL("A node of InPatchCell is not InPatchNode -- NodeUID : {0} -- CellUID : {1} -- Val : {2} -- PatchIndex : {3}",
              node.uniqueId(), icell->uniqueId(), var_test[node], patch.index());
          }
        }
      }
      if (dimension == 3) {
        CellDirectionMng cdm_z{ patch.cellDirection(MD_DirZ) };
        ENUMERATE_ (Cell, icell, cdm_z.inPatchCells()) {
          for (Node node : icell->nodes()) {
            if (var_test[node] != val) {
              ARCANE_FATAL("A node of InPatchCell is not InPatchNode -- NodeUID : {0} -- CellUID : {1} -- Val : {2} -- PatchIndex : {3}",
                node.uniqueId(), icell->uniqueId(), var_test[node], patch.index());
            }
          }
        }
      }
    }
    {
      constexpr Int16 val = 2;
      NodeDirectionMng ndm_y{ patch.nodeDirection(MD_DirY) };
      ENUMERATE_ (Node, inode, ndm_y.inPatchNodes()) {
        var_test[inode] = val;
      }

      CellDirectionMng cdm_x{ patch.cellDirection(MD_DirX) };
      ENUMERATE_ (Cell, icell, cdm_x.inPatchCells()) {
        for (Node node : icell->nodes()) {
          if (var_test[node] != val) {
            ARCANE_FATAL("A node of InPatchCell is not InPatchNode -- NodeUID : {0} -- CellUID : {1} -- Val : {2} -- PatchIndex : {3}",
              node.uniqueId(), icell->uniqueId(), var_test[node], patch.index());
          }
        }
      }
      CellDirectionMng cdm_y{ patch.cellDirection(MD_DirY) };
      ENUMERATE_ (Cell, icell, cdm_y.inPatchCells()) {
        for (Node node : icell->nodes()) {
          if (var_test[node] != val) {
            ARCANE_FATAL("A node of InPatchCell is not InPatchNode -- NodeUID : {0} -- CellUID : {1} -- Val : {2} -- PatchIndex : {3}",
              node.uniqueId(), icell->uniqueId(), var_test[node], patch.index());
          }
        }
      }
      if (dimension == 3) {
        CellDirectionMng cdm_z{ patch.cellDirection(MD_DirZ) };
        ENUMERATE_ (Cell, icell, cdm_z.inPatchCells()) {
          for (Node node : icell->nodes()) {
            if (var_test[node] != val) {
              ARCANE_FATAL("A node of InPatchCell is not InPatchNode -- NodeUID : {0} -- CellUID : {1} -- Val : {2} -- PatchIndex : {3}",
                node.uniqueId(), icell->uniqueId(), var_test[node], patch.index());
            }
          }
        }
      }
    }
    if (dimension == 3) {
      constexpr Int16 val = 3;
      NodeDirectionMng ndm_z{ patch.nodeDirection(MD_DirZ) };
      ENUMERATE_ (Node, inode, ndm_z.inPatchNodes()) {
        var_test[inode] = val;
      }

      CellDirectionMng cdm_x{ patch.cellDirection(MD_DirX) };
      ENUMERATE_ (Cell, icell, cdm_x.inPatchCells()) {
        for (Node node : icell->nodes()) {
          if (var_test[node] != val) {
            ARCANE_FATAL("A node of InPatchCell is not InPatchNode -- NodeUID : {0} -- CellUID : {1} -- Val : {2} -- PatchIndex : {3}",
              node.uniqueId(), icell->uniqueId(), var_test[node], patch.index());
          }
        }
      }
      CellDirectionMng cdm_y{ patch.cellDirection(MD_DirY) };
      ENUMERATE_ (Cell, icell, cdm_y.inPatchCells()) {
        for (Node node : icell->nodes()) {
          if (var_test[node] != val) {
            ARCANE_FATAL("A node of InPatchCell is not InPatchNode -- NodeUID : {0} -- CellUID : {1} -- Val : {2} -- PatchIndex : {3}",
              node.uniqueId(), icell->uniqueId(), var_test[node], patch.index());
          }
        }
      }
      CellDirectionMng cdm_z{ patch.cellDirection(MD_DirZ) };
      ENUMERATE_ (Cell, icell, cdm_z.inPatchCells()) {
        for (Node node : icell->nodes()) {
          if (var_test[node] != val) {
            ARCANE_FATAL("A node of InPatchCell is not InPatchNode -- NodeUID : {0} -- CellUID : {1} -- Val : {2} -- PatchIndex : {3}",
              node.uniqueId(), icell->uniqueId(), var_test[node], patch.index());
          }
        }
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AMRPatchTesterModule::
_test1_2()
{
  Integer dimension = mesh()->dimension();

  VariableFaceInt16 var_test(VariableBuildInfo(mesh(), "VarTest"));

  if (dimension == 2) {
    for (Integer p = 0; p < m_cartesian_mesh->nbPatch(); ++p) {
      auto patch = m_cartesian_mesh->amrPatch(p);

      FaceDirectionMng fdm_x{ patch.faceDirection(MD_DirX) };
      ENUMERATE_ (Face, iface, fdm_x.inPatchFaces()) {
        var_test[iface] = MD_DirX;
      }
      FaceDirectionMng fdm_y{ patch.faceDirection(MD_DirY) };
      ENUMERATE_ (Face, iface, fdm_y.inPatchFaces()) {
        var_test[iface] = MD_DirY;
      }

      CellDirectionMng cdm_x{ patch.cellDirection(MD_DirX) };
      ENUMERATE_ (Cell, icell, cdm_x.inPatchCells()) {
        Integer index = 0;
        for (Face face : icell->faces()) {
          Integer expected = (index++ % 2 == 0 ? MD_DirY : MD_DirX);
          if (var_test[face] != expected) {
            ARCANE_FATAL("A face of InPatchCell is not InPatchFace -- FaceUID : {0} -- CellUID : {1} -- Expected : {2} -- Found : {3}",
                         face.uniqueId(), icell->uniqueId(), expected, var_test[face]);
          }
        }
      }
    }
  }
  else {
    for (Integer p = 0; p < m_cartesian_mesh->nbPatch(); ++p) {
      auto patch = m_cartesian_mesh->amrPatch(p);

      FaceDirectionMng fdm_x{ patch.faceDirection(MD_DirX) };
      ENUMERATE_ (Face, iface, fdm_x.inPatchFaces()) {
        var_test[iface] = MD_DirX;
      }
      FaceDirectionMng fdm_y{ patch.faceDirection(MD_DirY) };
      ENUMERATE_ (Face, iface, fdm_y.inPatchFaces()) {
        var_test[iface] = MD_DirY;
      }
      FaceDirectionMng fdm_z{ patch.faceDirection(MD_DirZ) };
      ENUMERATE_ (Face, iface, fdm_z.inPatchFaces()) {
        var_test[iface] = MD_DirZ;
      }

      CellDirectionMng cdm_x{ patch.cellDirection(MD_DirX) };
      ENUMERATE_ (Cell, icell, cdm_x.inPatchCells()) {
        Integer index = 0;
        for (Face face : icell->faces()) {
          Integer expected = -1;
          if (index % 3 == 0) {
            expected = MD_DirZ;
          }
          else if (index % 3 == 1) {
            expected = MD_DirX;
          }
          else {
            expected = MD_DirY;
          }
          index++;
          if (var_test[face] != expected) {
            ARCANE_FATAL("A face of InPatchCell is not InPatchFace -- FaceUID : {0} -- CellUID : {1} -- Expected : {2} -- Found : {3}",
                         face.uniqueId(), icell->uniqueId(), expected, var_test[face]);
          }
        }
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AMRPatchTesterModule::
testMarkCellsToRefine(Integer level)
{
  CartesianMeshNumberingMng numbering(m_cartesian_mesh);

  constexpr Integer choose = 0;

  ENUMERATE_ (Cell, icell, mesh()->allLevelCells(level)) {
    const Integer pos_x = numbering.offsetLevelToLevel(numbering.cellUniqueIdToCoordX(*icell), level, 0);
    const Integer pos_y = numbering.offsetLevelToLevel(numbering.cellUniqueIdToCoordY(*icell), level, 0);

    if constexpr (choose == 0) {
      if (pos_x >= 2 && pos_x < 6 && pos_y >= 2 && pos_y < 5) {
        icell->mutableItemBase().addFlags(ItemFlags::II_Refine);
      }
      if (pos_x >= 7 && pos_x < 11 && pos_y >= 6 && pos_y < 9) {
        icell->mutableItemBase().addFlags(ItemFlags::II_Refine);
      }
    }

    if constexpr (choose == 1) {
      if (pos_x >= 2 && pos_x < 6 && pos_y >= 2 && pos_y < 5) {
        icell->mutableItemBase().addFlags(ItemFlags::II_Refine);
      }
      if (pos_x >= 6 && pos_x < 10 && pos_y >= 4 && pos_y < 7) {
        icell->mutableItemBase().addFlags(ItemFlags::II_Refine);
      }
    }

    if constexpr (choose == 2) {
      if (pos_x >= 6 && pos_x < 26 && pos_y >= 4 && pos_y < 7) {
        icell->mutableItemBase().addFlags(ItemFlags::II_Refine);
      }
      if (pos_x >= 7 && pos_x < 12 && pos_y >= 10 && pos_y < 26) {
        icell->mutableItemBase().addFlags(ItemFlags::II_Refine);
      }
    }

    if constexpr (choose == 3) {
      if (pos_x >= 3 && pos_x < 11 && pos_y >= 25 && pos_y < 37) {
        icell->mutableItemBase().addFlags(ItemFlags::II_Refine);
      }

      if (pos_x >= 19 && pos_x < 27 && pos_y >= 2 && pos_y < 19) {
        icell->mutableItemBase().addFlags(ItemFlags::II_Refine);
      }
      if (pos_x >= 19 && pos_x < 27 && pos_y >= 43 && pos_y < 60) {
        icell->mutableItemBase().addFlags(ItemFlags::II_Refine);
      }

      if (pos_x >= 5 && pos_x < 12 && pos_y >= 19 && pos_y < 29) {
        icell->mutableItemBase().addFlags(ItemFlags::II_Refine);
      }
      if (pos_x >= 7 && pos_x < 13 && pos_y >= 17 && pos_y < 26) {
        icell->mutableItemBase().addFlags(ItemFlags::II_Refine);
      }
      if (pos_x >= 9 && pos_x < 15 && pos_y >= 15 && pos_y < 23) {
        icell->mutableItemBase().addFlags(ItemFlags::II_Refine);
      }
      if (pos_x >= 11 && pos_x < 16 && pos_y >= 13 && pos_y < 22) {
        icell->mutableItemBase().addFlags(ItemFlags::II_Refine);
      }
      if (pos_x >= 15 && pos_x < 18 && pos_y >= 11 && pos_y < 21) {
        icell->mutableItemBase().addFlags(ItemFlags::II_Refine);
      }
      if (pos_x >= 18 && pos_x < 21 && pos_y >= 11 && pos_y < 20) {
        icell->mutableItemBase().addFlags(ItemFlags::II_Refine);
      }

      if (pos_x >= 5 && pos_x < 12 && pos_y >= 33 && pos_y < 43) {
        icell->mutableItemBase().addFlags(ItemFlags::II_Refine);
      }
      if (pos_x >= 7 && pos_x < 13 && pos_y >= 36 && pos_y < 45) {
        icell->mutableItemBase().addFlags(ItemFlags::II_Refine);
      }
      if (pos_x >= 9 && pos_x < 15 && pos_y >= 39 && pos_y < 47) {
        icell->mutableItemBase().addFlags(ItemFlags::II_Refine);
      }
      if (pos_x >= 11 && pos_x < 16 && pos_y >= 40 && pos_y < 49) {
        icell->mutableItemBase().addFlags(ItemFlags::II_Refine);
      }
      if (pos_x >= 15 && pos_x < 18 && pos_y >= 41 && pos_y < 51) {
        icell->mutableItemBase().addFlags(ItemFlags::II_Refine);
      }
      if (pos_x >= 18 && pos_x < 21 && pos_y >= 42 && pos_y < 51) {
        icell->mutableItemBase().addFlags(ItemFlags::II_Refine);
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AMRPatchTesterModule::
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

ARCANE_REGISTER_MODULE_AMRPATCHTESTER(AMRPatchTesterModule);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace ArcaneTest

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
