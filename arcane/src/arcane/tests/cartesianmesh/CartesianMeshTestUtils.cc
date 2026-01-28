// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CartesianMeshTestUtils.cc                                   (C) 2000-2026 */
/*                                                                           */
/* Fonctions utilitaires pour les tests de 'CartesianMesh'.                  */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/tests/cartesianmesh/CartesianMeshTestUtils.h"

#include "arcane/utils/CheckedConvert.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/Real2.h"
#include "arcane/utils/ValueChecker.h"

#include "arcane/core/MeshUtils.h"

#include "arcane/core/IMesh.h"
#include "arcane/core/IItemFamily.h"
#include "arcane/core/ItemPrinter.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/IMeshModifier.h"
#include "arcane/core/IMeshUtilities.h"
#include "arcane/core/SimpleSVGMeshExporter.h"
#include "arcane/core/UnstructuredMeshConnectivity.h"

#if defined(ARCANE_HAS_ACCELERATOR_API)
#include "arcane/accelerator/Runner.h"
#include "arcane/accelerator/RunCommandEnumerate.h"
#include "arcane/accelerator/VariableViews.h"
#endif
#include "arcane/accelerator/core/IAcceleratorMng.h"

#include "arcane/cartesianmesh/ICartesianMesh.h"
#include "arcane/cartesianmesh/CellDirectionMng.h"
#include "arcane/cartesianmesh/FaceDirectionMng.h"
#include "arcane/cartesianmesh/NodeDirectionMng.h"
#include "arcane/cartesianmesh/CartesianConnectivity.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace ArcaneTest
{

using namespace Arcane;
using namespace Arcane::Accelerator;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartesianMeshTestUtils::
CartesianMeshTestUtils(ICartesianMesh* cm, IAcceleratorMng* am)
: TraceAccessor(cm->traceMng())
, m_cartesian_mesh(cm)
, m_mesh(cm->mesh())
, m_accelerator_mng(am)
, m_cell_center(VariableBuildInfo(m_mesh, "CellCenter"))
, m_face_center(VariableBuildInfo(m_mesh, "FaceCenter"))
, m_node_density(VariableBuildInfo(m_mesh, "NodeDensity"))
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartesianMeshTestUtils::
~CartesianMeshTestUtils()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshTestUtils::
testAll(bool is_amr)
{
  m_is_amr = is_amr;
  _testDirCell();
  _testDirCellAccelerator();
  _testDirFace();
  _testDirFaceAccelerator();
  _testDirNode();
  _testDirNodeAccelerator();
  _testDirCellNode();
  _testDirCellNodeAccelerator();
  _testDirCellFace();
  _testDirCellFaceAccelerator();
  if (m_mesh->dimension() == 3) {
    _testNodeToCellConnectivity3D();
    _testNodeToCellConnectivity3DAccelerator();
    _testCellToNodeConnectivity3D();
    _testCellToNodeConnectivity3DAccelerator();
  }
  else {
    _testNodeToCellConnectivity2D();
    _testCellToNodeConnectivity2D();
    _saveSVG();
  }
  _testConnectivityByDirection();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshTestUtils::
_computeCenters()
{
  IMesh* mesh = m_mesh;

  // Calcule le centre des mailles
  {
    VariableNodeReal3& nodes_coord = mesh->nodesCoordinates();
    ENUMERATE_CELL (icell, m_mesh->allCells()) {
      Cell cell = *icell;
      Real3 center;
      for (NodeLocalId inode : cell.nodeIds() )
        center += nodes_coord[inode];
      center /= cell.nbNode();
      m_cell_center[icell] = center;
    }
  }

  // Calcule le centre des faces
  {
    VariableNodeReal3& nodes_coord = mesh->nodesCoordinates();
    ENUMERATE_FACE (iface, m_mesh->allFaces()) {
      Face face = *iface;
      Real3 center;
      for (NodeLocalId inode : face.nodeIds() )
        center += nodes_coord[inode];
      center /= face.nbNode();
      m_face_center[iface] = center;
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshTestUtils::
_checkSameId(FaceLocalId item, FaceLocalId local_id)
{
  if (item != local_id)
    ARCANE_FATAL("Bad FaceLocalId item={0} local_id={1}", item, local_id);
}

void CartesianMeshTestUtils::
_checkSameId(CellLocalId item, CellLocalId local_id)
{
  if (item != local_id)
    ARCANE_FATAL("Bad CellLocalId item={0} local_id={1}", item, local_id);
}

void CartesianMeshTestUtils::
_checkSameId(NodeLocalId item, NodeLocalId local_id)
{
  if (item.localId() != local_id)
    ARCANE_FATAL("Bad NodeLocalId item={0} local_id={1}", item, local_id);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshTestUtils::
_checkItemGroupIsSorted(const ItemGroup& group)
{
  if (!group.checkIsSorted())
    ARCANE_FATAL("Node direction group '{0}' is not sorted", group.name());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshTestUtils::
_testDirCell()
{
  info() << "TEST_DIR_CELL";

  IMesh* mesh = m_mesh;
  Integer nb_dir = mesh->dimension();
  Integer nb_print = m_nb_print;
  CellDirectionMng cdm2;
  CellDirectionMng cdm3;
  for (Integer idir = 0; idir < nb_dir; ++idir) {
    CellDirectionMng cdm(m_cartesian_mesh->cellDirection(idir));
    cdm2 = m_cartesian_mesh->cellDirection(idir);
    cdm3 = cdm;
    eMeshDirection md = cdm.direction();
    Integer iprint = 0;
    info() << "DIRECTION=" << idir << " Cells=" << cdm.allCells().name();
    _checkItemGroupIsSorted(cdm.allCells());
    ENUMERATE_CELL (icell, cdm.allCells()) {
      Cell cell = *icell;
      DirCell dir_cell(cdm[icell]);
      Cell prev_cell = dir_cell.previous();
      Cell next_cell = dir_cell.next();
      // AH: Désactivation de ce if : bloquant s'il n'y a qu'une seule couche de mailles.
      // if (prev_cell.null() && next_cell.null())
      //   ARCANE_FATAL("Null previous and next cell");
      DirCell dir_cell2(cdm2[icell]);
      Cell prev_cell2 = dir_cell2.previous();
      Cell next_cell2 = dir_cell2.next();
      DirCell dir_cell3(cdm3[icell]);
      Cell prev_cell3 = dir_cell3.previous();
      Cell next_cell3 = dir_cell3.next();
      _checkSameId(prev_cell, prev_cell2);
      _checkSameId(next_cell, next_cell2);
      _checkSameId(prev_cell, prev_cell3);
      _checkSameId(next_cell, next_cell3);
      if (nb_print < 0 || iprint < nb_print) {
        ++iprint;
        if (!prev_cell.null() && !next_cell.null()) {
          info() << "Cell uid=" << ItemPrinter(cell) << " dir=" << md
                 << " prev=" << ItemPrinter(prev_cell) << " xyz=" << m_cell_center[prev_cell]
                 << " next=" << ItemPrinter(next_cell) << " xyz=" << m_cell_center[next_cell];
        }
        else {
          info() << "Cell uid=" << ItemPrinter(cell) << " dir=" << md;
          if (!prev_cell.null())
            info() << " prev=" << ItemPrinter(prev_cell) << " xyz=" << m_cell_center[prev_cell];
          if (!next_cell.null())
            info() << " next=" << ItemPrinter(next_cell) << " xyz=" << m_cell_center[next_cell];
        }
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshTestUtils::
_testDirCellAccelerator()
{
#if defined(ARCANE_HAS_ACCELERATOR_API)
  info() << "TEST_DIR_CELL_ACCELERATOR";

  IMesh* mesh = m_mesh;
  Integer nb_dir = mesh->dimension();
  CellDirectionMng cdm2;
  CellDirectionMng cdm3;

  auto queue = m_accelerator_mng->queue();

  VariableCellInt32 dummy_var(VariableBuildInfo(mesh, "DummyCellVariable"));
  dummy_var.fill(0);

  for (Integer idir = 0; idir < nb_dir; ++idir) {
    CellDirectionMng cdm(m_cartesian_mesh->cellDirection(idir));
    cdm2 = m_cartesian_mesh->cellDirection(idir);
    cdm3 = cdm;
    info() << "ACCELERATOR_DIRECTION=" << idir << " Cells=" << cdm.allCells().name();
    _checkItemGroupIsSorted(cdm.allCells());
    auto command = makeCommand(queue);
    auto inout_dummy_var = viewInOut(command, dummy_var);
    command << RUNCOMMAND_ENUMERATE(Cell, icell, cdm.allCells())
    {
      DirCellLocalId dir_cell(cdm.dirCellId(icell));
      CellLocalId prev_cell = dir_cell.previous();
      CellLocalId next_cell = dir_cell.next();
      // AH: Désactivation de ce if : bloquant s'il n'y a qu'une seule couche de mailles.
      // if (prev_cell.isNull() && next_cell.isNull()) {
      //   inout_dummy_var[icell] = -5;
      //   return;
      // }

      DirCellLocalId dir_cell2(cdm2.dirCellId(icell));
      CellLocalId prev_cell2 = dir_cell2.previous();
      CellLocalId next_cell2 = dir_cell2.next();
      DirCellLocalId dir_cell3(cdm3.dirCellId(icell));
      CellLocalId prev_cell3 = dir_cell3.previous();
      CellLocalId next_cell3 = dir_cell3.next();
      if (prev_cell != prev_cell2)
        inout_dummy_var[icell] = -10;
      if (next_cell != next_cell2)
        inout_dummy_var[icell] = -11;
      if (prev_cell != prev_cell3)
        inout_dummy_var[icell] = -12;
      if (next_cell != next_cell3)
        inout_dummy_var[icell] = -13;

      if (!prev_cell.isNull() && !next_cell.isNull()) {
        inout_dummy_var[icell] = 2;
      }
      else {
        if (!prev_cell.isNull())
          inout_dummy_var[icell] = inout_dummy_var[icell] + 1;
        if (!next_cell.isNull())
          inout_dummy_var[icell] = inout_dummy_var[icell] + 1;
      }
    };
    ENUMERATE_ (Cell, icell, cdm.allCells()) {
      if (dummy_var[icell] < 0)
        ARCANE_FATAL("Bad value for dummy_var id={0} v={1}", ItemPrinter(*icell), dummy_var[icell]);
    }
  }
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshTestUtils::
_testDirFace()
{
  IMesh* mesh = m_mesh;
  Integer nb_dir = mesh->dimension();
  for (Integer idir = 0; idir < nb_dir; ++idir)
    _testDirFace(idir);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshTestUtils::
_testDirFaceAccelerator()
{
  IMesh* mesh = m_mesh;
  Integer nb_dir = mesh->dimension();
  for (Integer idir = 0; idir < nb_dir; ++idir)
    _testDirFaceAccelerator(idir);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshTestUtils::
_testDirFace(int idir)
{
  // Teste l'utilisation de DirFace et vérifie que
  // les valeurs sont correctes: pour une face donnée,
  // il faut que selon la direction choisie, la coordonnée du centre
  // de la maille avant (previousCell()) soit inférieure à celle
  // de la maille après (nextCell()) et aussi celle de la maille avant
  // soit inférieure à ce
  Integer nb_print = m_nb_print;
  Integer nb_error = 0;
  Integer iprint = 0;

  FaceDirectionMng fdm2;
  FaceDirectionMng fdm(m_cartesian_mesh->faceDirection(idir));
  fdm2 = fdm;
  eMeshDirection md = fdm.direction();
  info() << "TEST_DIR_FACE for direction=" << idir << " -> " << (eMeshDirection)idir;
  _checkItemGroupIsSorted(fdm.allFaces());
  ENUMERATE_FACE (iface, fdm.allFaces()) {
    Face face = *iface;
    DirFace dir_face(fdm[iface]);
    DirFace dir_face2(fdm2[iface]);
    //Integer nb_node = cell.nbNode();
    Cell prev_cell = dir_face.previousCell();
    Cell next_cell = dir_face.nextCell();
    Cell prev_cell2 = dir_face2.previousCell();
    Cell next_cell2 = dir_face2.nextCell();
    _checkSameId(prev_cell, prev_cell2);
    _checkSameId(next_cell, next_cell2);
    _checkSameId(prev_cell, dir_face.previousCellId());
    _checkSameId(next_cell, dir_face.nextCellId());
    bool is_print = (nb_print < 0 || iprint < nb_print);
    ++iprint;
    Real face_coord = m_face_center[iface][idir];
    if (!prev_cell.null() && !next_cell.null()) {
      Real next_coord = m_cell_center[next_cell][idir];
      Real prev_coord = m_cell_center[prev_cell][idir];
      if (next_coord < prev_coord) {
        info() << "Bad ordering for face";
        is_print = true;
        ++nb_error;
      }
      if (is_print)
        info() << "Face uid=" << ItemPrinter(face) << " dir=" << md << " xyz=" << m_face_center[iface]
               << " prev=" << prev_cell.uniqueId() << " xyz=" << m_cell_center[prev_cell]
               << " next=" << next_cell.uniqueId() << " xyz=" << m_cell_center[next_cell];
    }
    else {
      if (!prev_cell.null()) {
        Real prev_coord = m_cell_center[prev_cell][idir];
        if (face_coord < prev_coord) {
          info() << "Bad ordering for face";
          is_print = true;
          ++nb_error;
        }
        if (is_print)
          info() << "Face uid=" << ItemPrinter(face) << " dir=" << md << " xyz=" << m_face_center[iface]
                 << " prev=" << prev_cell.uniqueId() << " xyz=" << m_cell_center[prev_cell];
      }
      if (!next_cell.null()) {
        Real next_coord = m_cell_center[next_cell][idir];
        if (next_coord < face_coord) {
          info() << "Bad ordering for face";
          is_print = true;
          ++nb_error;
        }
        if (is_print)
          info() << "Face uid=" << ItemPrinter(face) << " dir=" << md << " xyz=" << m_face_center[iface]
                 << " next=" << next_cell.uniqueId() << " xyz=" << m_cell_center[next_cell];
      }
    }
  }
  if (nb_error != 0)
    ARCANE_FATAL("Bad connectivity for DirFace nb_error={0}", nb_error);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshTestUtils::
_testDirFaceAccelerator(int idir)
{
#if defined(ARCANE_HAS_ACCELERATOR_API)
  // Teste l'utilisation de DirFace et vérifie que
  // les valeurs sont correctes: pour une face donnée,
  // il faut que selon la direction choisie, la coordonnée du centre
  // de la maille avant (previousCell()) soit inférieure à celle
  // de la maille après (nextCell()) et aussi celle de la maille avant
  // soit inférieure à ce

  FaceDirectionMng fdm2;
  FaceDirectionMng fdm(m_cartesian_mesh->faceDirection(idir));
  fdm2 = fdm;
  info() << "TEST_DIR_FACE_ACCELERATOR for direction=" << idir << " -> " << (eMeshDirection)idir;

  auto queue = m_accelerator_mng->defaultQueue();
  auto command = makeCommand(*queue);

  VariableFaceInt32 dummy_var(VariableBuildInfo(m_mesh, "DummyFaceVariable"));
  dummy_var.fill(0);
  auto inout_dummy_var = viewInOut(command, dummy_var);
  auto in_face_center = viewIn(command, m_face_center);
  auto in_cell_center = viewIn(command, m_cell_center);

  command << RUNCOMMAND_ENUMERATE(Face, iface, fdm.allFaces())
  {
    DirFaceLocalId dir_face(fdm.dirFaceId(iface));
    DirFaceLocalId dir_face2(fdm2.dirFaceId(iface));
    //Integer nb_node = cell.nbNode();
    CellLocalId prev_cell = dir_face.previousCell();
    CellLocalId next_cell = dir_face.nextCell();
    CellLocalId prev_cell2 = dir_face2.previousCell();
    CellLocalId next_cell2 = dir_face2.nextCell();
    if (prev_cell != prev_cell2)
      inout_dummy_var[iface] = -10;
    if (next_cell != next_cell2)
      inout_dummy_var[iface] = -11;
    if (prev_cell != dir_face.previousCellId())
      inout_dummy_var[iface] = -12;
    if (next_cell != dir_face.nextCellId())
      inout_dummy_var[iface] = -13;
    if (inout_dummy_var[iface] < 0)
      return;

    Real face_coord = in_face_center[iface][idir];
    if (!prev_cell.isNull() && !next_cell.isNull()) {
      Real next_coord = in_cell_center[next_cell][idir];
      Real prev_coord = in_cell_center[prev_cell][idir];
      if (next_coord < prev_coord) {
        inout_dummy_var[iface] = -20;
        return;
      }
    }
    else {
      if (!prev_cell.isNull()) {
        Real prev_coord = in_cell_center[prev_cell][idir];
        if (face_coord < prev_coord) {
          inout_dummy_var[iface] = -21;
          return;
        }
      }
      if (!next_cell.isNull()) {
        Real next_coord = in_cell_center[next_cell][idir];
        if (next_coord < face_coord) {
          inout_dummy_var[iface] = -22;
          return;
        }
      }
    }
  };

  ENUMERATE_ (Face, iface, fdm.allFaces()) {
    if (dummy_var[iface] < 0)
      ARCANE_FATAL("Bad value for dummy_var id={0} v={1}", ItemPrinter(*iface), dummy_var[iface]);
  }
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshTestUtils::
_testDirNode()
{
  info() << "TEST_DIR_NODE";
  m_node_density.fill(0.0);

  IMesh* mesh = m_mesh;
  Integer nb_dir = mesh->dimension();
  Integer nb_print = m_nb_print;
  Integer nb_error = 0;
  VariableNodeReal3& nodes_coord = mesh->nodesCoordinates();
  // Pour tester l'opérateur 'operator='
  NodeDirectionMng node_dm2;
  NodeDirectionMng node_dm3;
  for (Integer idir = 0; idir < nb_dir; ++idir) {
    NodeDirectionMng node_dm(m_cartesian_mesh->nodeDirection(idir));
    node_dm2 = m_cartesian_mesh->nodeDirection(idir);
    node_dm3 = node_dm;
    eMeshDirection md = node_dm.direction();
    Integer iprint = 0;
    info() << "DIRECTION=" << idir;
    NodeGroup dm_all_nodes = node_dm.allNodes();
    _checkItemGroupIsSorted(dm_all_nodes);
    ENUMERATE_NODE (inode, dm_all_nodes) {
      Node node = *inode;
      m_node_density[inode] += 1.0;
      DirNode dir_node(node_dm[inode]);
      DirNode dir_node2(node_dm2[inode]);
      DirNode dir_node3(node_dm3[inode]);
      Node prev_node = dir_node.previous();
      Node next_node = dir_node.next();
      Node prev_node2 = dir_node2.previous();
      Node next_node2 = dir_node2.next();
      Node prev_node3 = dir_node3.previous();
      Node next_node3 = dir_node3.next();
      _checkSameId(prev_node, prev_node2);
      _checkSameId(next_node, next_node2);
      _checkSameId(prev_node, prev_node3);
      _checkSameId(next_node, next_node3);
      _checkSameId(prev_node, dir_node.previousId());
      _checkSameId(next_node, dir_node.nextId());
      Real my_coord = nodes_coord[inode][idir];
      bool is_print = (nb_print < 0 || iprint < nb_print);
      ++iprint;
      if (is_print) {
        Int32 node_nb_cell = node.nbCell();
        info() << "DirNode node= " << ItemPrinter(node) << " nb_cell=" << node_nb_cell << " pos=" << nodes_coord[node];
        for (Integer k = 0; k < node_nb_cell; ++k) {
          Real3 cell_pos = m_cell_center[node.cell(k)];
          info() << "Node k=" << k << " cell_pos=" << cell_pos << " cell=" << ItemPrinter(node.cell(k));
        }
        for (Integer k = 0; k < 8; ++k) {
          Int32 cell_index = dir_node.cellIndex(k);
          Real3 cell_pos;
          if (cell_index != (-1)) {
            if ((1 + cell_index) > node_nb_cell)
              ARCANE_FATAL("Bad value for cell_index '{0}' node_nb_cell={1}", cell_index, node_nb_cell);
            cell_pos = m_cell_center[node.cell(cell_index)];
          }
          info() << "DirNode cellIndex k=" << k << " index=" << cell_index << " pos=" << cell_pos
                 << " cell_lid=" << dir_node.cellId(k)
                 << " cell=" << dir_node.cell(k);
        }
        info() << "DirNode direct "
               << " " << dir_node.nextLeftCellId()
               << " " << dir_node.nextRightCellId()
               << " " << dir_node.previousRightCellId()
               << " " << dir_node.previousLeftCellId()
               << " " << dir_node.topNextLeftCellId()
               << " " << dir_node.topNextRightCellId()
               << " " << dir_node.topPreviousRightCellId()
               << " " << dir_node.topPreviousLeftCellId();
        info() << "DirNode direct "
               << " " << ItemPrinter(dir_node.nextLeftCell())
               << " " << ItemPrinter(dir_node.nextRightCell())
               << " " << ItemPrinter(dir_node.previousRightCell())
               << " " << ItemPrinter(dir_node.previousLeftCell())
               << " " << ItemPrinter(dir_node.topNextLeftCell())
               << " " << ItemPrinter(dir_node.topNextRightCell())
               << " " << ItemPrinter(dir_node.topPreviousRightCell())
               << " " << ItemPrinter(dir_node.topPreviousLeftCell());
      }
      if (prev_node.null() && next_node.null())
        ARCANE_FATAL("Null previous and next node for node {0}", ItemPrinter(node));
      //TODO: Vérifier que les coordonnées autres que celle de idir sont bien les mêmes pour next,prev et my_coord.
      if (!prev_node.null() && !next_node.null()) {
        Real next_coord = nodes_coord[next_node][idir];
        Real prev_coord = nodes_coord[prev_node][idir];
        if (next_coord < prev_coord) {
          info() << "Bad ordering for node";
          is_print = true;
          ++nb_error;
        }
        if (is_print)
          info() << "Node uid=" << ItemPrinter(node) << " dir=" << md
                 << " prev=" << ItemPrinter(prev_node) << " xyz=" << prev_coord
                 << " next=" << ItemPrinter(next_node) << " xyz=" << next_coord;
      }
      else {
        if (!prev_node.null()) {
          Real prev_coord = nodes_coord[prev_node][idir];
          if (my_coord < prev_coord) {
            info() << "Bad ordering for node";
            is_print = true;
            ++nb_error;
          }
          if (is_print)
            info() << "Node uid=" << ItemPrinter(node) << " dir=" << md
                   << " prev=" << ItemPrinter(prev_node) << " xyz=" << nodes_coord[prev_node];
        }
        if (!next_node.null()) {
          Real next_coord = nodes_coord[next_node][idir];
          if (next_coord < my_coord) {
            info() << "Bad ordering for node";
            is_print = true;
            ++nb_error;
          }
          if (is_print)
            info() << "Node uid=" << ItemPrinter(node) << " dir=" << md
                   << " next=" << ItemPrinter(next_node) << " xyz=" << nodes_coord[next_node];
        }
      }
    }
  }

  if (nb_error != 0)
    ARCANE_FATAL("Bad connectivity for DirNode");

  // Vérifie qu'on a parcouru tous les noeuds
  ENUMERATE_NODE (inode, m_mesh->allNodes()) {
    Node node = *inode;
    if (node.cell(0).level() != 0)
      continue;
    if (m_node_density[inode] == 0.0) {
      ++nb_error;
      info() << "Bad value for node " << ItemPrinter(node);
    }
  }
  if (nb_error != 0)
    ARCANE_FATAL("Bad values for DirNode");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshTestUtils::
_testDirNodeAccelerator()
{
#if defined(ARCANE_HAS_ACCELERATOR_API)
  info() << "TEST_DIR_NODE Accelerator";
  m_node_density.fill(0.0);

  IMesh* mesh = m_mesh;
  Integer nb_dir = mesh->dimension();
  VariableNodeReal3& nodes_coord = mesh->nodesCoordinates();
  // Pour tester l'opérateur 'operator='
  NodeDirectionMng node_dm2;
  NodeDirectionMng node_dm3;

  //auto in_face_center = viewIn(command, m_face_center);
  //auto in_cell_center = viewIn(command, m_cell_center);

  VariableNodeInt32 dummy_var(VariableBuildInfo(m_mesh, "DummyNodeVariable"));

  UnstructuredMeshConnectivityView m_connectivity_view;
  m_connectivity_view.setMesh(mesh);
  auto cnc = m_connectivity_view.nodeCell();

  for (Integer idir = 0; idir < nb_dir; ++idir) {
    NodeDirectionMng node_dm(m_cartesian_mesh->nodeDirection(idir));
    node_dm2 = m_cartesian_mesh->nodeDirection(idir);
    node_dm3 = node_dm;
    info() << "DIRECTION=" << idir;
    NodeGroup dm_all_nodes = node_dm.allNodes();
    _checkItemGroupIsSorted(dm_all_nodes);

    auto queue = m_accelerator_mng->defaultQueue();
    auto command = makeCommand(*queue);

    dummy_var.fill(0);
    auto inout_dummy_var = viewInOut(command, dummy_var);
    auto in_nodes_coord = viewInOut(command, nodes_coord);

    command << RUNCOMMAND_ENUMERATE(Node, node, dm_all_nodes)
    {
      DirNodeLocalId dir_node(node_dm.dirNodeId(node));
      DirNodeLocalId dir_node2(node_dm2.dirNodeId(node));
      DirNodeLocalId dir_node3(node_dm3.dirNodeId(node));
      NodeLocalId prev_node = dir_node.previous();
      NodeLocalId next_node = dir_node.next();
      NodeLocalId prev_node2 = dir_node2.previous();
      NodeLocalId next_node2 = dir_node2.next();
      NodeLocalId prev_node3 = dir_node3.previous();
      NodeLocalId next_node3 = dir_node3.next();
      Int32 vx = 0;
      if (prev_node != prev_node2)
        vx = -10;
      if (next_node != next_node2)
        vx = -11;
      if (prev_node != prev_node3)
        vx = -12;
      if (next_node != next_node3)
        vx = -13;
      if (prev_node != dir_node.previousId())
        vx = -14;
      if (next_node != dir_node.nextId())
        vx = -15;
      inout_dummy_var[node] = vx;
      if (vx < 0)
        return;
      Real my_coord = in_nodes_coord[node][idir];
      {
        Int32 node_nb_cell = cnc.nbCell(node);
        for (Integer k = 0; k < 8; ++k) {
          Int32 cell_index = dir_node.cellIndex(k);
          if (cell_index != (-1)) {
            if ((1 + cell_index) > node_nb_cell) {
              inout_dummy_var[node] = -40 - k;
              return;
            }
          }
        }
        Int32 sum_id = 20 + dir_node.nextLeftCellId() +
        dir_node.nextRightCellId() +
        dir_node.previousRightCellId() +
        dir_node.previousLeftCellId() +
        dir_node.topNextLeftCellId() +
        dir_node.topNextRightCellId() +
        dir_node.topPreviousRightCellId() +
        dir_node.topPreviousLeftCellId();
        inout_dummy_var[node] = sum_id;
      }
      if (prev_node.isNull() && next_node.isNull()) {
        inout_dummy_var[node] = -20;
        return;
      }
      //TODO: Vérifier que les coordonnées autres que celle de idir sont bien les mêmes pour next,prev et my_coord.
      if (!prev_node.isNull() && !next_node.isNull()) {
        Real next_coord = in_nodes_coord[next_node][idir];
        Real prev_coord = in_nodes_coord[prev_node][idir];
        if (next_coord < prev_coord) {
          inout_dummy_var[node] = -30;
          return;
        }
      }
      else {
        if (!prev_node.isNull()) {
          Real prev_coord = in_nodes_coord[prev_node][idir];
          if (my_coord < prev_coord) {
            inout_dummy_var[node] = -31;
            return;
          }
        }
        if (!next_node.isNull()) {
          Real next_coord = in_nodes_coord[next_node][idir];
          if (next_coord < my_coord) {
            inout_dummy_var[node] = -32;
            return;
          }
        }
      }
    };

    ENUMERATE_ (Node, inode, node_dm.allNodes()) {
      if (dummy_var[inode] < 0)
        ARCANE_FATAL("Bad value for dummy_var id={0} v={1}", ItemPrinter(*inode), dummy_var[inode]);
    }
  }
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshTestUtils::
_testDirCellNode()
{
  IMesh* mesh = m_mesh;
  VariableNodeReal3& nodes_coord = mesh->nodesCoordinates();
  Integer nb_dir = mesh->dimension();

  Integer nb_print = m_nb_print;
  for (Integer idir = 0; idir < nb_dir; ++idir) {
    CellDirectionMng cdm(m_cartesian_mesh->cellDirection(idir));
    eMeshDirection md = cdm.direction();
    info() << "DIRECTION=" << idir;
    Integer iprint = 0;
    ENUMERATE_CELL (icell, cdm.allCells()) {
      Cell cell = *icell;
      CellLocalId cell_id(cell.localId());
      DirCellNode cn(cdm.cellNode(cell));
      DirCellNode cn2(cdm.cellNode(cell_id));
      if (cn.cellId() != cn2.cellId())
        ARCANE_FATAL("Bad DirCellNode");
      bool is_print = (nb_print < 0 || iprint < nb_print);
      ++iprint;
      if (is_print) {
        info() << "Cell uid=" << ItemPrinter(cell) << " dir=" << md;
        info() << "Cell nextLeft =" << ItemPrinter(cn.nextLeft()) << " xyz=" << nodes_coord[cn.nextLeft()];
        info() << "Cell nextRight=" << ItemPrinter(cn.nextRight()) << " xyz=" << nodes_coord[cn.nextRight()];
        info() << "Cell prevRight=" << ItemPrinter(cn.previousRight()) << " xyz=" << nodes_coord[cn.previousRight()];
        info() << "Cell prevLeft =" << ItemPrinter(cn.previousLeft()) << " xyz=" << nodes_coord[cn.previousLeft()];
        if (nb_dir == 3) {
          info() << "Cell topNextLeft =" << ItemPrinter(cn.topNextLeft()) << " xyz=" << nodes_coord[cn.topNextLeft()];
          info() << "Cell topNextRight=" << ItemPrinter(cn.topNextRight()) << " xyz=" << nodes_coord[cn.topNextRight()];
          info() << "Cell topPrevRight=" << ItemPrinter(cn.topPreviousRight()) << " xyz=" << nodes_coord[cn.topPreviousRight()];
          info() << "Cell topPrevLeft =" << ItemPrinter(cn.topPreviousLeft()) << " xyz=" << nodes_coord[cn.topPreviousLeft()];
        }
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshTestUtils::
_testDirCellNodeAccelerator()
{
#if defined(ARCANE_HAS_ACCELERATOR_API)
  IMesh* mesh = m_mesh;
  VariableNodeReal3& nodes_coord = mesh->nodesCoordinates();
  Integer nb_dir = mesh->dimension();

  VariableCellInt32 dummy_var(VariableBuildInfo(mesh, "DummyCellVariable"));
  VariableCellReal3 computed_center(VariableBuildInfo(mesh, "CellComputedCenter"));

  for (Integer idir = 0; idir < nb_dir; ++idir) {
    CellDirectionMng cdm(m_cartesian_mesh->cellDirection(idir));
    info() << "TEST_DIR_CELL_NODE_ACCELERATOR_DIRECTION=" << idir;

    auto queue = m_accelerator_mng->defaultQueue();
    auto command = makeCommand(*queue);

    auto inout_dummy_var = viewInOut(command, dummy_var);
    auto in_nodes_coord = viewIn(command, nodes_coord);
    auto out_computed_center = viewOut(command, computed_center);
    command << RUNCOMMAND_ENUMERATE(Cell, cell, cdm.allCells())
    {
      inout_dummy_var[cell] = 0;
      DirCellNodeLocalId cn(cdm.dirCellNodeId(cell));
      DirCellNodeLocalId cn2(cdm.dirCellNodeId(cell));
      if (cn.cellId() != cn2.cellId()) {
        inout_dummy_var[cell] = -10;
        return;
      }

      Real3 n1 = in_nodes_coord[cn.nextLeftId()];
      Real3 n2 = in_nodes_coord[cn.nextRightId()];
      Real3 n3 = in_nodes_coord[cn.previousRightId()];
      Real3 n4 = in_nodes_coord[cn.previousLeftId()];
      Real3 center = n1 + n2 + n3 + n4;
      if (nb_dir == 3) {
        Real3 n5 = in_nodes_coord[cn.topNextLeftId()];
        Real3 n6 = in_nodes_coord[cn.topNextRightId()];
        Real3 n7 = in_nodes_coord[cn.topPreviousRightId()];
        Real3 n8 = in_nodes_coord[cn.topPreviousLeftId()];
        center += n5 + n6 + n7 + n8;
        center /= 8.0;
      }
      else
        center /= 4.0;
      out_computed_center[cell] = center;
    };
    ENUMERATE_ (Cell, icell, cdm.allCells()) {
      if (dummy_var[icell] < 0) {
        ARCANE_FATAL("Bad value for dummy_var id={0} v={1}", ItemPrinter(*icell), dummy_var[icell]);
      }
      Real3 c1 = m_cell_center[icell];
      Real3 c2 = computed_center[icell];
      bool is_nearly_equal = math::isNearlyEqual(c1.x, c2.x) && math::isNearlyEqual(c1.y, c2.y) && math::isNearlyEqual(c1.z, c2.z);
      if (!is_nearly_equal)
        ARCANE_FATAL("Bad value for computed center id={0} center={1} computed={2}",
                     ItemPrinter(*icell), c1, c2);
    }
  }
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshTestUtils::
_testDirCellFace()
{
  IMesh* mesh = m_mesh;
  //VariableNodeReal3& nodes_coord = mesh->nodesCoordinates();
  Integer nb_dir = mesh->dimension();
  Integer nb_print = m_nb_print;
  for (Integer idir = 0; idir < nb_dir; ++idir) {
    CellDirectionMng cdm(m_cartesian_mesh->cellDirection(idir));
    eMeshDirection md = cdm.direction();
    info() << "DIRECTION=" << idir;
    Integer iprint = 0;
    ENUMERATE_CELL (icell, cdm.allCells()) {
      Cell cell = *icell;
      CellLocalId cell_id(cell.localId());
      DirCellFace cf(cdm.cellFace(cell));
      DirCellFace cf2(cdm.cellFace(cell_id));
      if (cf.cellId() != cf2.cellId())
        ARCANE_FATAL("Bad DirCellFace");
      bool is_print = (nb_print < 0 || iprint < nb_print);
      ++iprint;
      if (is_print) {
        info() << "CellFace uid=" << ItemPrinter(cell) << " dir=" << md;
        info() << "CellFace nextFace =" << ItemPrinter(cf.next()) << " xyz=" << m_face_center[cf.next()];
        info() << "CellFace prevFace=" << ItemPrinter(cf.previous()) << " xyz=" << m_face_center[cf.previous()];
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshTestUtils::
_testDirCellFaceAccelerator()
{
#if defined(ARCANE_HAS_ACCELERATOR_API)
  IMesh* mesh = m_mesh;
  //VariableNodeReal3& nodes_coord = mesh->nodesCoordinates();
  Integer nb_dir = mesh->dimension();

  UnstructuredMeshConnectivityView m_connectivity_view;
  m_connectivity_view.setMesh(mesh);
  auto cfc = m_connectivity_view.cellFace();

  VariableCellInt32 dummy_var(VariableBuildInfo(mesh, "DummyCellVariable"));

  for (Integer idir = 0; idir < nb_dir; ++idir) {
    CellDirectionMng cdm(m_cartesian_mesh->cellDirection(idir));

    auto queue = m_accelerator_mng->defaultQueue();
    auto command = makeCommand(*queue);

    auto inout_dummy_var = viewInOut(command, dummy_var);

    info() << "DIRECTION=" << idir;

    command << RUNCOMMAND_ENUMERATE(Cell, cell, cdm.allCells())
    {
      inout_dummy_var[cell] = 0;

      CellLocalId cell_id(cell.localId());
      DirCellFaceLocalId cf(cdm.dirCellFaceId(cell));
      DirCellFaceLocalId cf2(cdm.dirCellFaceId(cell_id));
      if (cf.cellId() != cf2.cellId()) {
        inout_dummy_var[cell] = -10;
        return;
      }

      Int32 next_index = cf.nextLocalIndex();
      Int32 previous_index = cf.previousLocalIndex();

      FaceLocalId next_face1 = cf.nextId();
      FaceLocalId next_face2 = cfc.faceId(cell, next_index);
      if (next_face1 != next_face2)
        inout_dummy_var[cell] = -11;

      FaceLocalId previous_face1 = cf.previousId();
      FaceLocalId previous_face2 = cfc.faceId(cell, previous_index);
      if (previous_face1 != previous_face2)
        inout_dummy_var[cell] = -12;
    };
    ENUMERATE_ (Cell, icell, cdm.allCells()) {
      if (dummy_var[icell] < 0) {
        ARCANE_FATAL("Bad value for dummy_var id={0} v={1}", ItemPrinter(*icell), dummy_var[icell]);
      }
    }
  }
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshTestUtils::
_testNodeToCellConnectivity3D()
{
  info() << "Test NodeToCell Connectivity3D";
  IMesh* mesh = m_mesh;
  VariableNodeReal3& nodes_coord = mesh->nodesCoordinates();
  CartesianConnectivity cc = m_cartesian_mesh->connectivity();
  ENUMERATE_NODE (inode, m_mesh->allNodes()) {
    Node node = *inode;
    Real3 node_coord = nodes_coord[inode];
    info(4) << "node_uid=" << node.uniqueId()
            << " UL=" << cc.upperLeftId(inode)
            << " UR=" << cc.upperRightId(inode)
            << " LR=" << cc.lowerRightId(inode)
            << " LL=" << cc.lowerLeftId(inode)
            << " TUL=" << cc.topZUpperLeftId(inode)
            << " TUR=" << cc.topZUpperRightId(inode)
            << " TLR=" << cc.topZLowerRightId(inode)
            << " TLL=" << cc.topZLowerLeftId(inode);
    {
      Cell upper_left = cc.upperLeft(node);
      if (!upper_left.null()) {
        Real3 c = m_cell_center[upper_left];
        if (c.y <= node_coord.y || c.x >= node_coord.x || c.z >= node_coord.z)
          ARCANE_FATAL("Bad upperLeft cell for node={0} c={1} n={2}", ItemPrinter(node), c, node_coord);
      }
    }
    {
      Cell upper_right = cc.upperRight(node);
      if (!upper_right.null()) {
        Real3 c = m_cell_center[upper_right];
        if (c.y <= node_coord.y || c.x <= node_coord.x || c.z >= node_coord.z)
          ARCANE_FATAL("Bad upperRight cell for node={0} c={1} n={2}", ItemPrinter(node), c, node_coord);
      }
    }
    {
      Cell lower_right = cc.lowerRight(node);
      if (!lower_right.null()) {
        Real3 c = m_cell_center[lower_right];
        if (c.y >= node_coord.y || c.x <= node_coord.x || c.z >= node_coord.z)
          ARCANE_FATAL("Bad lowerRight cell for node={0} c={1} n={2}", ItemPrinter(node), c, node_coord);
      }
    }
    {
      Cell lower_left = cc.lowerLeft(node);
      if (!lower_left.null()) {
        Real3 c = m_cell_center[lower_left];
        if (c.y >= node_coord.y || c.x >= node_coord.x || c.z >= node_coord.z)
          ARCANE_FATAL("Bad lowerLeft cell for node={0} c={1} n={2}", ItemPrinter(node), c, node_coord);
      }
    }
    {
      Cell top_upper_left = cc.topZUpperLeft(node);
      if (!top_upper_left.null()) {
        Real3 c = m_cell_center[top_upper_left];
        if (c.y <= node_coord.y || c.x >= node_coord.x || c.z <= node_coord.z)
          ARCANE_FATAL("Bad topZUpperLeft cell for node={0} c={1} n={2}", ItemPrinter(node), c, node_coord);
      }
    }
    {
      Cell top_upper_right = cc.topZUpperRight(node);
      if (!top_upper_right.null()) {
        Real3 c = m_cell_center[top_upper_right];
        if (c.y <= node_coord.y || c.x <= node_coord.x || c.z <= node_coord.z)
          ARCANE_FATAL("Bad topZUpperRight cell for node={0} c={1} n={2}", ItemPrinter(node), c, node_coord);
      }
    }
    {
      Cell top_lower_right = cc.topZLowerRight(node);
      if (!top_lower_right.null()) {
        Real3 c = m_cell_center[top_lower_right];
        if (c.y >= node_coord.y || c.x <= node_coord.x || c.z <= node_coord.z)
          ARCANE_FATAL("Bad topZLowerRight cell for node={0} c={1} n={2}", ItemPrinter(node), c, node_coord);
      }
    }
    {
      Cell top_lower_left = cc.topZLowerLeft(node);
      if (!top_lower_left.null()) {
        Real3 c = m_cell_center[top_lower_left];
        if (c.y >= node_coord.y || c.x >= node_coord.x || c.z <= node_coord.z)
          ARCANE_FATAL("Bad topZLowerLeft cell for node={0} c={1} n={2}", ItemPrinter(node), c, node_coord);
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshTestUtils::
_testNodeToCellConnectivity3DAccelerator()
{
#if defined(ARCANE_HAS_ACCELERATOR_API)
  info() << "Test NodeToCell Connectivity3D";
  IMesh* mesh = m_mesh;
  VariableNodeReal3& nodes_coord = mesh->nodesCoordinates();
  VariableNodeInt32 dummy_var(VariableBuildInfo(mesh, "DummyNodeVariable"));
  CartesianConnectivityLocalId cc = m_cartesian_mesh->connectivity();

  auto queue = m_accelerator_mng->defaultQueue();
  auto command = makeCommand(*queue);

  auto in_cell_center = viewIn(command, m_cell_center);
  auto in_node_coord = viewIn(command, nodes_coord);
  auto inout_dummy_var = viewInOut(command, dummy_var);
  command << RUNCOMMAND_ENUMERATE(Node, node, m_mesh->allNodes())
  {
    Real3 node_coord = in_node_coord[node];
    {
      CellLocalId upper_left = cc.upperLeftId(node);
      if (!upper_left.isNull()) {
        Real3 c = in_cell_center[upper_left];
        if (c.y <= node_coord.y || c.x >= node_coord.x || c.z >= node_coord.z)
          inout_dummy_var[node] = -10;
      }
    }
    {
      CellLocalId upper_right = cc.upperRightId(node);
      if (!upper_right.isNull()) {
        Real3 c = in_cell_center[upper_right];
        if (c.y <= node_coord.y || c.x <= node_coord.x || c.z >= node_coord.z)
          inout_dummy_var[node] = -11;
      }
    }
    {
      CellLocalId lower_right = cc.lowerRightId(node);
      if (!lower_right.isNull()) {
        Real3 c = in_cell_center[lower_right];
        if (c.y >= node_coord.y || c.x <= node_coord.x || c.z >= node_coord.z)
          inout_dummy_var[node] = -12;
      }
    }
    {
      CellLocalId lower_left = cc.lowerLeftId(node);
      if (!lower_left.isNull()) {
        Real3 c = in_cell_center[lower_left];
        if (c.y >= node_coord.y || c.x >= node_coord.x || c.z >= node_coord.z)
          inout_dummy_var[node] = -13;
      }
    }
    {
      CellLocalId top_upper_left = cc.topZUpperLeftId(node);
      if (!top_upper_left.isNull()) {
        Real3 c = in_cell_center[top_upper_left];
        if (c.y <= node_coord.y || c.x >= node_coord.x || c.z <= node_coord.z)
          inout_dummy_var[node] = -14;
      }
    }
    {
      CellLocalId top_upper_right = cc.topZUpperRightId(node);
      if (!top_upper_right.isNull()) {
        Real3 c = in_cell_center[top_upper_right];
        if (c.y <= node_coord.y || c.x <= node_coord.x || c.z <= node_coord.z)
          inout_dummy_var[node] = -15;
      }
    }
    {
      CellLocalId top_lower_right = cc.topZLowerRightId(node);
      if (!top_lower_right.isNull()) {
        Real3 c = in_cell_center[top_lower_right];
        if (c.y >= node_coord.y || c.x <= node_coord.x || c.z <= node_coord.z)
          inout_dummy_var[node] = -16;
      }
    }
    {
      CellLocalId top_lower_left = cc.topZLowerLeftId(node);
      if (!top_lower_left.isNull()) {
        Real3 c = in_cell_center[top_lower_left];
        if (c.y >= node_coord.y || c.x >= node_coord.x || c.z <= node_coord.z)
          inout_dummy_var[node] = -17;
      }
    }
  };
  ENUMERATE_ (Node, inode, m_mesh->allNodes()) {
    if (dummy_var[inode] < 0) {
      ARCANE_FATAL("Bad value for dummy_var id={0} v={1}", ItemPrinter(*inode), dummy_var[inode]);
    }
  }
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshTestUtils::
_testNodeToCellConnectivity2D()
{
  info() << "Test NodeToCell Connectivity 2D";
  IMesh* mesh = m_mesh;
  VariableNodeReal3& nodes_coord = mesh->nodesCoordinates();
  CartesianConnectivity cc = m_cartesian_mesh->connectivity();
  ENUMERATE_NODE (inode, mesh->allNodes()) {
    Node node = *inode;
    Real3 node_coord = nodes_coord[inode];
    info(4) << "node_uid=" << node.uniqueId()
            << " UL=" << cc.upperLeftId(inode)
            << " UR=" << cc.upperRightId(inode)
            << " LR=" << cc.lowerRightId(inode)
            << " LL=" << cc.lowerLeftId(inode);
    {
      Cell upper_left = cc.upperLeft(node);
      if (!upper_left.null()) {
        Real3 c = m_cell_center[upper_left];
        if (c.y <= node_coord.y || c.x >= node_coord.x)
          ARCANE_FATAL("Bad upperLeft cell for node={0} c={1} n={2}", ItemPrinter(node), c, node_coord);
      }
    }
    {
      Cell upper_right = cc.upperRight(node);
      if (!upper_right.null()) {
        Real3 c = m_cell_center[upper_right];
        if (c.y <= node_coord.y || c.x <= node_coord.x)
          ARCANE_FATAL("Bad upperRight cell for node={0} c={1} n={2}", ItemPrinter(node), c, node_coord);
      }
    }
    {
      Cell lower_right = cc.lowerRight(node);
      if (!lower_right.null()) {
        Real3 c = m_cell_center[lower_right];
        if (c.y >= node_coord.y || c.x <= node_coord.x)
          ARCANE_FATAL("Bad lowerRight cell for node={0} c={1} n={2}", ItemPrinter(node), c, node_coord);
      }
    }
    {
      Cell lower_left = cc.lowerLeft(node);
      if (!lower_left.null()) {
        Real3 c = m_cell_center[lower_left];
        if (c.y >= node_coord.y || c.x >= node_coord.x)
          ARCANE_FATAL("Bad lowerLeft cell for node={0} c={1} n={2}", ItemPrinter(node), c, node_coord);
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshTestUtils::
_testCellToNodeConnectivity3D()
{
  info() << "Test CellToNode Connectivity3D";
  IMesh* mesh = m_mesh;
  VariableNodeReal3& nodes_coord = mesh->nodesCoordinates();
  CartesianConnectivity cc = m_cartesian_mesh->connectivity();
  const bool is_not_amr = !m_is_amr;
  ENUMERATE_CELL (icell, m_mesh->allCells()) {
    Cell cell = *icell;
    Real3 cell_coord = m_cell_center[icell];
    info(4) << "cell_uid=" << cell.uniqueId()
            << " UL=" << cc.upperLeftId(icell)
            << " UR=" << cc.upperRightId(icell)
            << " LR=" << cc.lowerRightId(icell)
            << " LL=" << cc.lowerLeftId(icell)
            << " TUL=" << cc.topZUpperLeftId(icell)
            << " TUR=" << cc.topZUpperRightId(icell)
            << " TLR=" << cc.topZLowerRightId(icell)
            << " TLL=" << cc.topZLowerLeftId(icell);
    {
      Node upper_left = cc.upperLeft(cell);
      Cell ccell = cc.topZLowerRight(upper_left);
      Real3 n = nodes_coord[upper_left];
      if (n.y <= cell_coord.y || n.x >= cell_coord.x || n.z >= cell_coord.z)
        ARCANE_FATAL("Bad upperLeft node for cell={0}", ItemPrinter(cell));
      if (cell != ccell && is_not_amr)
        ARCANE_FATAL("Bad correspondance UL -> TLR cell={0} corresponding_cell={1}", ItemPrinter(cell), ccell);
    }
    {
      Node upper_right = cc.upperRight(cell);
      Cell ccell = cc.topZLowerLeft(upper_right);
      Real3 n = nodes_coord[upper_right];
      if (n.y <= cell_coord.y || n.x <= cell_coord.x || n.z >= cell_coord.z)
        ARCANE_FATAL("Bad upperRight node for cell={0}", ItemPrinter(cell));
      if (cell != ccell && is_not_amr)
        ARCANE_FATAL("Bad correspondance UR -> TLL cell={0} corresponding_cell={1}", ItemPrinter(cell), ccell);
    }
    {
      Node lower_right = cc.lowerRight(cell);
      Cell ccell = cc.topZUpperLeft(lower_right);
      Real3 n = nodes_coord[lower_right];
      if (n.y >= cell_coord.y || n.x <= cell_coord.x || n.z >= cell_coord.z)
        ARCANE_FATAL("Bad lowerRight node for cell={0}", ItemPrinter(cell));
      if (cell != ccell && is_not_amr)
        ARCANE_FATAL("Bad correspondance LR -> TUL cell={0} corresponding_cell={1}", ItemPrinter(cell), ccell);
    }
    {
      Node lower_left = cc.lowerLeft(cell);
      Cell ccell = cc.topZUpperRight(lower_left);
      Real3 n = nodes_coord[lower_left];
      if (n.y >= cell_coord.y || n.x >= cell_coord.x || n.z >= cell_coord.z)
        ARCANE_FATAL("Bad lowerLeft node for cell={0}", ItemPrinter(cell));
      if (cell != ccell && is_not_amr)
        ARCANE_FATAL("Bad correspondance LL -> TUR cell={0} corresponding_cell={1}", ItemPrinter(cell), ccell);
    }
    {
      Node top_upper_left = cc.topZUpperLeft(cell);
      Cell ccell = cc.lowerRight(top_upper_left);
      Real3 n = nodes_coord[top_upper_left];
      if (n.y <= cell_coord.y || n.x >= cell_coord.x || n.z <= cell_coord.z)
        ARCANE_FATAL("Bad topZUpperLeft node for cell={0}", ItemPrinter(cell));
      if (cell != ccell && is_not_amr)
        ARCANE_FATAL("Bad correspondance TUL -> LR cell={0} corresponding_cell={1}", ItemPrinter(cell), ccell);
    }
    {
      Node top_upper_right = cc.topZUpperRight(cell);
      Cell ccell = cc.lowerLeft(top_upper_right);
      Real3 n = nodes_coord[top_upper_right];
      if (n.y <= cell_coord.y || n.x <= cell_coord.x || n.z <= cell_coord.z)
        ARCANE_FATAL("Bad topZUpperRight node for cell={0}", ItemPrinter(cell));
      if (cell != ccell && is_not_amr)
        ARCANE_FATAL("Bad correspondance TUR -> LL cell={0} corresponding_cell={1}", ItemPrinter(cell), ccell);
    }
    {
      Node top_lower_right = cc.topZLowerRight(cell);
      Cell ccell = cc.upperLeft(top_lower_right);
      Real3 n = nodes_coord[top_lower_right];
      if (n.y >= cell_coord.y || n.x <= cell_coord.x || n.z <= cell_coord.z)
        ARCANE_FATAL("Bad topZLowerRight node for cell={0}", ItemPrinter(cell));
      if (cell != ccell && is_not_amr)
        ARCANE_FATAL("Bad correspondance TLR -> UL cell={0} corresponding_cell={1}", ItemPrinter(cell), ccell);
    }
    {
      Node top_lower_left = cc.topZLowerLeft(cell);
      Cell ccell = cc.upperRight(top_lower_left);
      Real3 n = nodes_coord[top_lower_left];
      if (n.y >= cell_coord.y || n.x >= cell_coord.x || n.z <= cell_coord.z)
        ARCANE_FATAL("Bad topZLowerLeft node for cell={0}", ItemPrinter(cell));
      if (cell != ccell && is_not_amr)
        ARCANE_FATAL("Bad correspondance TLL -> UR cell={0} corresponding_cell={1}", ItemPrinter(cell), ccell);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshTestUtils::
_testCellToNodeConnectivity3DAccelerator()
{
#if defined(ARCANE_HAS_ACCELERATOR_API)
  info() << "Test CellToNode Connectivity3D Accelerator";
  IMesh* mesh = m_mesh;
  VariableCellInt32 dummy_var(VariableBuildInfo(mesh, "DummyCellVariable"));
  VariableNodeReal3& nodes_coord = mesh->nodesCoordinates();
  CartesianConnectivityLocalId cc = m_cartesian_mesh->connectivity();
  const bool is_not_amr = !m_is_amr;

  auto queue = m_accelerator_mng->defaultQueue();
  auto command = makeCommand(*queue);

  auto in_cell_center = viewIn(command, m_cell_center);
  auto in_node_coord = viewIn(command, nodes_coord);
  auto inout_dummy_var = viewInOut(command, dummy_var);

  command << RUNCOMMAND_ENUMERATE(Cell, cell, m_mesh->allCells())
  {
    Real3 cell_coord = in_cell_center[cell];
    inout_dummy_var[cell] = 0;
    {
      NodeLocalId upper_left = cc.upperLeftId(cell);
      CellLocalId ccell = cc.topZLowerRightId(upper_left);
      Real3 n = in_node_coord[upper_left];
      if (n.y <= cell_coord.y || n.x >= cell_coord.x || n.z >= cell_coord.z)
        inout_dummy_var[cell] = -10;
      if (cell != ccell && is_not_amr)
        inout_dummy_var[cell] = -20;
    }
    {
      NodeLocalId upper_right = cc.upperRightId(cell);
      CellLocalId ccell = cc.topZLowerLeftId(upper_right);
      Real3 n = in_node_coord[upper_right];
      if (n.y <= cell_coord.y || n.x <= cell_coord.x || n.z >= cell_coord.z)
        inout_dummy_var[cell] = -11;
      if (cell != ccell && is_not_amr)
        inout_dummy_var[cell] = -21;
    }
    {
      NodeLocalId lower_right = cc.lowerRightId(cell);
      CellLocalId ccell = cc.topZUpperLeftId(lower_right);
      Real3 n = in_node_coord[lower_right];
      if (n.y >= cell_coord.y || n.x <= cell_coord.x || n.z >= cell_coord.z)
        inout_dummy_var[cell] = -12;
      if (cell != ccell && is_not_amr)
        inout_dummy_var[cell] = -22;
    }
    {
      NodeLocalId lower_left = cc.lowerLeftId(cell);
      CellLocalId ccell = cc.topZUpperRightId(lower_left);
      Real3 n = in_node_coord[lower_left];
      if (n.y >= cell_coord.y || n.x >= cell_coord.x || n.z >= cell_coord.z)
        inout_dummy_var[cell] = -13;
      if (cell != ccell && is_not_amr)
        inout_dummy_var[cell] = -23;
    }
    {
      NodeLocalId top_upper_left = cc.topZUpperLeftId(cell);
      CellLocalId ccell = cc.lowerRightId(top_upper_left);
      Real3 n = in_node_coord[top_upper_left];
      if (n.y <= cell_coord.y || n.x >= cell_coord.x || n.z <= cell_coord.z)
        inout_dummy_var[cell] = -14;
      if (cell != ccell && is_not_amr)
        inout_dummy_var[cell] = -24;
    }
    {
      NodeLocalId top_upper_right = cc.topZUpperRightId(cell);
      CellLocalId ccell = cc.lowerLeftId(top_upper_right);
      Real3 n = in_node_coord[top_upper_right];
      if (n.y <= cell_coord.y || n.x <= cell_coord.x || n.z <= cell_coord.z)
        inout_dummy_var[cell] = -15;
      if (cell != ccell && is_not_amr)
        inout_dummy_var[cell] = -25;
    }
    {
      NodeLocalId top_lower_right = cc.topZLowerRightId(cell);
      CellLocalId ccell = cc.upperLeftId(top_lower_right);
      Real3 n = in_node_coord[top_lower_right];
      if (n.y >= cell_coord.y || n.x <= cell_coord.x || n.z <= cell_coord.z)
        inout_dummy_var[cell] = -16;
      if (cell != ccell && is_not_amr)
        inout_dummy_var[cell] = -26;
    }
    {
      NodeLocalId top_lower_left = cc.topZLowerLeftId(cell);
      CellLocalId ccell = cc.upperRightId(top_lower_left);
      Real3 n = in_node_coord[top_lower_left];
      if (n.y >= cell_coord.y || n.x >= cell_coord.x || n.z <= cell_coord.z)
        inout_dummy_var[cell] = -17;
      if (cell != ccell && is_not_amr)
        inout_dummy_var[cell] = -27;
    }
  };
  ENUMERATE_ (Cell, icell, m_mesh->allCells()) {
    if (dummy_var[icell] < 0) {
      ARCANE_FATAL("Bad value for dummy_var id={0} v={1}", ItemPrinter(*icell), dummy_var[icell]);
    }
  }
#endif
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshTestUtils::
_testCellToNodeConnectivity2D()
{
  info() << "Test CellToNode Connectivity 2D";
  IMesh* mesh = m_mesh;
  VariableNodeReal3& nodes_coord = mesh->nodesCoordinates();
  CartesianConnectivity cc = m_cartesian_mesh->connectivity();
  const bool is_not_amr = !m_is_amr;
  ENUMERATE_CELL (icell, mesh->allCells()) {
    Cell cell = *icell;
    Real3 cell_coord = m_cell_center[icell];
    info(4) << "cell_uid=" << cell.uniqueId()
            << " UL=" << cc.upperLeftId(icell)
            << " UR=" << cc.upperRightId(icell)
            << " LR=" << cc.lowerRightId(icell)
            << " LL=" << cc.lowerLeftId(icell);
    {
      Node upper_left = cc.upperLeft(cell);
      Cell ccell = cc.lowerRight(upper_left);
      Real3 n = nodes_coord[upper_left];
      if (n.y <= cell_coord.y || n.x >= cell_coord.x)
        ARCANE_FATAL("Bad upperLeft node for cell={0} c={1} n={2}", ItemPrinter(cell), cell_coord, n);
      if (cell != ccell && is_not_amr)
        ARCANE_FATAL("Bad correspondance UL -> LR cell={0} corresponding_cell={1}", ItemPrinter(cell), ccell);
    }
    {
      Node upper_right = cc.upperRight(cell);
      Cell ccell = cc.lowerLeft(upper_right);
      Real3 n = nodes_coord[upper_right];
      if (n.y <= cell_coord.y || n.x <= cell_coord.x)
        ARCANE_FATAL("Bad upperRight node for cell={0} c={1} n={2}", ItemPrinter(cell), cell_coord, n);
      if (cell != ccell && is_not_amr)
        ARCANE_FATAL("Bad correspondance UR -> LF cell={0} corresponding_cell={1}", ItemPrinter(cell), ccell);
    }
    {
      Node lower_right = cc.lowerRight(cell);
      Cell ccell = cc.upperLeft(lower_right);
      Real3 n = nodes_coord[lower_right];
      if (n.y >= cell_coord.y || n.x <= cell_coord.x)
        ARCANE_FATAL("Bad lowerRight node for cell={0} c={1} n={2}", ItemPrinter(cell), cell_coord, n);
      if (cell != ccell && is_not_amr)
        ARCANE_FATAL("Bad correspondance UL -> LR cell={0} corresponding_cell={1}", ItemPrinter(cell), ccell);
    }
    {
      Node lower_left = cc.lowerLeft(cell);
      Cell ccell = cc.upperRight(lower_left);
      Real3 n = nodes_coord[lower_left];
      if (n.y >= cell_coord.y || n.x >= cell_coord.x)
        ARCANE_FATAL("Bad lowerLeft node for cell={0} c={1} n={2}", ItemPrinter(cell), cell_coord, n);
      if (cell != ccell && is_not_amr)
        ARCANE_FATAL("Bad correspondance LL -> UR cell={0} corresponding_cell={1}", ItemPrinter(cell), ccell);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename ItemType> void CartesianMeshTestUtils::
_testConnectivityByDirectionHelper(const ItemGroup& group)
{
  ValueChecker vc(A_FUNCINFO);
  Int32 nb_dim = m_mesh->dimension();

  info() << "Test ConnectivityByDirection 2D";
  CartesianConnectivity cc = m_cartesian_mesh->connectivity();
  ENUMERATE_ (ItemType, iitem, group) {
    vc.areEqual(cc.upperLeftId(iitem),cc.upperLeftId(iitem,0),"Item0 dir0");
    vc.areEqual(cc.upperRightId(iitem),cc.upperRightId(iitem,0),"Item1 dir0");
    vc.areEqual(cc.lowerRightId(iitem),cc.lowerRightId(iitem,0),"Item2 dir0");
    vc.areEqual(cc.lowerLeftId(iitem),cc.lowerLeftId(iitem,0),"Item3 dir0");

    if (nb_dim>1){
      vc.areEqual(cc.lowerLeftId(iitem),cc.upperLeftId(iitem,1),"Item0 dir1");
      vc.areEqual(cc.upperLeftId(iitem),cc.upperRightId(iitem,1),"Item1 dir1");
      vc.areEqual(cc.upperRightId(iitem),cc.lowerRightId(iitem,1),"Item2 dir1");
      vc.areEqual(cc.lowerRightId(iitem),cc.lowerLeftId(iitem,1),"Item3 dir1");
    }

    if (nb_dim>2){
      vc.areEqual(cc.upperRightId(iitem),cc.upperLeftId(iitem,2),"Item0 dir2");
      vc.areEqual(cc.topZUpperRightId(iitem),cc.upperRightId(iitem,2),"Item1 dir2");
      vc.areEqual(cc.topZLowerRightId(iitem),cc.lowerRightId(iitem,2),"Item2 dir2");
      vc.areEqual(cc.lowerRightId(iitem),cc.lowerLeftId(iitem,2),"Item3 dir2");

      vc.areEqual(cc.upperLeftId(iitem),cc.topZUpperLeftId(iitem,2),"Item4 dir2");
      vc.areEqual(cc.topZUpperLeftId(iitem),cc.topZUpperRightId(iitem,2),"Item5 dir2");
      vc.areEqual(cc.topZLowerLeftId(iitem),cc.topZLowerRightId(iitem,2),"Item6 dir2");
      vc.areEqual(cc.lowerLeftId(iitem),cc.topZLowerLeftId(iitem,2),"Item7 dir2");
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshTestUtils::
_testConnectivityByDirection()
{
  info() << "Test Node ConnectivityByDirection";
  _testConnectivityByDirectionHelper<Node>(m_mesh->allNodes());
  info() << "Test Cell ConnectivityByDirection";
  _testConnectivityByDirectionHelper<Cell>(m_mesh->allCells());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshTestUtils::
_sample(ICartesianMesh* cartesian_mesh)
{
  //! [SampleNodeToCell]
  CartesianConnectivity cc = cartesian_mesh->connectivity();
  ENUMERATE_NODE (inode, m_mesh->allNodes()) {
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

void CartesianMeshTestUtils::
_saveSVG()
{
  ICartesianMesh* cm = m_cartesian_mesh;
  IMesh* mesh = cm->mesh();
  info() << "Saving mesh to SVG format";
  std::ofstream ofile("toto.svg");
  SimpleSVGMeshExporter writer(ofile);
  writer.write(mesh->allCells());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace ArcaneTest

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
