// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CartesianMeshTestUtils.cc                                   (C) 2000-2021 */
/*                                                                           */
/* Fonctions utilitaires pour les tests de 'CartesianMesh'.                  */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/tests/CartesianMeshTestUtils.h"

#include "arcane/utils/CheckedConvert.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/Real2.h"

#include "arcane/MeshUtils.h"

#include "arcane/IMesh.h"
#include "arcane/IItemFamily.h"
#include "arcane/ItemPrinter.h"
#include "arcane/IParallelMng.h"

#include "arcane/IMesh.h"
#include "arcane/IItemFamily.h"
#include "arcane/IMeshModifier.h"
#include "arcane/IMeshUtilities.h"
#include "arcane/SimpleSVGMeshExporter.h"

#include "arcane/cea/ICartesianMesh.h"
#include "arcane/cea/CellDirectionMng.h"
#include "arcane/cea/FaceDirectionMng.h"
#include "arcane/cea/NodeDirectionMng.h"
#include "arcane/cea/CartesianConnectivity.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace ArcaneTest
{

using namespace Arcane;


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CartesianMeshTestUtils::
CartesianMeshTestUtils(ICartesianMesh* cm)
: TraceAccessor(cm->traceMng())
, m_cartesian_mesh(cm)
, m_mesh(cm->mesh())
, m_cell_center(VariableBuildInfo(m_mesh,"CellCenter"))
, m_face_center(VariableBuildInfo(m_mesh,"FaceCenter"))
, m_node_density(VariableBuildInfo(m_mesh,"NodeDensity"))
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
testAll()
{
  _testDirCell();
  _testDirFace();
  _testDirNode();
  _testDirCellNode();
  _testDirCellFace();
  if (m_mesh->dimension()==3){
    _testNodeToCellConnectivity3D();
    _testCellToNodeConnectivity3D();
  }
  else{
    _testNodeToCellConnectivity2D();
    _testCellToNodeConnectivity2D();
    _saveSVG();
  }
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
    ENUMERATE_CELL(icell,m_mesh->allCells()){
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
    ENUMERATE_FACE(iface,m_mesh->allFaces()){
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

void CartesianMeshTestUtils::
_checkSameId(Face item,FaceLocalId local_id)
{
  if (item.localId()!=local_id)
    ARCANE_FATAL("Bad FaceLocalId item={0} local_id={1}",item.localId(),local_id);
}

void CartesianMeshTestUtils::
_checkSameId(Cell item,CellLocalId local_id)
{
  if (item.localId()!=local_id)
    ARCANE_FATAL("Bad CellLocalId item={0} local_id={1}",item.localId(),local_id);
}

void CartesianMeshTestUtils::
_checkSameId(Node item,NodeLocalId local_id)
{
  if (item.localId()!=local_id)
    ARCANE_FATAL("Bad NodeLocalId item={0} local_id={1}",item.localId(),local_id);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshTestUtils::
_checkItemGroupIsSorted(const ItemGroup& group)
{
  if (!group.checkIsSorted())
    ARCANE_FATAL("Node direction group '{0}' is not sorted",group.name());
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
  for( Integer idir=0; idir<nb_dir; ++idir){
    CellDirectionMng cdm(m_cartesian_mesh->cellDirection(idir));
    cdm2 = m_cartesian_mesh->cellDirection(idir);
    cdm3 = cdm;
    eMeshDirection md = cdm.direction();
    Integer iprint = 0;
    info() << "DIRECTION=" << idir << " Cells=" << cdm.allCells().name();
    _checkItemGroupIsSorted(cdm.allCells());
    ENUMERATE_CELL(icell,cdm.allCells()){
      Cell cell = *icell;
      DirCell dir_cell(cdm[icell]);
      Cell prev_cell = dir_cell.previous();
      Cell next_cell = dir_cell.next();
      if (prev_cell.null() && next_cell.null())
        ARCANE_FATAL("Null previous and next cell");
      DirCell dir_cell2(cdm2[icell]);
      Cell prev_cell2 = dir_cell2.previous();
      Cell next_cell2 = dir_cell2.next();
      DirCell dir_cell3(cdm3[icell]);
      Cell prev_cell3 = dir_cell3.previous();
      Cell next_cell3 = dir_cell3.next();
      _checkSameId(prev_cell,prev_cell2);
      _checkSameId(next_cell,next_cell2);
      _checkSameId(prev_cell,prev_cell3);
      _checkSameId(next_cell,next_cell3);
      if (nb_print<0 || iprint<nb_print){
        ++iprint;
        if (!prev_cell.null() && !next_cell.null()){
          info() << "Cell uid=" << ItemPrinter(cell) << " dir=" << md
                 << " prev=" << ItemPrinter(prev_cell) << " xyz=" << m_cell_center[prev_cell]
                 << " next=" << ItemPrinter(next_cell) << " xyz=" << m_cell_center[next_cell];
        }
        else{
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
_testDirFace()
{
  IMesh* mesh = m_mesh;
  Integer nb_dir = mesh->dimension();
  for( Integer idir=0; idir<nb_dir; ++idir)
    _testDirFace(idir);
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
  ENUMERATE_FACE(iface,fdm.allFaces()){
    Face face = *iface;
    DirFace dir_face(fdm[iface]);
    DirFace dir_face2(fdm2[iface]);
    //Integer nb_node = cell.nbNode();
    Cell prev_cell = dir_face.previousCell();
    Cell next_cell = dir_face.nextCell();
    Cell prev_cell2 = dir_face2.previousCell();
    Cell next_cell2 = dir_face2.nextCell();
    _checkSameId(prev_cell,prev_cell2);
    _checkSameId(next_cell,next_cell2);
    _checkSameId(prev_cell,dir_face.previousCellId());
    _checkSameId(next_cell,dir_face.nextCellId());
    bool is_print = (nb_print<0 || iprint<nb_print);
    ++iprint;
    Real face_coord = m_face_center[iface][idir];
    if (!prev_cell.null() && !next_cell.null()){
      Real next_coord = m_cell_center[next_cell][idir];
      Real prev_coord = m_cell_center[prev_cell][idir];
      if (next_coord<prev_coord){
        info() << "Bad ordering for face";
        is_print = true;
        ++nb_error;
      }
      if (is_print)
        info() << "Face uid=" << ItemPrinter(face) << " dir=" << md << " xyz=" << m_face_center[iface]
               << " prev=" << prev_cell.uniqueId() << " xyz=" << m_cell_center[prev_cell]
               << " next=" << next_cell.uniqueId() << " xyz=" << m_cell_center[next_cell];
    }
    else{
      if (!prev_cell.null()){
        Real prev_coord = m_cell_center[prev_cell][idir];
        if (face_coord<prev_coord){
          info() << "Bad ordering for face";
          is_print = true;
          ++nb_error;
        }
        if (is_print)
          info() << "Face uid=" << ItemPrinter(face) << " dir=" << md << " xyz=" << m_face_center[iface]
                 << " prev=" << prev_cell.uniqueId() << " xyz=" << m_cell_center[prev_cell];
      }
      if (!next_cell.null()){
        Real next_coord = m_cell_center[next_cell][idir];
        if (next_coord<face_coord){
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
  if (nb_error!=0)
    ARCANE_FATAL("Bad connectivity for DirFace nb_error={0}",nb_error);
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
  for( Integer idir=0; idir<nb_dir; ++idir){
    NodeDirectionMng node_dm(m_cartesian_mesh->nodeDirection(idir));
    node_dm2 = m_cartesian_mesh->nodeDirection(idir);
    node_dm3 = node_dm;
    eMeshDirection md = node_dm.direction();
    Integer iprint = 0;
    info() << "DIRECTION=" << idir;
    NodeGroup dm_all_nodes = node_dm.allNodes();
    _checkItemGroupIsSorted(dm_all_nodes);
    ENUMERATE_NODE(inode,dm_all_nodes){
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
      _checkSameId(prev_node,prev_node2);
      _checkSameId(next_node,next_node2);
      _checkSameId(prev_node,prev_node3);
      _checkSameId(next_node,next_node3);
      _checkSameId(prev_node,dir_node.previousId());
      _checkSameId(next_node,dir_node.nextId());
      Real my_coord = nodes_coord[inode][idir];
      bool is_print = (nb_print<0 || iprint<nb_print);
      ++iprint;
      if (is_print){
        Int32 node_nb_cell = node.nbCell();
        info() << "DirNode node= " << ItemPrinter(node) << " nb_cell=" << node_nb_cell << " pos=" << nodes_coord[node];
        for( Integer k=0; k<node_nb_cell; ++k ){
          Real3 cell_pos = m_cell_center[node.cell(k)];
          info() << "Node k=" << k << " cell_pos=" << cell_pos << " cell=" << ItemPrinter(node.cell(k));
        }
        for( Integer k=0; k<8; ++k ){
          Int32 cell_index = dir_node.cellIndex(k);
          Real3 cell_pos;
          if (cell_index!=(-1)){
            if ((1+cell_index)>node_nb_cell)
              ARCANE_FATAL("Bad value for cell_index '{0}' node_nb_cell={1}",cell_index,node_nb_cell);
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
        ARCANE_FATAL("Null previous and next node for node {0}",ItemPrinter(node));
      //TODO: Vérifier que les coordonnées autres que celle de idir sont bien les mêmes pour next,prev et my_coord.
      if (!prev_node.null() && !next_node.null()){
        Real next_coord = nodes_coord[next_node][idir];
        Real prev_coord = nodes_coord[prev_node][idir];
        if (next_coord<prev_coord){
          info() << "Bad ordering for node";
          is_print = true;
          ++nb_error;
        }
        if (is_print)
          info() << "Node uid=" << ItemPrinter(node) << " dir=" << md
                 << " prev=" << ItemPrinter(prev_node) << " xyz=" << prev_coord
                 << " next=" << ItemPrinter(next_node) << " xyz=" << next_coord;
      }
      else{
        if (!prev_node.null()){
          Real prev_coord = nodes_coord[prev_node][idir];
          if (my_coord<prev_coord){
            info() << "Bad ordering for node";
            is_print = true;
            ++nb_error;
          }
          if (is_print)
            info() << "Node uid=" << ItemPrinter(node) << " dir=" << md
                   << " prev=" << ItemPrinter(prev_node) << " xyz=" << nodes_coord[prev_node];
        }
        if (!next_node.null()){
          Real next_coord = nodes_coord[next_node][idir];
          if (next_coord<my_coord){
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

  if (nb_error!=0)
    ARCANE_FATAL("Bad connectivity for DirNode");

  // Vérifie qu'on a parcouru tous les noeuds
  ENUMERATE_NODE(inode,m_mesh->allNodes()){
    Node node = *inode;
    if (node.cell(0).level()!=0)
      continue;
    if (m_node_density[inode]==0.0){
      ++nb_error;
      info() << "Bad value for node " << ItemPrinter(node);
    }
  }
  if (nb_error!=0)
    ARCANE_FATAL("Bad values for DirNode");
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
  for( Integer idir=0; idir<nb_dir; ++idir){
    CellDirectionMng cdm(m_cartesian_mesh->cellDirection(idir));
    eMeshDirection md = cdm.direction();
    info() << "DIRECTION=" << idir;
    Integer iprint = 0;
    ENUMERATE_CELL(icell,cdm.allCells()){
      Cell cell = *icell;
      CellLocalId cell_id(cell.localId());
      DirCellNode cn(cdm.cellNode(cell));
      DirCellNode cn2(cdm.cellNode(cell_id));
      if (cn.cellId()!=cn2.cellId())
        ARCANE_FATAL("Bad DirCellNode");
      bool is_print = (nb_print<0 || iprint<nb_print);
      ++iprint;
      if (is_print){
        info() << "Cell uid=" << ItemPrinter(cell) << " dir=" << md;
        info() << "Cell nextLeft =" << ItemPrinter(cn.nextLeft()) << " xyz=" << nodes_coord[cn.nextLeft()];
        info() << "Cell nextRight=" << ItemPrinter(cn.nextRight()) << " xyz=" << nodes_coord[cn.nextRight()];
        info() << "Cell prevRight=" << ItemPrinter(cn.previousRight()) << " xyz=" << nodes_coord[cn.previousRight()];
        info() << "Cell prevLeft =" << ItemPrinter(cn.previousLeft()) << " xyz=" << nodes_coord[cn.previousLeft()];
        if (nb_dir==3){
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
_testDirCellFace()
{
  IMesh* mesh = m_mesh;
  //VariableNodeReal3& nodes_coord = mesh->nodesCoordinates();
  Integer nb_dir = mesh->dimension();
  Integer nb_print = m_nb_print;
  for( Integer idir=0; idir<nb_dir; ++idir){
    CellDirectionMng cdm(m_cartesian_mesh->cellDirection(idir));
    eMeshDirection md = cdm.direction();
    info() << "DIRECTION=" << idir;
    Integer iprint = 0;
    ENUMERATE_CELL(icell,cdm.allCells()){
      Cell cell = *icell;
      CellLocalId cell_id(cell.localId());
      DirCellFace cf(cdm.cellFace(cell));
      DirCellFace cf2(cdm.cellFace(cell_id));
      if (cf.cellId()!=cf2.cellId())
        ARCANE_FATAL("Bad DirCellFace");
      bool is_print = (nb_print<0 || iprint<nb_print);
      ++iprint;
      if (is_print){
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
_testNodeToCellConnectivity3D()
{
  info() << "Test NodeToCell Connectivity3D";
  IMesh* mesh = m_mesh;
  VariableNodeReal3& nodes_coord = mesh->nodesCoordinates();
  CartesianConnectivity cc = m_cartesian_mesh->connectivity();
  ENUMERATE_NODE(inode,m_mesh->allNodes()){
    Node node = *inode;
    Real3 node_coord = nodes_coord[inode];
    info() << "node_uid=" << node.uniqueId()
           << " UL=" << cc.upperLeft(node).localId()
           << " UR=" << cc.upperRight(node).localId()
           << " LR=" << cc.lowerRight(node).localId()
           << " LL=" << cc.lowerLeft(node).localId()
           << " TUL=" << cc.topZUpperLeft(node).localId()
           << " TUR=" << cc.topZUpperRight(node).localId()
           << " TLR=" << cc.topZLowerRight(node).localId()
           << " TLL=" << cc.topZLowerLeft(node).localId();
    {
      Cell upper_left = cc.upperLeft(node);
      if (!upper_left.null()){
        Real3 c = m_cell_center[upper_left];
        if (c.y<=node_coord.y || c.x>=node_coord.x || c.z>=node_coord.z)
          ARCANE_FATAL("Bad upperLeft cell for node={0} c={1} n={2}",ItemPrinter(node),c,node_coord);
      }
    }
    {
      Cell upper_right = cc.upperRight(node);
      if (!upper_right.null()){
        Real3 c = m_cell_center[upper_right];
        if (c.y<=node_coord.y || c.x<=node_coord.x || c.z>=node_coord.z)
          ARCANE_FATAL("Bad upperRight cell for node={0} c={1} n={2}",ItemPrinter(node),c,node_coord);
      }
    }
    {
      Cell lower_right = cc.lowerRight(node);
      if (!lower_right.null()){
        Real3 c = m_cell_center[lower_right];
        if (c.y>=node_coord.y || c.x<=node_coord.x || c.z>=node_coord.z)
          ARCANE_FATAL("Bad lowerRight cell for node={0} c={1} n={2}",ItemPrinter(node),c,node_coord);
      }
    }
    {
      Cell lower_left = cc.lowerLeft(node);
      if (!lower_left.null()){
        Real3 c = m_cell_center[lower_left];
        if (c.y>=node_coord.y || c.x>=node_coord.x || c.z>=node_coord.z)
          ARCANE_FATAL("Bad lowerLeft cell for node={0} c={1} n={2}",ItemPrinter(node),c,node_coord);
      }
    }
    {
      Cell top_upper_left = cc.topZUpperLeft(node);
      if (!top_upper_left.null()){
        Real3 c = m_cell_center[top_upper_left];
        if (c.y<=node_coord.y || c.x>=node_coord.x || c.z<=node_coord.z)
          ARCANE_FATAL("Bad topZUpperLeft cell for node={0} c={1} n={2}",ItemPrinter(node),c,node_coord);
      }
    }
    {
      Cell top_upper_right = cc.topZUpperRight(node);
      if (!top_upper_right.null()){
        Real3 c = m_cell_center[top_upper_right];
        if (c.y<=node_coord.y || c.x<=node_coord.x || c.z<=node_coord.z)
          ARCANE_FATAL("Bad topZUpperRight cell for node={0} c={1} n={2}",ItemPrinter(node),c,node_coord);
      }
    }
    {
      Cell top_lower_right = cc.topZLowerRight(node);
      if (!top_lower_right.null()){
        Real3 c = m_cell_center[top_lower_right];
        if (c.y>=node_coord.y || c.x<=node_coord.x || c.z<=node_coord.z)
          ARCANE_FATAL("Bad topZLowerRight cell for node={0} c={1} n={2}",ItemPrinter(node),c,node_coord);
      }
    }
    {
      Cell top_lower_left = cc.topZLowerLeft(node);
      if (!top_lower_left.null()){
        Real3 c = m_cell_center[top_lower_left];
        if (c.y>=node_coord.y || c.x>=node_coord.x || c.z<=node_coord.z)
          ARCANE_FATAL("Bad topZLowerLeft cell for node={0} c={1} n={2}",ItemPrinter(node),c,node_coord);
      }
    }
  }
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
  ENUMERATE_NODE(inode,mesh->allNodes()){
    Node node = *inode;
    Real3 node_coord = nodes_coord[inode];
    {
      Cell upper_left = cc.upperLeft(node);
      if (!upper_left.null()){
        Real3 c = m_cell_center[upper_left];
        if (c.y<=node_coord.y || c.x>=node_coord.x)
          ARCANE_FATAL("Bad upperLeft cell for node={0} c={1} n={2}",ItemPrinter(node),c,node_coord);
      }
    }
    {
      Cell upper_right = cc.upperRight(node);
      if (!upper_right.null()){
        Real3 c = m_cell_center[upper_right];
        if (c.y<=node_coord.y || c.x<=node_coord.x)
          ARCANE_FATAL("Bad upperRight cell for node={0} c={1} n={2}",ItemPrinter(node),c,node_coord);
      }
    }
    {
      Cell lower_right = cc.lowerRight(node);
      if (!lower_right.null()){
        Real3 c = m_cell_center[lower_right];
        if (c.y>=node_coord.y || c.x<=node_coord.x)
          ARCANE_FATAL("Bad lowerRight cell for node={0} c={1} n={2}",ItemPrinter(node),c,node_coord);
      }
    }
    {
      Cell lower_left = cc.lowerLeft(node);
      if (!lower_left.null()){
        Real3 c = m_cell_center[lower_left];
        if (c.y>=node_coord.y || c.x>=node_coord.x)
          ARCANE_FATAL("Bad lowerLeft cell for node={0} c={1} n={2}",ItemPrinter(node),c,node_coord);
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
  ENUMERATE_CELL(icell,m_mesh->allCells()){
    Cell cell = *icell;
    Real3 cell_coord = m_cell_center[icell];
    {
      Node upper_left = cc.upperLeft(cell);
      Cell ccell = cc.topZLowerRight(upper_left);
      Real3 n = nodes_coord[upper_left];
      if (n.y<=cell_coord.y || n.x>=cell_coord.x || n.z>=cell_coord.z)
        ARCANE_FATAL("Bad upperLeft node for cell={0}",ItemPrinter(cell));
      if (cell!=ccell)
        ARCANE_FATAL("Bad correspondance UL -> TLR cell={0} corresponding_cell={1}",ItemPrinter(cell),ccell);
    }
    {
      Node upper_right = cc.upperRight(cell);
      Cell ccell = cc.topZLowerLeft(upper_right);
      Real3 n = nodes_coord[upper_right];
      if (n.y<=cell_coord.y || n.x<=cell_coord.x || n.z>=cell_coord.z)
        ARCANE_FATAL("Bad upperRight node for cell={0}",ItemPrinter(cell));
      if (cell!=ccell)
        ARCANE_FATAL("Bad correspondance UR -> TLL cell={0} corresponding_cell={1}",ItemPrinter(cell),ccell);
    }
    {
      Node lower_right = cc.lowerRight(cell);
      Cell ccell = cc.topZUpperLeft(lower_right);
      Real3 n = nodes_coord[lower_right];
      if (n.y>=cell_coord.y || n.x<=cell_coord.x || n.z>=cell_coord.z)
        ARCANE_FATAL("Bad lowerRight node for cell={0}",ItemPrinter(cell));
      if (cell!=ccell)
        ARCANE_FATAL("Bad correspondance LR -> TUL cell={0} corresponding_cell={1}",ItemPrinter(cell),ccell);
    }
    {
      Node lower_left = cc.lowerLeft(cell);
      Cell ccell = cc.topZUpperRight(lower_left);
      Real3 n = nodes_coord[lower_left];
      if (n.y>=cell_coord.y || n.x>=cell_coord.x || n.z>=cell_coord.z)
        ARCANE_FATAL("Bad lowerLeft node for cell={0}",ItemPrinter(cell));
      if (cell!=ccell)
        ARCANE_FATAL("Bad correspondance LL -> TUR cell={0} corresponding_cell={1}",ItemPrinter(cell),ccell);
    }
    {
      Node top_upper_left = cc.topZUpperLeft(cell);
      Cell ccell = cc.lowerRight(top_upper_left);
      Real3 n = nodes_coord[top_upper_left];
      if (n.y<=cell_coord.y || n.x>=cell_coord.x || n.z<=cell_coord.z)
        ARCANE_FATAL("Bad topZUpperLeft node for cell={0}",ItemPrinter(cell));
      if (cell!=ccell)
        ARCANE_FATAL("Bad correspondance TUL -> LR cell={0} corresponding_cell={1}",ItemPrinter(cell),ccell);
    }
    {
      Node top_upper_right = cc.topZUpperRight(cell);
      Cell ccell = cc.lowerLeft(top_upper_right);
      Real3 n = nodes_coord[top_upper_right];
      if (n.y<=cell_coord.y || n.x<=cell_coord.x || n.z<=cell_coord.z)
        ARCANE_FATAL("Bad topZUpperRight node for cell={0}",ItemPrinter(cell));
      if (cell!=ccell)
        ARCANE_FATAL("Bad correspondance TUR -> LL cell={0} corresponding_cell={1}",ItemPrinter(cell),ccell);
    }
    {
      Node top_lower_right = cc.topZLowerRight(cell);
      Cell ccell = cc.upperLeft(top_lower_right);
      Real3 n = nodes_coord[top_lower_right];
      if (n.y>=cell_coord.y || n.x<=cell_coord.x || n.z<=cell_coord.z)
        ARCANE_FATAL("Bad topZLowerRight node for cell={0}",ItemPrinter(cell));
      if (cell!=ccell)
        ARCANE_FATAL("Bad correspondance TLR -> UL cell={0} corresponding_cell={1}",ItemPrinter(cell),ccell);
    }
    {
      Node top_lower_left = cc.topZLowerLeft(cell);
      Cell ccell = cc.upperRight(top_lower_left);
      Real3 n = nodes_coord[top_lower_left];
      if (n.y>=cell_coord.y || n.x>=cell_coord.x || n.z<=cell_coord.z)
        ARCANE_FATAL("Bad topZLowerLeft node for cell={0}",ItemPrinter(cell));
      if (cell!=ccell)
        ARCANE_FATAL("Bad correspondance TLL -> UR cell={0} corresponding_cell={1}",ItemPrinter(cell),ccell);
    }
  }
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
  ENUMERATE_CELL(icell,mesh->allCells()){
    Cell cell = *icell;
    Real3 cell_coord = m_cell_center[icell];
    {
      Node upper_left = cc.upperLeft(cell);
      Real3 n = nodes_coord[upper_left];
      if (n.y<=cell_coord.y || n.x>=cell_coord.x)
        ARCANE_FATAL("Bad upperLeft node for cell={0} c={1} n={2}",ItemPrinter(cell),cell_coord,n);
    }
    {
      Node upper_right = cc.upperRight(cell);
      Real3 n = nodes_coord[upper_right];
      if (n.y<=cell_coord.y || n.x<=cell_coord.x)
        ARCANE_FATAL("Bad upperRight node for cell={0} c={1} n={2}",ItemPrinter(cell),cell_coord,n);
    }
    {
      Node lower_right = cc.lowerRight(cell);
      Real3 n = nodes_coord[lower_right];
      if (n.y>=cell_coord.y || n.x<=cell_coord.x)
        ARCANE_FATAL("Bad lowerRight node for cell={0} c={1} n={2}",ItemPrinter(cell),cell_coord,n);
    }
    {
      Node lower_left = cc.lowerLeft(cell);
      Real3 n = nodes_coord[lower_left];
      if (n.y>=cell_coord.y || n.x>=cell_coord.x)
        ARCANE_FATAL("Bad lowerLeft node for cell={0} c={1} n={2}",ItemPrinter(cell),cell_coord,n);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CartesianMeshTestUtils::
_sample(ICartesianMesh* cartesian_mesh)
{
  //! [SampleNodeToCell]
  CartesianConnectivity cc = cartesian_mesh->connectivity();
  ENUMERATE_NODE(inode,m_mesh->allNodes()){
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
  ofstream ofile("toto.svg");
  SimpleSVGMeshExporter writer(ofile);
  writer.write(mesh->allCells());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace ArcaneTest

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
