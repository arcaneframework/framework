// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ICartesianMeshNumberingMngInternal.h                        (C) 2000-2026 */
/*                                                                           */
/* Interface for Cartesian mesh numbering manager.                           */
/* In these managers, it is assumed that there is an interval of unique IDs  */
/* assigned to each mesh level.                                              */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifndef ARCANE_CARTESIANMESH_ICARTESIANMESHNUMBERINGMNGINTERNAL_H
#define ARCANE_CARTESIANMESH_ICARTESIANMESHNUMBERINGMNGINTERNAL_H

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/cartesianmesh/CartesianMeshGlobal.h"
#include "arcane/core/Item.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Interface for Cartesian mesh numbering manager.
 *
 * In these managers, it is assumed that there is an interval of unique IDs
 * assigned to each mesh level.
 *
 * \warning The mesh must not be renumbered if an implementation of
 * this interface is used (or, it should no longer be used after
 * renumbering (be careful with protections/resumptions)).
 */
class ARCANE_CARTESIANMESH_EXPORT ICartesianMeshNumberingMngInternal
{
 public:

  virtual ~ICartesianMeshNumberingMngInternal() = default;

 public:

  virtual void build() = 0;
  virtual void saveInfosInProperties() = 0;
  virtual void recreateFromDump() = 0;

  /*!
   * \brief Method allowing the renumbering of level 0 faces.
   *
   * This allows avoiding unique ID conversions during
   * certain operations on the faces and thus freeing the structures
   * that map between the two numbering schemes.
   *
   * Since these structures are not shared between subdomains, renumbering
   * allows partitioning the mesh without problems.
   */
  virtual void renumberingFacesLevel0FromOriginalArcaneNumbering() = 0;

  /*!
   * \brief Method allowing the description of the object's state.
   */
  virtual void printStatus() = 0;

  /*!
   * \brief Method allowing the preparation of a new level.
   *
   * Before refining or coarsening cells, it is
   * necessary to call this method to prepare the object
   * to provide information concerning the new level.
   *
   * It should also be noted that this new level must be the level
   * directly superior to the highest existing level or
   * directly inferior to the lowest existing level.
   *
   * \param level The new level to prepare.
   */
  virtual void prepareLevel(Int32 level) = 0;

  /*!
   * \brief Method allowing the update of the first level.
   *
   * Indeed, during coarsening, the new level is level -1. Since Arcane does not like negative levels,
   * we must update the information so that there are none.
   */
  virtual void updateFirstLevel() = 0;

  /*!
   * \brief Method allowing the retrieval of the first unique ID used by the cells of a level.
   * Calling this method with level and level+1 allows retrieving the unique ID interval
   * of a level.
   *
   * \param level The level.
   * \return The first cell UID of the level.
   */
  virtual Int64 firstCellUniqueId(Int32 level) const = 0;

  /*!
   * \brief Method allowing the retrieval of the first unique ID used by the nodes of a level.
   * Calling this method with level and level+1 allows retrieving the unique ID interval
   * of a level.
   *
   * \param level The level.
   * \return The first node UID of the level.
   */
  virtual Int64 firstNodeUniqueId(Int32 level) const = 0;

  /*!
   * \brief Method allowing the retrieval of the first unique ID used by the faces of a level.
   * Calling this method with level and level+1 allows retrieving the unique ID interval
   * of a level.
   *
   * \param level The level.
   * \return The first face UID of the level.
   */
  virtual Int64 firstFaceUniqueId(Int32 level) const = 0;

  /*!
   * \brief Method allowing the retrieval of the global number of cells in X for a level.
   *
   * \param level The level.
   * \return The number of cells in X.
   */
  virtual CartCoord globalNbCellsX(Int32 level) const = 0;

  /*!
   * \brief Method allowing the retrieval of the global number of cells in Y for a level.
   *
   * \param level The level.
   * \return The number of cells in Y.
   */
  virtual CartCoord globalNbCellsY(Int32 level) const = 0;

  /*!
   * \brief Method allowing the retrieval of the global number of cells in Z for a level.
   *
   * \param level The level.
   * \return The number of cells in Z.
   */
  virtual CartCoord globalNbCellsZ(Int32 level) const = 0;

  /*!
   * \brief Method allowing the retrieval of the global number of nodes in X for a level.
   *
   * \param level The level.
   * \return The number of nodes in X.
   */
  virtual CartCoord globalNbNodesX(Int32 level) const = 0;

  /*!
   * \brief Method allowing the retrieval of the global number of nodes in Y for a level.
   *
   * \param level The level.
   * \return The number of nodes in Y.
   */
  virtual CartCoord globalNbNodesY(Int32 level) const = 0;

  /*!
   * \brief Method allowing the retrieval of the global number of nodes in Z for a level.
   *
   * \param level The level.
   * \return The number of nodes in Z.
   */
  virtual CartCoord globalNbNodesZ(Int32 level) const = 0;

  /*!
   * \brief Method allowing the retrieval of the global number of faces in X for a level.
   *
   * Suppose we have the following faces:
   *  ┌─0──┬──2─┐
   * 4│   6│   8│
   *  ├─5──┼─7──┤
   * 9│  11│  13│
   *  └─10─┴─12─┘
   *
   * So, we have 2x2 cells.
   * In X, we have 3 faces.
   *
   * For the number of faces in Cartesian view, see \a globalNbFacesXCartesianView.
   *
   * \param level The level.
   * \return The number of faces in X.
   */
  virtual CartCoord globalNbFacesX(Int32 level) const = 0;

  /*!
   * \brief Method allowing the retrieval of the global number of faces in Y for a level.
   *
   * Suppose we have the following faces:
   *  ┌─0──┬──2─┐
   * 4│   6│   8│
   *  ├─5──┼─7──┤
   * 9│  11│  13│
   *  └─10─┴─12─┘
   *
   * So, we have 2x2 cells.
   * In Y, we have 3 faces.
   *
   * For the number of faces in Cartesian view, see \a globalNbFacesYCartesianView.
   *
   * \param level The level.
   * \return The number of faces in Y.
   */
  virtual CartCoord globalNbFacesY(Int32 level) const = 0;

  /*!
   * \brief Method allowing the retrieval of the global number of faces in Z for a level.
   *
   * Suppose we have the following faces:
   *  ┌─0──┬──2─┐
   * 4│   6│   8│
   *  ├─5──┼─7──┤
   * 9│  11│  13│
   *  └─10─┴─12─┘
   *
   * If we have 2x2x2 cells, we will have 3 faces in Z.
   *
   * For the number of faces in Cartesian view, see \a globalNbFacesZCartesianView.
   *
   * \param level The level.
   * \return The number of faces in Z.
   */
  virtual CartCoord globalNbFacesZ(Int32 level) const = 0;

  /*!
   * \brief Method allowing the retrieval of the size of the "Cartesian grid" view
   *        containing the faces.
   *
   * In 2D, this view can be: (for a 2x2 cell mesh):
   *     x =  0  1  2  3  4
   *        ┌──┬──┬──┬──┬──┐
   * y = -1 │ 0│  │ 2│  │ 4│
   *        ┌──┬──┬──┬──┬──┐
   * y = 0  │  │ 1│  │ 3│  │
   *        ├──┼──┼──┼──┼──┤
   * y = 1  │ 5│  │ 7│  │ 9│
   *        ├──┼──┼──┼──┼──┤
   * y = 2  │  │ 6│  │ 8│  │
   *        ├──┼──┼──┼──┼──┤
   * y = 3  │10│  │12│  │14│
   *        ├──┼──┼──┼──┼──┤
   * y = 4  │  │11│  │13│  │
   *        └──┴──┴──┴──┴──┘
   * (in this view, the cells are located at odd X and Y
   * (so here, [1, 1], [3, 1], [1, 3] and [3, 3])).
   *
   * \note In 2D, it is considered that there is an imaginary level y=-1.
   * \warning In order to start the numbering at 0, in the methods
   * returning a 2D face unique ID, we use FaceUID-1.
   *
   * And in 3D (for a 2x2x2 cell mesh):
   *         z = 0            │ z = 1            │ z = 2            │ z = 3            │ z = 4
   *      x =  0  1  2  3  4  │   0  1  2  3  4  │   0  1  2  3  4  │   0  1  2  3  4  │   0  1  2  3  4
   *         ┌──┬──┬──┬──┬──┐ │ ┌──┬──┬──┬──┬──┐ │ ┌──┬──┬──┬──┬──┐ │ ┌──┬──┬──┬──┬──┐ │ ┌──┬──┬──┬──┬──┐
   *  y = 0  │  │  │  │  │  │ │ │  │12│  │13│  │ │ │  │  │  │  │  │ │ │  │18│  │19│  │ │ │  │  │  │  │  │
   *         ├──┼──┼──┼──┼──┤ │ ├──┼──┼──┼──┼──┤ │ ├──┼──┼──┼──┼──┤ │ ├──┼──┼──┼──┼──┤ │ ├──┼──┼──┼──┼──┤
   *  y = 1  │  │24│  │25│  │ │ │ 0│  │ 1│  │ 2│ │ │  │28│  │29│  │ │ │ 6│  │ 7│  │ 8│ │ │  │32│  │33│  │
   *         ├──┼──┼──┼──┼──┤ │ ├──┼──┼──┼──┼──┤ │ ├──┼──┼──┼──┼──┤ │ ├──┼──┼──┼──┼──┤ │ ├──┼──┼──┼──┼──┤
   *  y = 2  │  │  │  │  │  │ │ │  │14│  │15│  │ │ │  │  │  │  │  │ │ │  │20│  │21│  │ │ │  │  │  │  │  │
   *         ├──┼──┼──┼──┼──┤ │ ├──┼──┼──┼──┼──┤ │ ├──┼──┼──┼──┼──┤ │ ├──┼──┼──┼──┼──┤ │ ├──┼──┼──┼──┼──┤
   *  y = 3  │  │26│  │27│  │ │ │ 3│  │ 4│  │ 5│ │ │  │30│  │31│  │ │ │ 9│  │10│  │11│ │ │  │34│  │35│  │
   *         ├──┼──┼──┼──┼──┤ │ ├──┼──┼──┼──┼──┤ │ ├──┼──┼──┼──┼──┤ │ ├──┼──┼──┼──┼──┤ │ ├──┼──┼──┼──┼──┤
   *  y = 4  │  │  │  │  │  │ │ │  │16│  │17│  │ │ │  │  │  │  │  │ │ │  │22│  │23│  │ │ │  │  │  │  │  │
   *         └──┴──┴──┴──┴──┘ │ └──┴──┴──┴──┴──┘ │ └──┴──┴──┴──┴──┘ │ └──┴──┴──┴──┴──┘ │ └──┴──┴──┴──┴──┘
   *                          │                  │                  │                  │
   *
   * (in this view, the cells are located at odd X, Y, and Z
   * (so here, [1, 1, 1], [3, 1, 1], [1, 3, 1], etc.)).
   *
   * \param level The level.
   * \return The size of the grid in X.
   */
  virtual CartCoord globalNbFacesXCartesianView(Int32 level) const = 0;

  /*!
   * \brief Method allowing the retrieval of the size of the "Cartesian grid" view
   *        containing the faces.
   *
   * An example of this view is available in the documentation of \a globalNbFacesXCartesianView.
   *
   * \param level The level.
   * \return The size of the grid in Y.
   */
  virtual CartCoord globalNbFacesYCartesianView(Int32 level) const = 0;

  /*!
   * \brief Method allowing the retrieval of the size of the "Cartesian grid" view
   *        containing the faces.
   *
   * An example of this view is available in the documentation of \a globalNbFacesXCartesianView.
   *
   * \param level The level.
   * \return The size of the grid in Z.
   */
  virtual CartCoord globalNbFacesZCartesianView(Int32 level) const = 0;

  /*!
   * \brief Method allowing the retrieval of the total number of cells in a level.
   *
   * \param level The level.
   * \return The number of cells in the level.
   */
  virtual Int64 nbCellInLevel(Int32 level) const = 0;

  /*!
   * \brief Method allowing the retrieval of the total number of nodes in a level.
   *
   * \param level The level.
   * \return The number of nodes in the level.
   */
  virtual Int64 nbNodeInLevel(Int32 level) const = 0;

  /*!
   * \brief Method allowing the retrieval of the total number of faces in a level.
   *
   * \param level The level.
   * \return The number of faces in the level.
   */
  virtual Int64 nbFaceInLevel(Int32 level) const = 0;

  /*!
   * \brief Method allowing the retrieval of the refinement pattern used in each cell.
   * For example, if the pattern is 2, each parent cell will have 2*2 child cells (2*2*2 in 3D).
   *
   * \return The refinement pattern.
   */
  virtual Int32 pattern() const = 0;

  /*!
   * \brief Method allowing the retrieval of the level of a cell with its UID.
   *
   * \param uid The unique ID of the cell.
   * \return The level of the cell.
   */
  virtual Int32 cellLevel(Int64 uid) const = 0;

  /*!
   * \brief Method allowing the retrieval of the level of a node with its UID.
   *
   * \param uid The unique ID of the node.
   * \return The level of the node.
   */
  virtual Int32 nodeLevel(Int64 uid) const = 0;

  /*!
   * \brief Method allowing the retrieval of the level of a face with its UID.
   *
   * \param uid The unique ID of the face.
   * \return The level of the face.
   */
  virtual Int32 faceLevel(Int64 uid) const = 0;

  /*!
   * \brief Method allowing the retrieval of the position of the first child node/cell from the position
   * of the parent node/cell.
   *
   * Example: if we have a 2D mesh of 2*2 cells and a refinement pattern of 2,
   * we know that the level 1 grid (for level 1 patches) will be 4*4 cells.
   * The first child node/cell of the parent node/cell (Xp=1,Yp=0) will have the position Xf=Xp*Pattern=2 (same for Y).
   *
   * \param coord The X, Y, or Z position of the parent node/cell.
   * \param level_from The parent level.
   * \param level_to The child level.
   * \return The position of the first child of the parent node/cell.
   */
  virtual CartCoord offsetLevelToLevel(CartCoord coord, Int32 level_from, Int32 level_to) const = 0;

  /*!
   * \brief Method to obtain the position of the first child node/cell from the position
   * of the parent node/cell.
   *
   * Example: if we have a 2D mesh of 2*2 cells and a refinement pattern of 2,
   * we know that the level 1 grid (for level 1 patches) will be 4*4 cells.
   * The first child node/cell of the parent node/cell (Xp=1,Yp=0) will have the position Xf=Xp*Pattern=2 (same for Y).
   *
   * \param coord The position of the parent node/cell.
   * \param level_from The parent level.
   * \param level_to The child level.
   * \return The position of the first child of the parent node/cell.
   */
  virtual CartCoord3 offsetLevelToLevel(CartCoord3 coord, Int32 level_from, Int32 level_to) const = 0;

  /*!
   * \brief Method to obtain the position of the first child face from the position
   * of the parent face.
   *
   * Note, the coordinates used here are the coordinates of the faces in "Cartesian view"
   * (see \a globalNbFacesXCartesianView ).
   *
   * \param coord The X, Y, or Z position of the parent face.
   * \param level_from The parent level.
   * \param level_to The child level.
   * \return The position of the first child of the parent face.
   */
  virtual CartCoord faceOffsetLevelToLevel(CartCoord coord, Int32 level_from, Int32 level_to) const = 0;

  /*!
   * \brief Method to retrieve the coordinates of a cell using its uniqueId.
   *
   * \param uid The uniqueId of the cell.
   * \param level The level of the cell.
   * \return The position of the cell.
   */
  virtual CartCoord3 cellUniqueIdToCoord(Int64 uid, Int32 level) = 0;

  /*!
   * \brief Method to retrieve the coordinates of a cell.
   *
   * \param cell The cell.
   * \return The position of the cell.
   */
  virtual CartCoord3 cellUniqueIdToCoord(Cell cell) = 0;

  /*!
   * \brief Method to retrieve the X coordinate of a cell using its uniqueId.
   *
   * \param uid The uniqueId of the cell.
   * \param level The level of the cell.
   * \return The X position of the cell.
   */
  virtual CartCoord cellUniqueIdToCoordX(Int64 uid, Int32 level) = 0;

  /*!
   * \brief Method to retrieve the X coordinate of a cell.
   *
   * \param cell The cell.
   * \return The X position of the cell.
   */
  virtual CartCoord cellUniqueIdToCoordX(Cell cell) = 0;

  /*!
   * \brief Method to retrieve the Y coordinate of a cell using its uniqueId.
   *
   * \param uid The uniqueId of the cell.
   * \param level The level of the cell.
   * \return The Y position of the cell.
   */
  virtual CartCoord cellUniqueIdToCoordY(Int64 uid, Int32 level) = 0;

  /*!
   * \brief Method to retrieve the Y coordinate of a cell.
   *
   * \param cell The cell.
   * \return The Y position of the cell.
   */
  virtual CartCoord cellUniqueIdToCoordY(Cell cell) = 0;

  /*!
   * \brief Method to retrieve the Z coordinate of a cell using its uniqueId.
   *
   * \param uid The uniqueId of the cell.
   * \param level The level of the cell.
   * \return The Z position of the cell.
   */
  virtual CartCoord cellUniqueIdToCoordZ(Int64 uid, Int32 level) = 0;

  /*!
   * \brief Method to retrieve the Z coordinate of a cell.
   *
   * \param cell The cell.
   * \return The Z position of the cell.
   */
  virtual CartCoord cellUniqueIdToCoordZ(Cell cell) = 0;

  /*!
   * \brief Method to retrieve the X coordinate of a node using its uniqueId.
   *
   * \param uid The uniqueId of the node.
   * \param level The level of the node.
   * \return The X position of the node.
   */
  virtual CartCoord nodeUniqueIdToCoordX(Int64 uid, Int32 level) = 0;

  /*!
   * \brief Method to retrieve the X coordinate of a node.
   *
   * \param node The node.
   * \return The X position of the node.
   */
  virtual CartCoord nodeUniqueIdToCoordX(Node node) = 0;

  /*!
   * \brief Method to retrieve the Y coordinate of a node using its uniqueId.
   *
   * \param uid The uniqueId of the node.
   * \param level The level of the node.
   * \return The Y position of the node.
   */
  virtual CartCoord nodeUniqueIdToCoordY(Int64 uid, Int32 level) = 0;

  /*!
   * \brief Method to retrieve the Y coordinate of a node.
   *
   * \param node The node.
   * \return The Y position of the node.
   */
  virtual CartCoord nodeUniqueIdToCoordY(Node node) = 0;

  /*!
   * \brief Method to retrieve the Z coordinate of a node using its uniqueId.
   *
   * \param uid The uniqueId of the node.
   * \param level The level of the node.
   * \return The Z position of the node.
   */
  virtual CartCoord nodeUniqueIdToCoordZ(Int64 uid, Int32 level) = 0;

  /*!
   * \brief Method to retrieve the Z coordinate of a node.
   *
   * \param node The node.
   * \return The Z position of the node.
   */
  virtual CartCoord nodeUniqueIdToCoordZ(Node node) = 0;

  /*!
   * \brief Method to retrieve the X coordinate of a face using its uniqueId.
   *
   * Note, the coordinates used here are the coordinates of the faces in "Cartesian view"
   * (see \a globalNbFacesXCartesianView ).
   *
   * \param uid The uniqueId of the face.
   * \param level The level of the face.
   * \return The X position of the face.
   */
  virtual CartCoord faceUniqueIdToCoordX(Int64 uid, Int32 level) = 0;

  /*!
   * \brief Method to retrieve the X coordinate of a face.
   *
   * Note, the coordinates used here are the coordinates of the faces in "Cartesian view"
   * (see \a globalNbFacesXCartesianView ).
   *
   * \param face The face.
   * \return The X position of the face.
   */
  virtual CartCoord faceUniqueIdToCoordX(Face face) = 0;

  /*!
   * \brief Method to retrieve the Y coordinate of a face using its uniqueId.
   *
   * Note, the coordinates used here are the coordinates of the faces in "Cartesian view"
   * (see \a globalNbFacesXCartesianView ).
   *
   * \param uid The uniqueId of the face.
   * \param level The level of the face.
   * \return The Y position of the face.
   */
  virtual CartCoord faceUniqueIdToCoordY(Int64 uid, Int32 level) = 0;

  /*!
   * \brief Method to retrieve the Y coordinate of a face.
   *
   * Note, the coordinates used here are the coordinates of the faces in "Cartesian view"
   * (see \a globalNbFacesXCartesianView ).
   *
   * \param face The face.
   * \return The Y position of the face.
   */
  virtual CartCoord faceUniqueIdToCoordY(Face face) = 0;

  /*!
   * \brief Method to retrieve the Z coordinate of a face using its uniqueId.
   *
   * Note, the coordinates used here are the coordinates of the faces in "Cartesian view"
   * (see \a globalNbFacesXCartesianView ).
   *
   * \param uid The uniqueId of the face.
   * \param level The level of the face.
   * \return The Z position of the face.
   */
  virtual CartCoord faceUniqueIdToCoordZ(Int64 uid, Int32 level) = 0;

  /*!
   * \brief Method to retrieve the Z coordinate of a face.
   *
   * Note, the coordinates used here are the coordinates of the faces in "Cartesian view"
   * (see \a globalNbFacesXCartesianView ).
   *
   * \param face The face.
   * \return The Z position of the face.
   */
  virtual CartCoord faceUniqueIdToCoordZ(Face face) = 0;

  /*!
   * \brief Method to retrieve the uniqueId of a cell from its position and level.
   *
   * \param cell_coord The position of the cell.
   * \param level The level of the cell.
   * \return The uniqueId of the cell.
   */
  virtual Int64 cellUniqueId(CartCoord3 cell_coord, Int32 level) = 0;

  /*!
   * \brief Method to retrieve the uniqueId of a cell from its position and level.
   *
   * \param cell_coord The position of the cell.
   * \param level The level of the cell.
   * \return The uniqueId of the cell.
   */
  virtual Int64 cellUniqueId(CartCoord2 cell_coord, Int32 level) = 0;

  /*!
   * \brief Method to retrieve the uniqueId of a node from its position and level.
   *
   * \param level The level of the node.
   * \param cell_coord The position of the node.
   * \return The uniqueId of the node.
   */
  virtual Int64 nodeUniqueId(CartCoord3 node_coord, Int32 level) = 0;

  /*!
   * \brief Method to retrieve the uniqueId of a node from its position and level.
   *
   * \param level The level of the node.
   * \param cell_coord The position of the node.
   * \return The uniqueId of the node.
   */
  virtual Int64 nodeUniqueId(CartCoord2 node_coord, Int32 level) = 0;

  /*!
   * \brief Method to retrieve the uniqueId of a face from its position and level.
   *
   * Note, the coordinates used here are the coordinates of the faces in "Cartesian view"
   * (see \a globalNbFacesXCartesianView ).
   *
   * \param level The level of the face.
   * \param cell_coord The position of the face.
   * \return The uniqueId of the face.
   */
  virtual Int64 faceUniqueId(CartCoord3 face_coord, Int32 level) = 0;

  /*!
   * \brief Method to retrieve the uniqueId of a face from its position and level.
   *
   * Note, the coordinates used here are the coordinates of the faces in "Cartesian view"
   * (see \a globalNbFacesXCartesianView ).
   *
   * \param level The level of the face.
   * \param cell_coord The position of the face.
   * \return The uniqueId of the face.
   */
  virtual Int64 faceUniqueId(CartCoord2 face_coord, Int32 level) = 0;

  /*!
   * \brief Method to retrieve the number of nodes in a cell.
   *
   * \return The number of nodes in a cell.
   */
  virtual Int32 nbNodeByCell() = 0;

  /*!
   * \brief Method to retrieve the uniqueIds of the nodes of a cell from
   * its coordinates.
   *
   * The order in which the uniqueIds are placed corresponds to the enumeration order of the nodes
   * of an Arcane cell.
   *      3--2
   * ^y   |  |
   * |    0--1
   * ->x
   *
   * \param cell_coord The position of the cell.
   * \param level The level of the cell (and thus the nodes).
   * \param uid [OUT] The uniqueIds of the cell. The size of the ArrayView must be equal to nbNodeByCell().
   */
  virtual void cellNodeUniqueIds(CartCoord3 cell_coord, Int32 level, ArrayView<Int64> uid) = 0;

  /*!
   * \brief Method to retrieve the uniqueIds of the nodes of a cell from
   * its coordinates.
   *
   * The order in which the uniqueIds are placed corresponds to the enumeration order of the nodes
   * of an Arcane cell.
   *      3--2
   * ^y   |  |
   * |    0--1
   * ->x
   *
   * \param cell_coord The position of the cell.
   * \param level The level of the cell (and thus the nodes).
   * \param uid [OUT] The uniqueIds of the cell. The size of the ArrayView must be equal to nbNodeByCell().
   */
  virtual void cellNodeUniqueIds(CartCoord2 cell_coord, Int32 level, ArrayView<Int64> uid) = 0;

  /*!
   * \brief Method to retrieve the uniqueIds of the nodes of a cell from
   * its uniqueId.
   *
   * The order in which the uniqueIds are placed corresponds to the enumeration order of the nodes
   * of an Arcane cell.
   *      3--2
   * ^y   |  |
   * |    0--1
   * ->x
   *
   * \param cell_uid The uniqueId of the cell.
   * \param level The level of the cell (and thus the nodes).
   * \param uid [OUT] The uniqueIds of the cell. The size of the ArrayView must be equal to nbNodeByCell().
   */
  virtual void cellNodeUniqueIds(Int64 cell_uid, Int32 level, ArrayView<Int64> uid) = 0;

  /*!
   * \brief Method to retrieve the uniqueIds of the nodes of a cell.
   *
   * The order in which the uniqueIds are placed corresponds to the enumeration order of the nodes
   * of an Arcane cell.
   *      3--2
   * ^y   |  |
   * |    0--1
   * ->x
   *
   * \param cell The cell.
   * \param uid [OUT] The uniqueIds of the cell. The size of the ArrayView must be equal to nbNodeByCell().
   */
  virtual void cellNodeUniqueIds(Cell cell, ArrayView<Int64> uid) = 0;

  /*!
   * \brief Method to retrieve the number of faces in a cell.
   *
   * \return The number of faces in a cell.
   */
  virtual Int32 nbFaceByCell() = 0;

  /*!
   * \brief Method to retrieve the uniqueIds of a cell's faces based on its coordinates.
   *
   * The order in which the uniqueIds are placed corresponds to the enumeration order of the faces
   * of an Arcane cell.
   *      -2-
   * ^y   3 1
   * |    -0-
   * ->x
   *
   * \param cell_coord The position of the cell.
   * \param level The level of the cell (and thus the faces).
   * \param uid [OUT] The uniqueIds of the cell. The size of the ArrayView must be equal to nbFaceByCell().
   */
  virtual void cellFaceUniqueIds(CartCoord3 cell_coord, Int32 level, ArrayView<Int64> uid) = 0;

  /*!
   * \brief Method to retrieve the uniqueIds of a cell's faces based on its coordinates.
   *
   * The order in which the uniqueIds are placed corresponds to the enumeration order of the faces
   * of an Arcane cell.
   *      -2-
   * ^y   3 1
   * |    -0-
   * ->x
   *
   * \param cell_coord The position of the cell.
   * \param level The level of the cell (and thus the faces).
   * \param uid [OUT] The uniqueIds of the cell. The size of the ArrayView must be equal to nbFaceByCell().
   */
  virtual void cellFaceUniqueIds(CartCoord2 cell_coord, Int32 level, ArrayView<Int64> uid) = 0;

  /*!
   * \brief Method to retrieve the uniqueIds of a cell's faces based on its uniqueId.
   *
   * The order in which the uniqueIds are placed corresponds to the enumeration order of the faces
   * of an Arcane cell.
   *      -2-
   * ^y   3 1
   * |    -0-
   * ->x
   *
   * \param cell_uid The uniqueId of the cell.
   * \param level The level of the cell (and thus the faces).
   * \param uid [OUT] The uniqueIds of the cell. The size of the ArrayView must be equal to nbFaceByCell().
   */
  virtual void cellFaceUniqueIds(Int64 cell_uid, Int32 level, ArrayView<Int64> uid) = 0;

  /*!
   * \brief Method to retrieve the uniqueIds of a cell's faces.
   *
   * The order in which the uniqueIds are placed corresponds to the enumeration order of the faces
   * of an Arcane cell.
   *      -2-
   * ^y   3 1
   * |    -0-
   * ->x
   *
   * \param cell The cell.
   * \param uid [OUT] The uniqueIds of the cell. The size of the ArrayView must be equal to nbFaceByCell().
   */
  virtual void cellFaceUniqueIds(Cell cell, ArrayView<Int64> uid) = 0;

  /*!
   * \brief Method to retrieve the uniqueIds of cells around a cell.
   *
   * If there is no cell in an adjacent location (e.g., if we are at the edge of the mesh),
   * a uniqueId of -1 is used.
   *
   * The passed view must have a size of 27.
   *
   * \param cell_coord The position of the cell.
   * \param level The level of the cell at the center.
   * \param uid [OUT] The uniqueIds of the surrounding cells.
   */
  virtual void cellUniqueIdsAroundCell(CartCoord3 cell_coord, Int32 level, ArrayView<Int64> uid) = 0;

  /*!
   * \brief Method to retrieve the uniqueIds of cells around a cell.
   *
   * If there is no cell in an adjacent location (e.g., if we are at the edge of the mesh),
   * a uniqueId of -1 is used.
   *
   * The passed view must have a size of 9.
   *
   * \param cell_coord The position of the cell.
   * \param level The level of the cell at the center.
   * \param uid [OUT] The uniqueIds of the surrounding cells.
   */
  virtual void cellUniqueIdsAroundCell(CartCoord2 cell_coord, Int32 level, ArrayView<Int64> uid) = 0;

  /*!
   * \brief Method to retrieve the uniqueIds of cells around the cell passed as a parameter.
   *
   * If there is no cell in an adjacent location (e.g., if we are at the edge of the mesh),
   * a uniqueId of -1 is used.
   *
   * The passed view must have a size of 9 in 2D and 27 in 3D.
   *
   * \param cell_uid The uniqueId of the cell at the center.
   * \param level The level of the cell at the center.
   * \param uid [OUT] The uniqueIds of the surrounding cells.
   */
  virtual void cellUniqueIdsAroundCell(Int64 cell_uid, Int32 level, ArrayView<Int64> uid) = 0;

  /*!
   * \brief Method to retrieve the uniqueIds of cells around the cell passed as a parameter.
   *
   * If there is no cell in an adjacent location (e.g., if we are at the edge of the mesh),
   * a uniqueId of -1 is used.
   *
   * The passed view must have a size of 9 in 2D and 27 in 3D.
   *
   * \param cell The cell at the center.
   * \param uid [OUT] The uniqueIds of the surrounding cells.
   */
  virtual void cellUniqueIdsAroundCell(Cell cell, ArrayView<Int64> uid) = 0;

  /*!
   * \brief Method to retrieve the uniqueIds of cells around a node.
   *
   * If there is no cell in an adjacent location (e.g., if we are at the edge of the mesh),
   * a uniqueId of -1 is used.
   *
   * The passed view must have a size of 8.
   *
   * \param node_coord The position of the node.
   * \param level The level of the node.
   * \param uid [OUT] The uniqueIds of the surrounding cells.
   */
  virtual void cellUniqueIdsAroundNode(CartCoord3 node_coord, Int32 level, ArrayView<Int64> uid) = 0;

  /*!
   * \brief Method to retrieve the uniqueIds of cells around a node.
   *
   * If there is no cell in an adjacent location (e.g., if we are at the edge of the mesh),
   * a uniqueId of -1 is used.
   *
   * The passed view must have a size of 4.
   *
   * \param node_coord The position of the node.
   * \param level The level of the node.
   * \param uid [OUT] The uniqueIds of the surrounding cells.
   */
  virtual void cellUniqueIdsAroundNode(CartCoord2 node_coord, Int32 level, ArrayView<Int64> uid) = 0;

  /*!
   * \brief Method to retrieve the uniqueIds of cells around the node passed as a parameter.
   *
   * If there is no cell in an adjacent location (e.g., if we are at the edge of the mesh),
   * a uniqueId of -1 is used.
   *
   * The passed view must have a size of 4 in 2D or 8 in 3D.
   *
   * \param node_uid The uniqueId of the node.
   * \param level The level of the node.
   * \param uid [OUT] The uniqueIds of the surrounding cells.
   */
  virtual void cellUniqueIdsAroundNode(Int64 node_uid, Int32 level, ArrayView<Int64> uid) = 0;

  /*!
   * \brief Method to retrieve the uniqueIds of cells around the node passed as a parameter.
   *
   * If there is no cell in an adjacent location (e.g., if we are at the edge of the mesh),
   * a uniqueId of -1 is used.
   *
   * The passed view must have a size of 4 in 2D or 8 in 3D.
   *
   * \param node The node.
   * \param uid [OUT] The uniqueIds of the surrounding cells.
   */
  virtual void cellUniqueIdsAroundNode(Node node, ArrayView<Int64> uid) = 0;

  /*!
   * \brief Method to define the spatial coordinates of the nodes of child cells of a parent cell.
   * This method must be called after calling endUpdate().
   *
   * \param parent_cell The parent cell.
   */
  virtual void setChildNodeCoordinates(Cell parent_cell) = 0;

  /*!
   * \brief Method to define the spatial coordinates of the nodes of a parent cell.
   * This method must be called after calling endUpdate().
   *
   * \param parent_cell The parent cell.
   */
  virtual void setParentNodeCoordinates(Cell parent_cell) = 0;

  /*!
   * \brief Method to retrieve the uniqueId of a cell's parent.
   *
   * If \a do_fatal is true, a fatal error is generated if the parent does not exist; otherwise, the returned uniqueId has the value NULL_ITEM_UNIQUE_ID.
   *
   * \param uid The uniqueId of the child cell.
   * \param level The level of the child cell.
   * \return The uniqueId of the parent cell of the cell passed as a parameter.
   */
  virtual Int64 parentCellUniqueIdOfCell(Int64 uid, Int32 level, bool do_fatal = true) = 0;

  /*!
   * \brief Method to retrieve the uniqueId of a cell's parent.
   *
   * If \a do_fatal is true, a fatal error is generated if the parent does not exist; otherwise, the returned uniqueId has the value NULL_ITEM_UNIQUE_ID.
   *
   * \param cell The child cell.
   * \return The uniqueId of the parent cell of the cell passed as a parameter.
   */
  virtual Int64 parentCellUniqueIdOfCell(Cell cell, bool do_fatal = true) = 0;

  /*!
   * \brief Method to retrieve the uniqueId of a child cell of a parent cell based on the position of the child cell within the parent cell.
   *
   * \param cell The parent cell.
   * \param child_coord_in_parent The position of the child within the parent cell.
   * \return The uniqueId of the requested child cell.
   */
  virtual Int64 childCellUniqueIdOfCell(Cell cell, CartCoord3 child_coord_in_parent) = 0;

  /*!
   * \brief Method to retrieve the uniqueId of a child cell of a parent cell based on the position of the child cell within the parent cell.
   *
   * \param cell The parent cell.
   * \param child_coord_in_parent The position of the child within the parent cell.
   * \return The uniqueId of the requested child cell.
   */
  virtual Int64 childCellUniqueIdOfCell(Cell cell, CartCoord2 child_coord_in_parent) = 0;

  /*!
   * \brief Method to retrieve the uniqueId of a child cell of a parent cell based on the index of the child cell within the parent cell.
   *
   * \param cell The parent cell.
   * \param child_index_in_parent The index of the child within the parent cell.
   * \return The uniqueId of the requested child cell.
   */
  virtual Int64 childCellUniqueIdOfCell(Cell cell, Int32 child_index_in_parent) = 0;

  /*!
   * \brief Method to retrieve a child cell of a parent cell based on the position of the child cell within the parent cell.
   *
   * \param cell The parent cell.
   * \param child_coord_in_parent The position of the child within the parent cell.
   * \return The requested child cell.
   */
  virtual Cell childCellOfCell(Cell cell, CartCoord3 child_coord_in_parent) = 0;

  /*!
   * \brief Method to retrieve a child cell of a parent cell based on the position of the child cell within the parent cell.
   *
   * \param cell The parent cell.
   * \param child_coord_in_parent The position of the child within the parent cell.
   * \return The requested child cell.
   */
  virtual Cell childCellOfCell(Cell cell, CartCoord2 child_coord_in_parent) = 0;

  /*!
   * \brief Method to retrieve the uniqueId of a node's parent.
   *
   * If \a do_fatal is true, a fatal error is generated if the parent does not exist; otherwise, the returned uniqueId has the value NULL_ITEM_UNIQUE_ID.
   *
   * \param uid The uniqueId of the child node.
   * \param level The level of the child node.
   * \return The uniqueId of the parent node of the child node.
   */
  virtual Int64 parentNodeUniqueIdOfNode(Int64 uid, Int32 level, bool do_fatal = true) = 0;

  /*!
   * \brief Method to retrieve the uniqueId of a node's parent.
   *
   * If \a do_fatal is true, a fatal error is generated if the parent does not exist; otherwise, the returned uniqueId has the value NULL_ITEM_UNIQUE_ID.
   *
   * \param node The child node.
   * \return The uniqueId of the parent node of the node passed as a parameter.
   */
  virtual Int64 parentNodeUniqueIdOfNode(Node node, bool do_fatal = true) = 0;

  /*!
   * \brief Method to retrieve the uniqueId of a child node of a parent node.
   *
   * \param uid The uniqueId of the child node.
   * \param level The level of the child node.
   * \return The uniqueId of the requested child node.
   */
  virtual Int64 childNodeUniqueIdOfNode(Int64 uid, Int32 level) = 0;

  /*!
   * \brief Method to retrieve the uniqueId of a child node of a parent node.
   *
   * \param node The parent node.
   * \return The uniqueId of the requested child node.
   */
  virtual Int64 childNodeUniqueIdOfNode(Node node) = 0;

  /*!
   * \brief Method to retrieve the uniqueId of a face's parent.
   *
   * If \a do_fatal is true, a fatal error is generated if the parent does not exist; otherwise, the returned uniqueId has the value NULL_ITEM_UNIQUE_ID.
   *
   * \param uid The uniqueId of the child face.
   * \param level The level of the child face.
   * \return The uniqueId of the parent face of the child face.
   */
  virtual Int64 parentFaceUniqueIdOfFace(Int64 uid, Int32 level, bool do_fatal = true) = 0;

  /*!
   * \brief Method to retrieve the uniqueId of a face's parent.
   *
   * If \a do_fatal is true, a fatal error is generated if the parent does not exist; otherwise, the returned uniqueId has the value NULL_ITEM_UNIQUE_ID.
   *
   * \param face The child face.
   * \return The uniqueId of the parent face of the face passed as a parameter.
   */
  virtual Int64 parentFaceUniqueIdOfFace(Face face, bool do_fatal = true) = 0;

  /*!
   * \brief Method to retrieve the uniqueId of a child face of a parent face based on the index of the child face within the parent face.
   *
   * \param uid The uniqueId of the parent face.
   * \param level The level of the parent face.
   * \param child_index_in_parent The index of the child within the parent face.
   * \return The uniqueId of the requested child face.
   */
  virtual Int64 childFaceUniqueIdOfFace(Int64 uid, Int32 level, Int32 child_index_in_parent) = 0;

  /*!
   * \brief Method to retrieve the uniqueId of a child face of a parent face based on the index of the child face within the parent face.
   *
   * \param face The parent face.
   * \param child_index_in_parent The index of the child within the parent face.
   * \return The uniqueId of the requested child face.
   */
  virtual Int64 childFaceUniqueIdOfFace(Face face, Int32 child_index_in_parent) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
