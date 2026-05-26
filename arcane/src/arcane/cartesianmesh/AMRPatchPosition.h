// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AMRPatchPosition.h                                          (C) 2000-2026 */
/*                                                                           */
/* Position of an AMR patch in a Cartesian mesh.                             */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CARTESIANMESH_AMRPATCHPOSITION_H
#define ARCANE_CARTESIANMESH_AMRPATCHPOSITION_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/cartesianmesh/CartesianMeshGlobal.h"
#include "arcane/utils/Vector3.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup ArcaneCartesianMesh
 * \brief Class allowing the definition of a patch position in the Cartesian
 * mesh.
 *
 * The position of a patch is designated by the position of two cells in the
 * grid. The "min" position and the "max" position form an enclosing box.
 *
 * \warning The cell at the "min" position is included in the box, but the
 * cell at the "max" position is excluded.
 *
 * \note The patch position is global for the Cartesian mesh. The decomposition
 * into subdomains is not taken into account (example with the \a nbCells()
 * method of this class, which gives the number of cells in the patch without
 * considering subdomains).
 *
 * Cell positions can be obtained via the CartesianMeshNumberingMng.
 *
 * \warning This class is only valid for a refinement pattern of 2 (modifying
 * this should not be complex, if needed).
 */
class ARCANE_CARTESIANMESH_EXPORT AMRPatchPosition
{
 public:

  /*!
   * \brief Constructor for a null position.
   * A null position is defined by a level = -2.
   */
  AMRPatchPosition();
  AMRPatchPosition(Int32 level, CartCoord3 min_point, CartCoord3 max_point, Int32 overlap_layer_size);

  /*!
   * \brief Copy constructor.
   * \param src The position to copy.
   */
  AMRPatchPosition(const AMRPatchPosition& src);
  AMRPatchPosition& operator=(const AMRPatchPosition&) = default;

  ~AMRPatchPosition();

 public:

  bool operator==(const AMRPatchPosition& other) const = default;

 public:

  /*!
   * \brief Method to retrieve the patch level.
   * \return The patch level.
   */
  Int32 level() const;

  /*!
   * \brief Method to set the patch level.
   * \param level The patch level.
   */
  void setLevel(Int32 level);

  /*!
   * \brief Method to retrieve the min position of the enclosing box.
   *
   * \return The min position.
   */
  CartCoord3 minPoint() const;

  /*!
   * \brief Method to set the min position of the enclosing box.
   * \param min_point the min position.
   */
  void setMinPoint(CartCoord3 min_point);

  /*!
   * \brief Method to retrieve the max position of the enclosing box.
   *
   * \return The max position.
   */
  CartCoord3 maxPoint() const;

  /*!
   * \brief Method to set the max position of the enclosing box.
   * \param max_point the max position.
   */
  void setMaxPoint(CartCoord3 max_point);

  /*!
   * \brief Method to retrieve the number of overlap cell layers of the patch.
   *
   * \return the number of overlap cell layers
   */
  Int32 overlapLayerSize() const;

  /*!
   * \brief Method to set the number of overlap cell layers of the patch.
   * \param layer_size the number of overlap cell layers
   */
  void setOverlapLayerSize(Int32 layer_size);

  /*!
   * \brief Method to retrieve the min position of the enclosing box including the overlap cell layer.
   * \return The min position with the overlap cell layer.
   */
  CartCoord3 minPointWithOverlap() const;

  /*!
   * \brief Method to retrieve the max position of the enclosing box including the overlap cell layer.
   * \return The max position with the overlap cell layer.
   */
  CartCoord3 maxPointWithOverlap() const;

  /*!
   * \brief Method to know the number of cells in the patch according to its position.
   *
   * \warning The number of cells is calculated using the min and max positions
   * (without the overlap layer). This number is therefore the same for all
   * subdomains. Be careful not to compare this number with the number of cells
   * in the cell group that may be associated with this class, which may be
   * different for each subdomain.
   *
   * \return The number of cells in the patch.
   */
  Int64 nbCells() const;

  /*!
   * \brief Method to cut the patch into two patches according to a cut point.
   *
   * \param cut_point The cut point.
   * \param dim The dimension that must be cut.
   * \return The two patch positions resulting from the cut.
   */
  std::pair<AMRPatchPosition, AMRPatchPosition> cut(CartCoord cut_point, Integer dim) const;

  /*!
   * \brief Method to know if our patch can be merged with \a other_patch.
   *
   * \param other_patch The patch to check.
   * \return True if merging is possible.
   */
  bool canBeFusion(const AMRPatchPosition& other_patch) const;

  /*!
   * \brief Method to merge \a other_patch with ours.
   *
   * A check for possible merging (via \a canBeFusion()) is performed before
   * merging. If merging is impossible, false is returned. Otherwise, we merge
   * and return true. If merged, \a other_patch becomes null.
   *
   * \param other_patch The patch to merge with.
   * \return true if the merge was successful, false if the merge is impossible.
   */
  bool fusion(AMRPatchPosition& other_patch);

  /*!
   * \brief Method to know if the patch is null.
   *
   * \warning The validity of the position is not checked.
   *
   * \return True if the patch is null.
   */
  bool isNull() const;

  /*!
   * \brief Method to create an \a AMRPatchPosition for the higher level.
   *
   * \param dim The dimension of the mesh.
   * \param higher_level The highest refinement level of the mesh.
   * \param overlap_layer_size_top_level The number of overlap cell layers for patches at the highest refinement level.
   * \return A higher-level \a AMRPatchPosition.
   */
  AMRPatchPosition patchUp(Integer dim, Int32 higher_level, Int32 overlap_layer_size_top_level) const;

  /*!
   * \brief Method to create an \a AMRPatchPosition for the lower level.
   *
   * If the min position is not divisible by two, it is rounded down to the lower integer.
   * If the max position is not divisible by two, it is rounded up to the higher integer.
   *
   * For the overlap layer, this method ensures that there will never be more than one level difference between two cells of different levels.
   *
   * \warning patch.patchDown(patch.patchUp(X)) != patch and patch.patchUp(patch.patchDown(X)) != patch.
   *
   * \param dim The dimension of the mesh.
   * \param higher_level The highest refinement level of the mesh.
   * \param overlap_layer_size_top_level The number of overlap cell layers for patches at the highest refinement level.
   * \return A lower-level \a AMRPatchPosition.
   */
  AMRPatchPosition patchDown(Integer dim, Int32 higher_level, Int32 overlap_layer_size_top_level) const;

  /*!
   * \brief Method to know the size of the patch (in number of cells per direction).
   *
   * \return The size of the patch.
   */
  CartCoord3 length() const;

  /*!
   * \brief Method to know if a cell at position x,y,z is included in this patch.
   *
   * To include the overlap layer, use the \a isInWithOverlap() method.
   *
   * \param x X position of the cell.
   * \param y Y position of the cell.
   * \param z Z position of the cell.
   *
   * \return True if the cell is in the patch.
   */
  bool isIn(CartCoord x, CartCoord y, CartCoord z) const;

  /*!
   * \brief Method to know if a cell is included in this patch.
   *
   * To include the overlap layer, use the \a isInWithOverlap() method.
   *
   * \param coord Position of the cell.
   *
   * \return True if the cell is in the patch.
   */
  bool isIn(CartCoord3 coord) const;

  /*!
   * \brief Method to know if a cell at position x,y,z is included in this patch with overlap layer.
   *
   * \param x X position of the cell.
   * \param y Y position of the cell.
   * \param z Z position of the cell.
   *
   * \return True if the cell is in the patch.
   */
  bool isInWithOverlap(CartCoord x, CartCoord y, CartCoord z) const;

  /*!
   * \brief Method to know if a cell is included in this patch with overlap layer.
   *
   * \param coord Position of the cell.
   *
   * \return True if the cell is in the patch.
   */
  bool isInWithOverlap(CartCoord3 coord) const;

  /*!
   * \brief Method to know if a cell at position x,y,z is included in this patch with a specified overlap layer.
   *
   * \param x X position of the cell.
   * \param y Y position of the cell.
   * \param z Z position of the cell.
   * \param overlap The number of overlap cells in the layer.
   *
   * \return True if the cell is in the patch.
   */
  bool isInWithOverlap(CartCoord x, CartCoord y, CartCoord z, Integer overlap) const;

  /*!
   * \brief Method to know if a cell is included in this patch with a specified overlap layer.
   *
   * \param coord Position of the cell.
   * \param overlap The number of overlap cells in the layer.
   *
   * \return True if the cell is in the patch.
   */
  bool isInWithOverlap(CartCoord3 coord, Integer overlap) const;

  /*!
   * \brief Method to know if our patch is in contact with \a other.
   *
   * \param other The patch to check.
   * \return True if the patches are in contact.
   */
  bool haveIntersection(const AMRPatchPosition& other) const;

  /*!
   * \brief Method to calculate the number of overlap cell layers for a given level.
   *
   * \param level The requested level.
   * \param higher_level The highest refinement level.
   * \param overlap_layer_size_top_level The number of layers for the highest refinement level.
   * \return The number of overlap cell layers for the requested level.
   */
  static Int32 computeOverlapLayerSize(Int32 level, Int32 higher_level, Int32 overlap_layer_size_top_level);

  /*!
   * \brief Method to calculate the number of overlap cell layers for our patch.
   *
   * \param higher_level The highest refinement level.
   * \param overlap_layer_size_top_level The number of layers for the highest refinement level.
   */
  void computeOverlapLayerSize(Int32 higher_level, Int32 overlap_layer_size_top_level);

 private:

  Int32 m_level;
  CartCoord3 m_min_point;
  CartCoord3 m_max_point;
  Int32 m_overlap_layer_size;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
