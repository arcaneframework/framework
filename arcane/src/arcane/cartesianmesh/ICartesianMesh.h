// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ICartesianMesh.h                                            (C) 2000-2026 */
/*                                                                           */
/* Interface of a Cartesian mesh.                                            */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CARTESIANMESH_ICARTESIANMESH_H
#define ARCANE_CARTESIANMESH_ICARTESIANMESH_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/cartesianmesh/CartesianMeshGlobal.h"

#include "arcane/core/ArcaneTypes.h"
#include "arcane/core/MeshHandle.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup ArcaneCartesianMesh
 * \brief Interface of a Cartesian mesh.
 */
class ARCANE_CARTESIANMESH_EXPORT ICartesianMesh
{
 public:

  virtual ~ICartesianMesh() {} //<! Frees resources

  /*!
   * \brief Retrieves or creates the reference associated with \a mesh.
   *
   * If no material manager is associated with \a mesh, it
   * will be created when this method is called if \a create is \a true.
   * If \a create is \a false, no manager is associated
   * to the mesh, a null pointer is returned.
   * The returned instance remains valid as long as the mesh \a mesh exists.
   */
  static ICartesianMesh* getReference(const MeshHandleOrMesh& mesh, bool create = true);

 public:

  virtual void build() = 0;

 public:

  //! Mesh associated with this Cartesian mesh
  virtual IMesh* mesh() const = 0;

  //! Associated trace manager.
  virtual ITraceMng* traceMng() const = 0;

  //! List of cells in direction \a dir
  virtual CellDirectionMng cellDirection(eMeshDirection dir) = 0;

  //! List of cells in direction \a dir (0, 1 or 2)
  virtual CellDirectionMng cellDirection(Integer idir) = 0;

  //! List of faces in direction \a dir
  virtual FaceDirectionMng faceDirection(eMeshDirection dir) = 0;

  //! List of faces in direction \a dir (0, 1 or 2)
  virtual FaceDirectionMng faceDirection(Integer idir) = 0;

  //! List of nodes in direction \a dir
  virtual NodeDirectionMng nodeDirection(eMeshDirection dir) = 0;

  //! List of nodes in direction \a dir (0, 1 or 2)
  virtual NodeDirectionMng nodeDirection(Integer idir) = 0;

  /*!
   * \brief Calculates information for directional access.
   *
   * Currently, the following restrictions exist:
   * - only calculates information on cell entities.
   * - assumes that cell 0 is in a corner (only works
   * for the meshgenerator).
   * - directional information is invalidated if the mesh changes.
   */
  virtual void computeDirections() = 0;

  /*!
   * \brief Recalculates Cartesian information after a restart.
   *
   * This method must be called instead of computeDirections()
   * during a restart.
   */
  virtual void recreateFromDump() = 0;

  //! Connectivity information
  virtual CartesianConnectivity connectivity() = 0;

  /*!
   * \brief Number of patches in the mesh.
   *
   * There is always at least one patch that represents the Cartesian mesh
   *
   * \deprecated Use CartesianMeshAMRMng instead.
   */
  virtual Int32 nbPatch() const = 0;

  /*!
   * \brief Returns the \a index-th patch of the mesh.
   *
   * If the mesh is Cartesian, there is only one patch.
   *
   * The returned instance remains valid as long as this instance is not destroyed.
   */
  virtual ICartesianMeshPatch* patch(Int32 index) const = 0;

  /*!
   * \brief Returns the \a index-th patch of the mesh.
   *
   * If the mesh is Cartesian, there is only one patch.
   *
   * The returned instance remains valid as long as this instance is not destroyed.
   *
   * \deprecated Use CartesianMeshAMRMng::amrPatch() instead.
   */
  virtual CartesianPatch amrPatch(Int32 index) const = 0;

  /*!
   * \brief View of the list of patches.
   *
   * \deprecated Use CartesianMeshAMRMng::amrPatch() instead.
   */
  virtual CartesianMeshPatchListView patches() const = 0;

  /*!
   * \brief Refines a block of the Cartesian mesh in 2D.
   *
   * This method can only be called if the mesh is an
   * AMR mesh (IMesh::isAmrActivated()==true).
   *
   * The cells whose center positions are between
   * \a position and \a (position+length) are refined and the corresponding
   * connectivity information is updated.
   *
   * This operation is collective.
   *
   * \deprecated Use CartesianMeshAMRMng::refineZone() instead.
   */
  virtual void refinePatch2D(Real2 position, Real2 length) = 0;

  /*!
   * \brief Refines a block of the Cartesian mesh in 3D.
   *
   * This method can only be called if the mesh is an
   * AMR mesh (IMesh::isAmrActivated()==true).
   *
   * The cells whose center positions are between
   * \a position and \a (position+length) are refined and the corresponding
   * connectivity information is updated.
   *
   * This operation is collective.
   *
   * \deprecated Use CartesianMeshAMRMng::refineZone() instead.
   */
  virtual void refinePatch3D(Real3 position, Real3 length) = 0;

  /*!
   * \brief Refines a block of the Cartesian mesh.
   *
   * This method can only be called if the mesh is an
   * AMR mesh (IMesh::isAmrActivated()==true).
   *
   * The cells whose center positions are between
   * \a position and \a (position+length) are refined and the corresponding
   * connectivity information is updated.
   *
   * This operation is collective.
   *
   * \deprecated Use CartesianMeshAMRMng::refineZone() instead.
   */
  virtual void refinePatch(const AMRZonePosition& position) = 0;

  /*!
   * \brief Coarsens a block of the Cartesian mesh in 2D.
   *
   * This method can only be called if the mesh is an
   * AMR mesh (IMesh::isAmrActivated()==true).
   *
   * The cells whose center positions are between
   * \a position and \a (position+length) are coarsened and the corresponding
   * connectivity information is updated.
   *
   * All cells in the coarsening zone must be of the same
   * level.
   *
   * Patches that no longer contain cells after calling this method
   * will be deleted.
   *
   * This operation is collective.
   *
   * \deprecated Use CartesianMeshAMRMng::coarseZone2D() instead.
   */
  virtual void coarseZone2D(Real2 position, Real2 length) = 0;

  /*!
   * \brief Coarsens a block of the Cartesian mesh in 3D.
   *
   * This method can only be called if the mesh is an
   * AMR mesh (IMesh::isAmrActivated()==true).
   *
   * The cells whose center positions are between
   * \a position and \a (position+length) are coarsened and the corresponding
   * connectivity information is updated.
   *
   * All cells in the coarsening zone must be of the same
   * level.
   *
   * Patches that no longer contain cells after calling this method
   * will be deleted.
   *
   * This operation is collective.
   *
   * \deprecated Use CartesianMeshAMRMng::coarseZone2D() instead.
   */
  virtual void coarseZone3D(Real3 position, Real3 length) = 0;

  /*!
   * \brief Coarsens a block of the Cartesian mesh.
   *
   * This method can only be called if the mesh is an
   * AMR mesh (IMesh::isAmrActivated()==true).
   *
   * The cells whose center positions are between
   * \a position and \a (position+length) are coarsened and the corresponding
   * connectivity information is updated.
   *
   * All cells in the coarsening zone must be of the same
   * level.
   *
   * Patches that no longer contain cells after calling this method
   * will be deleted.
   *
   * This operation is collective.
   *
   * \deprecated Use CartesianMeshAMRMng::coarseZone2D() instead.
   */
  virtual void coarseZone(const AMRZonePosition& position) = 0;

  /*!
   * \brief Method for deleting one or more layers
   * of ghost cells at a defined refinement level.
   *
   * The desired number of ghost cell layers may be increased
   * by the method. It is necessary to retrieve the returned value
   * to get the final number of ghost cell layers.
   *
   * \param level The refinement level concerned by the deletion
   * of ghost cells.
   *
   * \param target_nb_ghost_layers The desired number of layers after
   * calling this method. ATTENTION: It may be adjusted by the method.
   *
   * \return The final number of ghost cell layers.
   *
   * \deprecated Use CartesianMeshAMRMng::reduceNbGhostLayers() instead.
   */
  virtual Integer reduceNbGhostLayers(Integer level, Integer target_nb_ghost_layers) = 0;

  /*!
   * \brief Renumbers the uniqueId() of entities.
   *
   * Based on the values of \a v, the uniqueId() of faces and/or
   * entities of the patches is renumbered to have the same numbering
   * regardless of the decomposition.
   */
  virtual void renumberItemsUniqueId(const CartesianMeshRenumberingInfo& v) = 0;

  //! Performs checks on the validity of the instance.
  virtual void checkValid() const = 0;

  /*!
   * \brief Creates an instance to manage mesh coarsening.
   * \deprecated Use Arcane::CartesianMeshUtils::createCartesianMeshCoarsening2() instead.
   */
  ARCANE_DEPRECATED_REASON("Y2024: Use Arcane::CartesianMeshUtils::createCartesianMeshCoarsening2() instead")
  virtual Ref<CartesianMeshCoarsening> createCartesianMeshCoarsening() = 0;

  virtual void computeDirectionsPatchV2(Integer index) = 0;

 public:

  //! Internal Arcane API
  virtual ICartesianMeshInternal* _internalApi() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal.
 * This method is reserved for Arcane.
 * Use ICartesianMesh::getReference() to create an instance.
 */
extern "C++" ARCANE_CARTESIANMESH_EXPORT ICartesianMesh*
arcaneCreateCartesianMesh(IMesh* mesh);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
