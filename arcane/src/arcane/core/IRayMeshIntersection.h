// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IRayMeshIntersection.h                                      (C) 2000-2025 */
/*                                                                           */
/* Calculation of the intersection between segments and the surface of a     */
/* mesh.                                                                     */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IRAYMESHINTERSECTION_H
#define ARCANE_CORE_IRAYMESHINTERSECTION_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Generic interface for calculating the intersection of a ray with a face.
 */
class IRayFaceIntersector
{
 public:

  virtual ~IRayFaceIntersector() {}

 public:

  /*!
   * \brief Calculates the intersection between a ray and a face.
   *
   * \param origin ray origin position.
   * \param direction direction of the ray.
   * \param orig_face_local_id local ID of the ray's origin face
   * \param face_nodes positions of the face nodes.
   * \param face_local_id local ID of the face. If it is not
   * a sub-domain face, it equals ITEM_NULL_LOCAL_ID.
   * \param user_value user value to be filled by the caller if necessary.
   * \param distance returned, intersection distance if one exists.
   * \param intersection_position returned, position of the intersection point.
   * \return true if an intersection is found, false otherwise.
   */
  virtual bool computeIntersection(Real3 origin, Real3 direction,
                                   Int32 orig_face_local_id,
                                   Int32 face_local_id,
                                   Real3ConstArrayView face_nodes,
                                   Int32* user_value,
                                   Real* distance, Real3* intersection_position) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Calculation of the intersection between a set of segments and the surface
 * of a mesh.
 */
class IRayMeshIntersection
{
 public:

  //! Frees resources.
  virtual ~IRayMeshIntersection() = default;

 public:

  //! Builds the instance.
  virtual void build() = 0;

 public:

  /*!
   * \brief Calculates the intersection.
   *
   * In return, the array \a faces_local_id contains the local ID
   * of the intersected face for each segment. If a segment
   * does not intersect any face, the corresponding local_id is NULL_ITEM_LOCAL_ID.
   */
  virtual void compute(Real3ConstArrayView segments_position,
                       Real3ConstArrayView segments_direction,
                       Int32ConstArrayView orig_faces_local_id,
                       Int32ArrayView user_values,
                       Real3ArrayView intersections,
                       RealArrayView distances,
                       Int32ArrayView faces_local_id) = 0;

  /*!
   * \brief Calculates the intersection of rays.
   *
   * Calculates the intersection of rays in the family \a ray_family
   * with the surface of the mesh. The position and direction
   * of the rays are given by the variables \a rays_position
   * and \a rays_direction. The array \a rays_orig_face contains
   * the local ID of the face from which the ray originates. This array
   * is used in the IRayFaceIntersector.
   *
   * In return, \a rays_face will contain for each ray the localId()
   * of the intersected face
   * or NULL_ITEM_LOCAL_ID if a ray does not intersect any face.
   * The array \a rays_intersection returns the position
   * of the intersection point and \a rays_distance the distance
   * of the intersection point relative to the ray origin.
   * The array \a user_values is filled in return by
   * the IRayFaceIntersector.
   *
   * In parallel, the rays in the family are exchanged
   * between sub-domains so that a ray is in the
   * same sub-domain as the owner of the intersected face.
   * If a ray does not intersect any face, it remains in
   * this sub-domain.
   */
  virtual void compute(IItemFamily* ray_family,
                       VariableParticleReal3& rays_position,
                       VariableParticleReal3& rays_direction,
                       VariableParticleInt32& rays_orig_face,
                       VariableParticleInt32& user_values,
                       VariableParticleReal3& intersections,
                       VariableParticleReal& distances,
                       VariableParticleInt32& rays_face) = 0;

  /*!
   * \brief Sets the intersection callback.
   *
   * This allows the caller to specify its method for calculating
   * the intersection of a ray with a face. If this method is not
   * called, a default intersector is used.   
   */
  virtual void setFaceIntersector(IRayFaceIntersector* intersector) = 0;

  //! Intersector used (0 if none specified)
  virtual IRayFaceIntersector* faceIntersector() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
