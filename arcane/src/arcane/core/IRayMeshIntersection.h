// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IRayMeshIntersection.h                                      (C) 2000-2025 */
/*                                                                           */
/* Calcul de l'intersection entre des segments et la surface d'un maillage.  */
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
 * \brief Interface générique du calcul de l'intersection d'un rayon avec une face.
 */
class IRayFaceIntersector
{
 public:

  virtual ~IRayFaceIntersector() {}

 public:

  /*!
   * \brief Calcul l'intersection entre un rayon et une face.
   *
   * \param origin position d'origine du rayon.
   * \param direction direction du rayon.
   * \param orig_face_local_id numéro local de la face d'origine du rayon
   * \param face_nodes positions des noeuds de la face.
   * \param face_local_id numéro local de la face. S'il ne s'agit
   * pas d'une face du sous-domaine, vaut ITEM_NULL_LOCAL_ID.
   * \param user_value valeur utilisateur à remplir par l'appelant si besoin
   * \param distance en retour, distance d'intersection s'il y en a une
   * \param intersection_position en retour, position du point d'intersection
   * \return true si une intersection est trouvée, false sinon.
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
 * \brief Calcul de l'intersection entre un ensemble de segments et la surface
 * d'un maillage.
 */
class IRayMeshIntersection
{
 public:

  //! Libère les ressources
  virtual ~IRayMeshIntersection() = default;

 public:

  //! Construit l'instance
  virtual void build() = 0;

 public:

  /*!
   * \brief Calcule l'intersection.
   *
   * En retour, le tableau \a faces_local_id contient le numéro
   * local de la face coupée pour chaque segment. Si un segment
   * ne coupe pas de face, le local_id correspondant est NULL_ITEM_LOCAL_ID.
   */
  virtual void compute(Real3ConstArrayView segments_position,
                       Real3ConstArrayView segments_direction,
                       Int32ConstArrayView orig_faces_local_id,
                       Int32ArrayView user_values,
                       Real3ArrayView intersections,
                       RealArrayView distances,
                       Int32ArrayView faces_local_id) = 0;

  /*!
   * \brief Calcule l'intersection de rayons.
   *
   * Calcul l'intersection des rayons de la famille \a ray_family
   * avec la surface du maillage. La position et la direction
   * des rayons est donnée par les variables \a rays_position
   * et \a rays_direction. Le tableau \a rays_orig_face contient
   * le numéro local de la face dont le rayon est originaire. Ce tableau
   * est utilisé dans le IRayFaceIntersector.
   *
   * En retour \a rays_face contiendra pour chaque rayon le localId()
   * de la face intersectée
   * ou NULL_ITEM_LOCAL_ID si un rayon n'intersecte aucune face.
   * Le tableau \a rays_intersection contient en retour la position
   * du point d'intersection et \a rays_distance la distance du
   * point d'intersection par rapport à l'origine du rayon.
   * Le tableau \a user_values est remplit en retour par
   * le IRayFaceIntersector
   *
   * En parallèle, les rayons de la famille sont échangés
   * entre sous-domaines pour qu'un rayon se trouve dans le
   * même sous-domaine que celui propriétaire de la face
   * intersectée.
   * Si un rayon n'intersecte pas de face, il reste dans
   * ce sous-domaine.
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
   * \brief Positionne le callback d'intersection.
   *
   * Cela permet à l'appelant de spécifier sa méthode de calcul
   * d'intersection d'un rayon avec une face. Si cette méthode n'est
   * pas appelée, un intersecteur par défaut est utilisé.   
   */
  virtual void setFaceIntersector(IRayFaceIntersector* intersector) = 0;

  //! Intersecteur utilisé (0 si aucun spécifié)
  virtual IRayFaceIntersector* faceIntersector() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

