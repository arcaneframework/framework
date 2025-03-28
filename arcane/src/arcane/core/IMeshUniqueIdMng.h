// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMeshUniqueIdMng.h                                          (C) 2000-2025 */
/*                                                                           */
/* Interface du gestionnaire de numérotation des uniqueId() d'un maillage.   */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IMESHUNIQUEIDMNG_H
#define ARCANE_IMESHUNIQUEIDMNG_H
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
 * \internal
 *
 * Interface du gestionnaire de numérotation des uniqueId() des entités
 * d'un maillage.
 *
 * Ce gestionnaire permet de gérer le calcul des uniqueId() des entités
 * du maillages qui sont implicitement créées comme les faces ou les arêtes.
 */
class ARCANE_CORE_EXPORT IMeshUniqueIdMng
{
 public:

  //! Libère les ressources
  virtual ~IMeshUniqueIdMng() = default;

 public:

  /*!
   * \brief Positionne la version de la numérotation des faces.
   *
   * Les valeurs valides sont 0, 1, 2 et 3. La valeur par défaut est 1.
   * Si la version vaut 0 alors il n'y a pas de renumérotation. En parallèle,
   * il faut alors que les uniqueId() des faces soient cohérents entre
   * les sous-domaines.
   */
  virtual void setFaceBuilderVersion(Integer n) = 0;

  //! Version de la numérotation des faces.
  virtual Integer faceBuilderVersion() const = 0;

  /*!
   * \brief Positionne la version de la numérotation des arêtes.
   *
   * Les valeurs valides sont 0, 1 et 2. La valeur 1 fonctionne quel que
   * soit le nombre de mailles mais il faut que le maillage soit lu par
   * un seul processeur. La valeur 2 ne fonctionne que si le maximum des
   * uniqueId() des noeuds ne dépasse pas 2^31.
   *
   * Si la version vaut 0 alors il n'y a pas de renumérotation. En parallèle,
   * il faut alors que les uniqueId() des faces soient cohérents entre
   * les sous-domaines.
   */
  virtual void setEdgeBuilderVersion(Integer n) = 0;

  //! Version de la numérotation des arêtes
  virtual Integer edgeBuilderVersion() const = 0;

  /*!
   * \brief Indique si on détermine les uniqueId() des arêtes et des faces en
   * fonction des uniqueId() des noeuds qui les constituent.
   *
   * Cette méthode doit être appelée avant de positionner la dimension
   * du maillage (IPrimaryMesh::setDimension()).
   *
   * Si actif, lors de la création à la volée d'une arête ou d'une face
   * on utilise MeshUtils::generateHashUniqueId() pour générer le uniqueId()
   * de l'entité. Cela permet en parallèle de créer automatiquement
   * les arêtes ou les faces.
   *
   * \warning Si ce mécanisme est utilisé, il ne faut pas le mélanger
   * avec la création manuelle des arêtes ou des faces (via IMeshModifier)
   * ou alors il faut utiliser MeshUtils::generateHashUniqueId() pour générer
   * le même identifiant que celui créé à la volée.
   */
  virtual void setUseNodeUniqueIdToGenerateEdgeAndFaceUniqueId(bool v) = 0;

  //! Indique le mécanisme utilisé pour numéroter les arêtes ou les faces
  virtual bool isUseNodeUniqueIdToGenerateEdgeAndFaceUniqueId() const = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
