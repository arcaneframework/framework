// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMeshUniqueIdMng.h                                          (C) 2000-2021 */
/*                                                                           */
/* Interface du gestionnaire de numérotation des uniqueId() d'un maillage.   */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IMESHUNIQUEIDMNG_H
#define ARCANE_IMESHUNIQUEIDMNG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/ArcaneTypes.h"

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
  virtual ~IMeshUniqueIdMng() =default;

 public:

  /*!
   * \brief Positionne la version de la numérotation des faces.
   *
   * Les valeurs valides sont 0, 1, 2 et 3. La valeur par défaut est 1.
   * Si la version vaut 0 alors il n'y a pas de renumérotation. En parallèle,
   * il faut alors que les uniqueId() des faces soient cohérents entre
   * les sous-domaines
   */  
  virtual void setFaceBuilderVersion(Integer n) =0;

  //! Version de la numérotation des faces.
  virtual Integer faceBuilderVersion() const =0;

  /*!
   * \brief Positionne la version de la numérotation des arêtes.
   */  
  virtual void setEdgeBuilderVersion(Integer n) =0;

  //! Version de la numérotation des arêtes
  virtual Integer edgeBuilderVersion() const =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
