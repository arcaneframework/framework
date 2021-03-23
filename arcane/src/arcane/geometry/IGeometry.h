// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#ifndef ARCGEOSIM_GEOMETRY_IGEOMETRY_H
#define ARCGEOSIM_GEOMETRY_IGEOMETRY_H

#include "arcane/utils/Real3.h"
#include "arcane/Item.h"
#include "arcane/MathUtils.h"

namespace Arcane
{
namespace Numerics
{

using namespace Arcane;

/*!
 * \brief Interface de calculs géométriques.
 */
class ARCANE_GEOMETRY_EXPORT IGeometry
{
public:
  /** Constructeur de la classe */
  IGeometry()
    {
      ;
    }

  /** Destructeur de la classe */
  virtual ~IGeometry() { }

public:
  //! Calcul du centre de masse
  virtual Real3 computeCenter(const ItemWithNodes & item) = 0;

  //! Calcul de la mesure orientée
  /*! Dans le cas d'un élément plan, ceci correspond à
   *  la normale moyenne unitaire * mesure de l'élément
   *  et dans le cas d'un simple élément volumique nous obtenons
   *  volume * z (ou z=(0,0,1))
   */
  virtual Real3 computeOrientedMeasure(const ItemWithNodes & item) = 0;

  //! Calcul de la mesure (sans orientation)
  virtual Real  computeMeasure(const ItemWithNodes & item) = 0;

  //! Calcul de la longueyr
  /*! Uniquement pour les Items linéïques */
  virtual Real  computeLength(const ItemWithNodes & item) = 0;

  //! Calcul de l'aire
  /*! Uniquement pour les Items surfaciques */
  virtual Real  computeArea(const ItemWithNodes & item) = 0;

  //! Calcul du volume
  /*! Uniquement pour les Items volumiques */
  virtual Real  computeVolume(const ItemWithNodes & item) = 0;

  //! Calcul du centre
  /*! Uniquement pour les Items surfaciques */
  virtual Real3 computeSurfaceCenter(Integer n, const Real3 * coords) = 0;

  //! Calcul de l'aire orientée (ie normale)
  /*! Uniquement pour les Items surfaciques */
  virtual Real3 computeOrientedArea(Integer n, const Real3 * coords) = 0;

  //! Calcul de longueur d'un segment défini par deux points
  virtual Real computeLength(const Real3& m, const Real3& n) = 0;
};

} // End namespace Arcane

} // End namespace Numerics

#endif
