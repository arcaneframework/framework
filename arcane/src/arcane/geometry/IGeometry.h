// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
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
 * \brief Geometric calculation interface.
 */
class ARCANE_GEOMETRY_EXPORT IGeometry
{
public:
  /** Class constructor */
  IGeometry()
    {
      ;
    }

  /** Class destructor */
  virtual ~IGeometry() { }

public:
  //! Calculation of the center of mass
  virtual Real3 computeCenter(const ItemWithNodes & item) = 0;

  //! Calculation of the oriented measure
  /*! In the case of a planar element, this corresponds to
   *  the unit average normal * element measure
   *  and in the case of a simple volumetric element we obtain
   *  volume * z (or z=(0,0,1))
   */
  virtual Real3 computeOrientedMeasure(const ItemWithNodes & item) = 0;

  //! Calculation of the measure (without orientation)
  virtual Real  computeMeasure(const ItemWithNodes & item) = 0;

  //! Calculation of the length
  /*! Only for linear Items */
  virtual Real  computeLength(const ItemWithNodes & item) = 0;

  //! Calculation of the area
  /*! Only for surface Items */
  virtual Real  computeArea(const ItemWithNodes & item) = 0;

  //! Calculation of the volume
  /*! Only for volumetric Items */
  virtual Real  computeVolume(const ItemWithNodes & item) = 0;

  //! Calculation of the center
  /*! Only for surface Items */
  virtual Real3 computeSurfaceCenter(Integer n, const Real3 * coords) = 0;

  //! Calculation of the oriented area (i.e., normal)
  /*! Only for surface Items */
  virtual Real3 computeOrientedArea(Integer n, const Real3 * coords) = 0;

  //! Calculation of the length of a segment defined by two points
  virtual Real computeLength(const Real3& m, const Real3& n) = 0;
};

} // End namespace Arcane

} // End namespace Numerics

#endif
