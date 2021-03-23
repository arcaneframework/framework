// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Euclidian3Geometry.h                                        (C) 2000-2020 */
/*                                                                           */
/* Calculs géométriques en 3D Euclidienne.                                   */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_GEOMETRY_EUCLIDIAN_EUCLIDIAN3GEOMETRY_H
#define ARCANE_GEOMETRY_EUCLIDIAN_EUCLIDIAN3GEOMETRY_H
/*---------------------------------------------------------------------------*/

#include "arcane/geometry/IGeometry.h"
#include "arcane/ArcaneTypes.h"
#include "arcane/MeshVariable.h"
#include "arcane/VariableTypedef.h"
#include "arcane/MathUtils.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Numerics
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class Euclidian3Geometry
: public IGeometry
{
public:
  Euclidian3Geometry(const VariableNodeReal3 & coords)
    : m_coords(coords) { }

  virtual ~Euclidian3Geometry() { }
  
  //@{ @name Inherited methods from IGeometry

  //! Calcul du centre de masse
  Real3 computeCenter(const ItemWithNodes & item);

  //! Calcul de la mesure orientée
  /*! Dans le cas d'un élément plan, ceci correspond à 
   *  la normale moyenne unitaire * mesure de l'élément
   *  et dans le cas d'un simple élément volumique nous obtenons
   *  volume * z (ou z=(0,0,1))
   */
  Real3 computeOrientedMeasure(const ItemWithNodes & item);

  //! Calcul de la mesure (sans orientation)
  Real  computeMeasure(const ItemWithNodes & item);

  //! Calcul de la longueyr
  /*! Uniquement pour les Items linéïques */
  Real  computeLength(const ItemWithNodes & item);

  //! Calcul de l'aire 
  /*! Uniquement pour les Items surfaciques */
  Real  computeArea(const ItemWithNodes & item);

  //! Calcul du volume
  /*! Uniquement pour les Items volumiques */
  Real  computeVolume(const ItemWithNodes & item);

  //! Calcul du centre
  Real3 computeSurfaceCenter(Integer n, const Real3 * coords);

  //! Calcul de l'aire orientée (ie normale)
  Real3 computeOrientedArea(Integer n, const Real3 * coords);

  //! Calcul de longueur d'un segment défini par deux points
  Real computeLength(const Real3& m, const Real3& n);

  //@}
  
public:
  //@{ @name specific computations on basic types (interface depends to the dimension)
  struct IComputeLine {
    IComputeLine(Euclidian3Geometry * geom) : m_geom(geom), m_coords(geom->m_coords) { }
    virtual ~IComputeLine() { }
    virtual void computeOrientedMeasureAndCenter(const ItemWithNodes & item, Real3 & orientation, Real3 & center) = 0;
    Euclidian3Geometry * m_geom;
    const VariableNodeReal3 & m_coords;
  };

  struct ComputeLine2 : public IComputeLine { 
    ComputeLine2(Euclidian3Geometry * geom) : IComputeLine(geom) { }
    void computeOrientedMeasureAndCenter(const ItemWithNodes & item, Real3 & orientation, Real3 & center);
  };

  struct IComputeSurface {
    IComputeSurface(Euclidian3Geometry * geom) : m_geom(geom), m_coords(geom->m_coords) { }
    virtual ~IComputeSurface() { }
    virtual void computeOrientedMeasureAndCenter(const ItemWithNodes & item, Real3 & orientation, Real3 & center) = 0;
    Euclidian3Geometry * m_geom;
    const VariableNodeReal3 & m_coords;
  };

  struct ComputeTriangle3
  : public IComputeSurface {
    ComputeTriangle3(Euclidian3Geometry * geom) : IComputeSurface(geom) { }
    void computeOrientedMeasureAndCenter(const ItemWithNodes & item, Real3 & orientation, Real3 & center);
  };

  struct ComputeQuad4 : public IComputeSurface { 
    ComputeQuad4(Euclidian3Geometry * geom) : IComputeSurface(geom) { }
    void computeOrientedMeasureAndCenter(const ItemWithNodes & item, Real3 & orientation, Real3 & center);
  };

  struct ComputePentagon5 : public IComputeSurface { 
    ComputePentagon5(Euclidian3Geometry * geom) : IComputeSurface(geom) { }
    void computeOrientedMeasureAndCenter(const ItemWithNodes & item, Real3 & orientation, Real3 & center);
  };

  struct ComputeHexagon6 : public IComputeSurface { 
    ComputeHexagon6(Euclidian3Geometry * geom) : IComputeSurface(geom) { }
    void computeOrientedMeasureAndCenter(const ItemWithNodes & item, Real3 & orientation, Real3 & center);
  };

  struct IComputeVolume {
    IComputeVolume(Euclidian3Geometry * geom) : m_geom(geom), m_coords(geom->m_coords) { }
    virtual ~IComputeVolume() { }
    virtual void computeOrientedMeasureAndCenter(const ItemWithNodes & item, Real & measure, Real3 & center) = 0;
    virtual void computeVolumeArea(const ItemWithNodes & item, Real & area) = 0;
    Euclidian3Geometry * m_geom;
    const VariableNodeReal3 & m_coords;
  };

  struct ComputeTetraedron4 : public IComputeVolume { 
    ComputeTetraedron4(Euclidian3Geometry * geom) : IComputeVolume(geom) { }    
    void computeOrientedMeasureAndCenter(const ItemWithNodes & item, Real & measure, Real3 & center);
    void computeVolumeArea(const ItemWithNodes & item, Real & area);
  };

  struct ComputeHeptaedron10 : public IComputeVolume { 
    ComputeHeptaedron10(Euclidian3Geometry * geom) : IComputeVolume(geom) { }    
    void computeOrientedMeasureAndCenter(const ItemWithNodes & item, Real & measure, Real3 & center)
    {
      ARCANE_UNUSED(item);
      ARCANE_UNUSED(measure);
      ARCANE_UNUSED(center);
      ARCANE_THROW(NotImplementedException,"");
    }
    void computeVolumeArea(const ItemWithNodes & item, Real & area)
    {
      ARCANE_UNUSED(item);
      ARCANE_UNUSED(area);
      ARCANE_THROW(NotImplementedException,"");
    }
  };

  struct ComputeOctaedron12
  : public IComputeVolume
  {
    ComputeOctaedron12(Euclidian3Geometry * geom) : IComputeVolume(geom) { }    
    void computeOrientedMeasureAndCenter(const ItemWithNodes & item, Real & measure, Real3 & center)
    {
      ARCANE_UNUSED(item);
      ARCANE_UNUSED(measure);
      ARCANE_UNUSED(center);
      ARCANE_THROW(NotImplementedException,"");
    }
    void computeVolumeArea(const ItemWithNodes & item, Real & area)
    {
      ARCANE_UNUSED(item);
      ARCANE_UNUSED(area);
      ARCANE_THROW(NotImplementedException,"");
    }
  };

  struct ComputeGenericVolume : public IComputeVolume { 
    ComputeGenericVolume(Euclidian3Geometry * geom) : IComputeVolume(geom) { }    
    void computeOrientedMeasureAndCenter(const ItemWithNodes & item, Real & measure, Real3 & center);
    void computeVolumeArea(const ItemWithNodes & item, Real & area);
  };

  typedef ComputeGenericVolume ComputePyramid5;
  typedef ComputeGenericVolume ComputePentaedron6;
  typedef ComputeGenericVolume ComputeHexaedron8;
  typedef ComputeGenericVolume ComputeHemiHexa7;
  typedef ComputeGenericVolume ComputeHemiHexa6;
  typedef ComputeGenericVolume ComputeHemiHexa5;
  typedef ComputeGenericVolume ComputeAntiWedgeLeft6;
  typedef ComputeGenericVolume ComputeAntiWedgeRight6;
  typedef ComputeGenericVolume ComputeDiTetra5;
  //@}

  //@{ @name primitive used for volume decomposition
  static inline Real3 computeTriangleNormal(const Real3 & n0, const Real3 & n1, const Real3 & n2)
  {
    return math::vecMul(n1-n0,n2-n0) / 2.0;
  }

  static inline Real computeTriangleSurface(const Real3 & n0, const Real3 & n1, const Real3 & n2)
  {
    return math::normeR3(computeTriangleNormal(n0,n1,n2));
  }

  static inline Real3 computeTriangleCenter(const Real3 & n0, const Real3 & n1, const Real3 & n2)
  {
    return (n0+n1+n2) / 3.0;
  }

  static inline Real computeTetraedronVolume(const Real3 & n0, const Real3 & n1, const Real3 & n2, const Real3 & n3)
  {
    return math::mixteMul(n1-n0,n2-n0,n3-n0) / 6.0;
  }

  static inline Real3 computeTetraedronCenter(const Real3 & n0, const Real3 & n1, const Real3 & n2, const Real3 & n3)
  {
    return 0.25 * (n0+n1+n2+n3);
  }

  static inline Real3 computeQuadrilateralCenter(const Real3 & n0, const Real3 & n1, const Real3 & n2, const Real3 & n3)
  {
    const Real s0 = computeTriangleSurface(n0,n1,n2);
    const Real s1 = computeTriangleSurface(n0,n2,n3);
    const Real s2 = computeTriangleSurface(n1,n2,n3);
    const Real s3 = computeTriangleSurface(n0,n1,n3);
    return (s0 * computeTriangleCenter(n0,n1,n2) +
            s1 * computeTriangleCenter(n0,n2,n3) +
            s2 * computeTriangleCenter(n1,n2,n3) +
            s3 * computeTriangleCenter(n0,n1,n3)) / (s0+s1+s2+s3);    
  }

  static inline Real3 computePentagonalCenter(const Real3 & n0, const Real3 & n1, const Real3 & n2, const Real3 & n3, const Real3 & n4)
  {
    const Real s0 = computeTriangleSurface(n4,n0,n1);
    const Real s1 = computeTriangleSurface(n0,n1,n2);
    const Real s2 = computeTriangleSurface(n1,n2,n3);
    const Real s3 = computeTriangleSurface(n2,n3,n4);
    const Real s4 = computeTriangleSurface(n3,n4,n0);
    const Real s5 = computeTriangleSurface(n0,n2,n3);
    const Real s6 = computeTriangleSurface(n1,n3,n4);
    const Real s7 = computeTriangleSurface(n2,n4,n0);
    const Real s8 = computeTriangleSurface(n3,n0,n1);
    const Real s9 = computeTriangleSurface(n4,n1,n2);

    return (2*(s0 * computeTriangleCenter(n4,n0,n1) +
               s1 * computeTriangleCenter(n0,n1,n2) +
               s2 * computeTriangleCenter(n1,n2,n3) +
               s3 * computeTriangleCenter(n2,n3,n4) +
               s4 * computeTriangleCenter(n3,n4,n0)) +
            s5 * computeTriangleCenter(n0,n2,n3) +
            s6 * computeTriangleCenter(n1,n3,n4) +
            s7 * computeTriangleCenter(n2,n4,n0) +
            s8 * computeTriangleCenter(n3,n0,n1) +
            s9 * computeTriangleCenter(n4,n1,n2)) / (2*(s0+s1+s2+s3+s4) + s5 + s6 + s7 + s8 + s9);
  }

  static inline Real3 computeHexagonalCenter(const Real3 & n0, const Real3 & n1, const Real3 & n2, const Real3 & n3, const Real3 & n4, const Real3 & n5)
  {
    const Real s0 = computeTriangleSurface(n0,n1,n5);
    const Real s1 = computeTriangleSurface(n1,n2,n3);
    const Real s2 = computeTriangleSurface(n3,n4,n5);
    const Real s3 = computeTriangleSurface(n1,n3,n5);
    const Real s4 = computeTriangleSurface(n0,n1,n2);
    const Real s5 = computeTriangleSurface(n2,n3,n4);
    const Real s6 = computeTriangleSurface(n4,n5,n0);
    const Real s7 = computeTriangleSurface(n0,n2,n4);

    return (s0 * computeTriangleCenter(n0,n1,n5) +
            s1 * computeTriangleCenter(n1,n2,n3) +
            s2 * computeTriangleCenter(n3,n4,n5) +
            s3 * computeTriangleCenter(n1,n3,n5) +
            s4 * computeTriangleCenter(n0,n1,n2) +
            s5 * computeTriangleCenter(n2,n3,n4) +
            s6 * computeTriangleCenter(n4,n5,n0) +
            s7 * computeTriangleCenter(n0,n2,n4)) / (s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7);
  }
  //@}

protected:
  const VariableNodeReal3 & m_coords;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Numerics

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
