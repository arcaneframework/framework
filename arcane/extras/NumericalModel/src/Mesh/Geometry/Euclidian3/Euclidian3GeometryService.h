﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
#ifndef ARCGEOSIM_GEOMETRY_EUCLIDIAN3GEOMETRYSERVICE_H
#define ARCGEOSIM_GEOMETRY_EUCLIDIAN3GEOMETRYSERVICE_H
/* Author : havep at Wed Nov 14 13:55:36 2007
 * Generated by createNew
 */

#include "Mesh/Geometry/IGeometryMng.h"
#include "Mesh/Geometry/Euclidian3/Euclidian3Geometry.h"

namespace Arcane { }
using namespace Arcane;

#include "Euclidian3Geometry_axl.h"
#include "Mesh/Geometry/Impl/GeometryServiceBase.h"

class Euclidian3GeometryService :
  public ArcaneEuclidian3GeometryObject,
  public GeometryServiceBase
{
public:
  /** Constructeur de la classe */
  Euclidian3GeometryService(const Arcane::ServiceBuildInfo & sbi);
  
  /** Destructeur de la classe */
  virtual ~Euclidian3GeometryService();
  
public:
    //! Initialisation
  void init();

  //@{ @name property management by group

  //! Update property values for an ItemGroup
  void update(ItemGroup group);

  //! Reset property for an ItemGroup
  void reset(ItemGroup group);
  
  //@}

  //! Get underlying geometry
  IGeometry * geometry();

public:
  //@{ Extended interface for GeometryServiceBase

  //! Access to traceMng
  ITraceMng * traceMng() { return ArcaneEuclidian3GeometryObject::traceMng(); }

  //! Access to Mesh
  IMesh * mesh() { return ArcaneEuclidian3GeometryObject::subDomain()->mesh(); }

  //! Name of instancied class
  const char * className() const { return "Euclidian3Geometry"; }

  //@}

private: 
  Euclidian3Geometry * m_geometry;
};

#endif /* ARCGEOSIM_GEOMETRY_EUCLIDIAN3GEOMETRYSERVICE_H */
