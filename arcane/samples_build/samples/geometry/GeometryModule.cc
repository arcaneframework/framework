// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#include <arcane/MathUtils.h>
#include <arcane/ITimeLoopMng.h>
#include <arcane/IMesh.h>
#include <arcane/ItemPrinter.h>

#include "Geometry_axl.h"

using namespace Arcane;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Real
Hexaedron8Volume(ItemWithNodes item,const VariableNodeReal3& n)
{
  Real3 n0 = n[item.node(0)];
  Real3 n1 = n[item.node(1)];
  Real3 n2 = n[item.node(2)];
  Real3 n3 = n[item.node(3)];
  Real3 n4 = n[item.node(4)];
  Real3 n5 = n[item.node(5)];
  Real3 n6 = n[item.node(6)];
  Real3 n7 = n[item.node(7)];

  Real v1 = math::matDet((n6 - n1) + (n7 - n0), n6 - n3, n2 - n0);
  Real v2 = math::matDet(n7 - n0, (n6 - n3) + (n5 - n0), n6 - n4);
  Real v3 = math::matDet(n6 - n1, n5 - n0, (n6 - n4) + (n2 - n0));

  Real res = (v1 + v2 + v3) / 12.0;

  return res;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Real
Pyramid5Volume(ItemWithNodes item,const VariableNodeReal3& n)
{
  Real3 n0 = n[item.node(0)];
  Real3 n1 = n[item.node(1)];
  Real3 n2 = n[item.node(2)];
  Real3 n3 = n[item.node(3)];
  Real3 n4 = n[item.node(4)];

  return math::matDet(n1 - n0, n3 - n0, n4 - n0) +
  math::matDet(n3 - n2, n1 - n2, n4 - n2);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Real
Quad4Surface(ItemWithNodes item,const VariableNodeReal3& n)
{
  Real3 n0 = n[item.node(0)];
  Real3 n1 = n[item.node(1)];
  Real3 n2 = n[item.node(2)];
  Real3 n3 = n[item.node(3)];

  Real x1 = n1.x - n0.x;
  Real y1 = n1.y - n0.y;
  Real x2 = n2.x - n1.x;
  Real y2 = n2.y - n1.y;
  Real surface = x1 * y2 - y1 * x2;

  x1 = n2.x - n0.x;
  y1 = n2.y - n0.y;
  x2 = n3.x - n2.x;
  y2 = n3.y - n2.y;

  surface += x1 * y2 - y1 * x2;

  return surface;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Real
Triangle3Surface(ItemWithNodes item,const VariableNodeReal3& n)
{
  Real3 n0 = n[item.node(0)];
  Real3 n1 = n[item.node(1)];
  Real3 n2 = n[item.node(2)];
  
  Real x1 = n1.x - n0.x;
  Real y1 = n1.y - n0.y;
  Real x2 = n2.x - n1.x;
  Real y2 = n2.y - n1.y;

  return x1 * y2 - y1 * x2;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class GeometryDispatcher
{
 public:
  GeometryDispatcher(VariableNodeReal3& node_coords)
  : m_node_coords(node_coords)
  {
    // Met à null par défaut.
    for( Integer i=0; i<NB_BASIC_ITEM_TYPE; ++i )
      m_functions[i] = nullptr;
    // Spécifie les fonctions pour chaque type qu'on supporte
    m_functions[IT_Triangle3] = Triangle3Surface;
    m_functions[IT_Quad4] = Quad4Surface;
    m_functions[IT_Pyramid5] = Pyramid5Volume;
    m_functions[IT_Hexaedron8] = Hexaedron8Volume;
  }
 public:
  Real apply(ItemWithNodes item)
  {
    Int32 item_type = item.type();
    auto f = m_functions[item_type];
    if (f!=nullptr)
      return f(item,m_node_coords);
    return (-1.0);
  }
 private:
  // Le nombre et les valeurs pour chaque type sont définis dans ArcaneTypes.h
  std::function<Real(ItemWithNodes item,const VariableNodeReal3& n)> m_functions[NB_BASIC_ITEM_TYPE];
  VariableNodeReal3 m_node_coords;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class GeometryModule
: public ArcaneGeometryObject
{
 public:
  explicit GeometryModule(const ModuleBuildInfo& mbi)
  : ArcaneGeometryObject(mbi)
  {
  }
  ~GeometryModule() override = default;

 public:

  void init() override;
  void computeSurfacesAndVolumes() override;
  VersionInfo versionInfo() const override { return VersionInfo(1, 0, 0); }

 private:
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void GeometryModule::
init()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void GeometryModule::
computeSurfacesAndVolumes()
{
  // Test d'arrêt de la boucle en temps
  if (m_global_iteration()>=options()->maxIteration())
    subDomain()->timeLoopMng()->stopComputeLoop(true);

  GeometryDispatcher geom_dispatcher(mesh()->nodesCoordinates());

  ENUMERATE_CELL(icell,allCells()){
    Cell cell = *icell;
    Real v = geom_dispatcher.apply(cell);
    info() << "Cell " << ItemPrinter(cell) << " Volume=" << v;
  }

  ENUMERATE_FACE(iface,allFaces()){
    Face face = *iface;
    Real v = geom_dispatcher.apply(face);
    info() << "Face " << ItemPrinter(face) << " Surface=" << v;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_MODULE_GEOMETRY(GeometryModule);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
