// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* GeometricUnitTest.cc                                        (C) 2000-2023 */
/*                                                                           */
/* Service de test de la géométrie.                                          */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ScopedPtr.h"
#include "arcane/utils/ValueChecker.h"

#include "arcane/BasicUnitTest.h"
#include "arcane/ItemPrinter.h"
#include "arcane/IMesh.h"

#include "arcane/geometric/GeomShapeMng.h"
#include "arcane/geometric/BarycentricGeomShapeComputer.h"

#include "arcane/tests/ArcaneTestGlobal.h"

#include "arcane/tests/GeometricUnitTest_axl.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace ArcaneTest
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arcane;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Service de test de la géométrie.
 */
class GeometricUnitTest
: public ArcaneGeometricUnitTestObject
{
 public:

  GeometricUnitTest(const ServiceBuildInfo& cb);
  ~GeometricUnitTest();

 public:

  virtual void initializeTest();
  virtual void executeTest();

 private:

  geometric::GeomShapeMng m_shape_mng;

 private:
  
  void _checkCoords(const CellGroup& cells);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE_GEOMETRICUNITTEST(GeometricUnitTest,GeometricUnitTest);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

GeometricUnitTest::
GeometricUnitTest(const ServiceBuildInfo& sb)
: ArcaneGeometricUnitTestObject(sb)
, m_shape_mng(sb.mesh())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

GeometricUnitTest::
~GeometricUnitTest()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void GeometricUnitTest::
_checkCoords(const CellGroup& cells)
{
  VariableNodeReal3& node_coords(mesh()->nodesCoordinates());
  geometric::GeomShapeView shape_view;
  ENUMERATE_CELL(icell,cells){
    Cell cell = *icell;
    m_shape_mng.initShape(shape_view,*icell);

    // Vérifie que les coordonnées des noeuds sont OK.
    Integer nb_node = cell.nbNode();
    for( Integer z=0; z<nb_node; ++z )
      if (node_coords[cell.node(z)] != shape_view.node(z))
        throw FatalErrorException(A_FUNCINFO,"Invalid node coordinates for GeomShape");
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void GeometricUnitTest::
executeTest()
{
  ValueChecker vc(A_FUNCINFO);
  VariableNodeReal3& node_coords(mesh()->nodesCoordinates());

  // Met à jour les coordonnées des formes géométriques élément par élément
  ENUMERATE_CELL (icell, allCells()) {
    Cell cell = *icell;
    geometric::BarycentricGeomShapeComputer::computeAll(m_shape_mng.mutableShapeView(cell), node_coords, cell);
  }
  _checkCoords(allCells());

  // Met à jour les coordonnées de manière globale.
  geometric::BarycentricGeomShapeComputer::computeAll(m_shape_mng, node_coords, allCells());
  _checkCoords(allCells());

  geometric::GeomShapeView shape_view;
  ENUMERATE_CELL (icell, allCells()) {
    Cell cell = *icell;
    m_shape_mng.initShape(shape_view, *icell);
    Cell shape_cell = shape_view.cell();

    // Vérifie maille OK.
    if (shape_cell != cell)
      ARCANE_FATAL("Invalid cell for shape '{0}'", cell.uniqueId());
    info() << "Cell type=" << cell.typeInfo()->typeName();
    if (cell.type() == IT_Hexaedron8) {
      geometric::Hexaedron8Element hex_element(node_coords, cell);
      for (int z = 0; z < 8; ++z) {
        vc.areEqual(hex_element[z], node_coords[cell.node(z)], "Node");
      }
      Real3 to_add(1.0, 2.0, 3.0);
      geometric::Hexaedron8ElementView view1(hex_element.view());
      for (int z = 0; z < 8; ++z) {
        view1.setValue(z, view1[z] + to_add);
      }
      for (int z = 0; z < 8; ++z) {
        vc.areEqual(hex_element[z], node_coords[cell.node(z)] + to_add, "Node2");
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void GeometricUnitTest::
initializeTest()
{
  m_shape_mng.initialize();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
