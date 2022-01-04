// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* GeometricUnitTest.cc                                        (C) 2000-2014 */
/*                                                                           */
/* Service de test de la géométrie.                                          */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"
#include "arcane/utils/ScopedPtr.h"

#include "arcane/BasicUnitTest.h"
#include "arcane/ItemPrinter.h"
#include "arcane/IMesh.h"

#include "arcane/geometric/GeomShapeMng.h"
#include "arcane/geometric/BarycentricGeomShapeComputer.h"

#include "arcane/tests/ArcaneTestGlobal.h"

#include "arcane/tests/GeometricUnitTest_axl.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANETEST_BEGIN_NAMESPACE

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
  VariableNodeReal3& node_coords(mesh()->nodesCoordinates());

  // Met à jour les coordonnées des formes géométriques élément par élément
  ENUMERATE_CELL(icell,allCells()){
    Cell cell = *icell;
    geometric::BarycentricGeomShapeComputer::computeAll(m_shape_mng.mutableShapeView(cell),node_coords,cell);
  }
  _checkCoords(allCells());

  // Met à jour les coordonnées de manière globale.
  geometric::BarycentricGeomShapeComputer::computeAll(m_shape_mng,node_coords,allCells());
  _checkCoords(allCells());

  geometric::GeomShapeView shape_view;
  ENUMERATE_CELL(icell,allCells()){
    Cell cell = *icell;
    m_shape_mng.initShape(shape_view,*icell);
    Cell shape_cell = shape_view.cell();

    // Vérifie maille OK.
    if (shape_cell!=cell)
      throw FatalErrorException(A_FUNCINFO,"Invalid cell for shape");
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

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANETEST_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
