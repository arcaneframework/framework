// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* GeomShapeView.cc                                            (C) 2000-2014 */
/*                                                                           */
/* Gestion des formes géométriques 2D et 3D.                                 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"

#include "arcane/geometric/GeomElement.h"
#include "arcane/geometric/GeomShapeView.h"
#include "arcane/geometric/CellConnectivity.h"
//#include "arcane/geometric/CellGeomList.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE
GEOMETRIC_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*
 * TODO: Verifier que les IT_* correspondents aux énumérations de GeomType.
 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class GeomShapeConnectivity
{
 public:

  GeomShapeConnectivity()
  {
    for( int i=0; i<NB_BASIC_ITEM_TYPE; ++i )
      GeomShapeView::global_cell_connectivity[i] = 0;

    GeomShapeView::global_cell_connectivity[IT_NullType] = &m_null_connectivity;
    GeomShapeView::global_cell_connectivity[IT_Vertex] = &m_vertex_connectivity;
    GeomShapeView::global_cell_connectivity[IT_Line2] = &m_line2_connectivity;
    GeomShapeView::global_cell_connectivity[IT_Triangle3] = &m_triangle3_connectivity;
    GeomShapeView::global_cell_connectivity[IT_Quad4] = &m_quad4_connectivity;
    GeomShapeView::global_cell_connectivity[IT_Pentagon5] = &m_pentagon5_connectivity;
    GeomShapeView::global_cell_connectivity[IT_Hexagon6] = &m_hexagon6_connectivity;
    GeomShapeView::global_cell_connectivity[IT_Tetraedron4] = &m_tetraedron4_connectivity;
    GeomShapeView::global_cell_connectivity[IT_Pyramid5] = &m_pyramid5_connectivity;
    GeomShapeView::global_cell_connectivity[IT_Pentaedron6] = &m_pentaedron6_connectivity;
    GeomShapeView::global_cell_connectivity[IT_Hexaedron8] = &m_hexaedron8_connectivity;
    GeomShapeView::global_cell_connectivity[IT_Heptaedron10] = &m_heptaedron10_connectivity;
    GeomShapeView::global_cell_connectivity[IT_Octaedron12] = &m_octaedron12_connectivity;
  }

 public:
  NullConnectivity m_null_connectivity;
  VertexConnectivity m_vertex_connectivity;
  Line2Connectivity m_line2_connectivity;
  Triangle3Connectivity m_triangle3_connectivity;
  Quad4Connectivity m_quad4_connectivity;
  Pentagon5Connectivity m_pentagon5_connectivity;
  Hexagon6Connectivity m_hexagon6_connectivity;
  Tetraedron4Connectivity m_tetraedron4_connectivity;
  Pyramid5Connectivity m_pyramid5_connectivity;
  Pentaedron6Connectivity m_pentaedron6_connectivity;
  Hexaedron8Connectivity m_hexaedron8_connectivity;
  Heptaedron10Connectivity m_heptaedron10_connectivity;
  Octaedron12Connectivity m_octaedron12_connectivity;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void GeomShapeView::
initializeConnectivity()
{
  if (global_connectivity)
    return;

  global_connectivity = new GeomShapeConnectivity();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CellConnectivity* GeomShapeView::global_cell_connectivity[NB_BASIC_ITEM_TYPE];
GeomShapeConnectivity* GeomShapeView::global_connectivity = 0;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class GeomShapeStdBuilder
: public GeomShapeMutableView
{
 public:
  GeomShapeStdBuilder(const GeomShapeMutableView& gsv)
  : GeomShapeMutableView(gsv)
  {
  }
 public:
  void computeNodePositionFromHexa(HexaElementConstView hexa);
  void computeNodePositionFromQuad(QuadElementConstView quad);
 private:
  inline void
  _addFaceD(Integer fid,Integer id1,Integer id2,Integer id3,Integer id4)
  {
    setFace(fid,
             Real3( 0.25 * ( node(id1).x + node(id2).x + node(id3).x + node(id4).x ),
                    0.25 * ( node(id1).y + node(id2).y + node(id3).y + node(id4).y ),
                    0.25 * ( node(id1).z + node(id2).z + node(id3).z + node(id4).z )));
  }

  inline void
  _addFace2D(Integer fid,Integer id1,Integer id2)
  {
    setFace(fid,
            Real3( 0.5 * ( node(id1).x + node(id2).x ),
                   0.5 * ( node(id1).y + node(id2).y ),
                   0.0));
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void GeomShapeView::
initFromHexa(HexaElementConstView hexa,GeomShape& geom_cell)
{
  _setNullItem(IT_Hexaedron8);

  GeomShapeStdBuilder s(geom_cell.toMutableView());
  s.computeNodePositionFromHexa(hexa);

  geom_cell._setArray(*this);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void GeomShapeView::
initFromQuad(QuadElementConstView quad,GeomShape& geom_cell)
{
  _setNullItem(IT_Quad4);

  GeomShapeStdBuilder s(geom_cell.toMutableView());
  s.computeNodePositionFromQuad(quad);

  geom_cell._setArray(*this);
}

/*---------------------------------------------------------------------------*/
/*----- Définition des sous volumes de contrôle -----------------------------*/
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*--------------- En 2D -----------------------------------------------------*/

void GeomShapeView::
fillSubZoneElement(Quad4ElementView sub_zone, Integer id)
{
  const NodeConnectic & nc = nodeConnectic(id);

  sub_zone.init(node(nodeAssociation(id)),face(nc.face(0)), center(),face(nc.face(1)));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void GeomShapeView::
fillSubZoneQuad4(Quad4ElementView sub_zone,Integer i)
{
  switch(i){
   case 0: sub_zone.init(node(0),face( 0),center(),face(3)); break;
   case 1: sub_zone.init(node(1),face( 1),center(),face(0)); break;
   case 2: sub_zone.init(node(2),face( 2),center(),face(1)); break;
   case 3: sub_zone.init(node(3),face( 3),center(),face(2)); break;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void GeomShapeView::
fillSubZoneTriangle3(Quad4ElementView sub_zone,Integer i)
{
  switch(i){
  case 0: sub_zone.init(node(0),face(0),center(),node(0));break;
  case 1: sub_zone.init(node(1),face(1),center(),face(0));break;
  case 2: sub_zone.init(node(2),face(2),center(),face(1));break;
  case 3: sub_zone.init(node(0),node(0),center(),face(2));break;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void GeomShapeView::
fillSubZonePentagon5(QuadElementView,Integer)
{
  throw NotImplementedException(A_FUNCINFO);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void GeomShapeView::
fillSubZoneHexagon6(QuadElementView,Integer)
{
  throw NotImplementedException(A_FUNCINFO);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void GeomShapeView::
fillSubZoneElement(HexaElementView sub_zone,Integer i)
{
  const NodeConnectic & nc = nodeConnectic(i);

  sub_zone.init(node(nodeAssociation(i)),
                edge(nc.edge(0)),face(nc.face(0)),edge(nc.edge(1)),
                edge(nc.edge(2)),face(nc.face(2)),center(),face(nc.face(1)));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void GeomShapeView::
fillSubZoneHexaedron8(HexaElementView sub_zone,Integer i)
{
  switch(i){
   case 0: sub_zone.init(node(0),edge( 0),face(0),edge( 3),edge( 4),face(2),center(),face(1)); break;
   case 1: sub_zone.init(node(1),edge( 1),face(0),edge( 0),edge( 5),face(4),center(),face(2)); break;
   case 2: sub_zone.init(node(2),edge( 2),face(0),edge( 1),edge( 6),face(5),center(),face(4)); break;
   case 3: sub_zone.init(node(3),edge( 3),face(0),edge( 2),edge( 7),face(1),center(),face(5)); break;
   case 4: sub_zone.init(node(4),edge(11),face(3),edge( 8),edge( 4),face(1),center(),face(2)); break;
   case 5: sub_zone.init(node(5),edge( 8),face(3),edge( 9),edge( 5),face(2),center(),face(4)); break;
   case 6: sub_zone.init(node(6),edge( 9),face(3),edge(10),edge( 6),face(4),center(),face(5)); break;
   case 7: sub_zone.init(node(7),edge(10),face(3),edge(11),edge( 7),face(5),center(),face(1)); break;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void GeomShapeView::
fillSubZonePyramid5(HexaElementView sub_zone,Integer i)
{
  switch(i){
  case 0: sub_zone.init(node(0),edge(0),face(0),edge(3),edge(4),face(2),center(),face(1)); break;
  case 1: sub_zone.init(node(1),edge(1),face(0),edge(0),edge(5),face(3),center(),face(2)); break;
  case 2: sub_zone.init(node(2),edge(2),face(0),edge(1),edge(6),face(4),center(),face(3)); break;
  case 3: sub_zone.init(node(3),edge(3),face(0),edge(2),edge(7),face(1),center(),face(4)); break;
  case 4: sub_zone.init(node(4),node(4),node(4),node(4),edge(4),face(1),center(),face(2)); break;
  case 5: sub_zone.init(node(4),node(4),node(4),node(4),edge(5),face(2),center(),face(3)); break;
  case 6: sub_zone.init(node(4),node(4),node(4),node(4),edge(6),face(3),center(),face(4)); break;
  case 7: sub_zone.init(node(4),node(4),node(4),node(4),edge(7),face(4),center(),face(1)); break;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void GeomShapeView::
fillSubZonePentaedron6(HexaElementView sub_zone,Integer i)
{
  switch(i){
  case 0: sub_zone.init(node(0),edge(0),face(0),edge(2),edge( 3),face(2),center(),face(1)); break;
  case 1: sub_zone.init(node(1),edge(1),face(0),edge(0),edge( 4),face(4),center(),face(2)); break;
  case 2: sub_zone.init(node(2),edge(2),face(0),edge(1),edge( 5),face(1),center(),face(4)); break;
  case 3: sub_zone.init(node(3),edge(8),face(3),edge(6),edge( 3),face(1),center(),face(2)); break;
  case 4: sub_zone.init(node(4),edge(6),face(3),edge(7),edge( 4),face(2),center(),face(4)); break;
  case 5: sub_zone.init(node(5),edge(7),face(3),edge(8),edge( 5),face(4),center(),face(1)); break;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void GeomShapeView::
fillSubZoneTetraedron4(HexaElementView sub_zone,Integer i)
{
  switch(i){
  case 0: sub_zone.init(node(0),edge(0),face(0),edge( 2),edge(3),face(2),center(),face(1)); break;
  case 1: sub_zone.init(node(1),edge(1),face(0),edge( 0),edge(4),face(3),center(),face(2)); break;
  case 2: sub_zone.init(node(2),edge(2),face(0),edge( 1),edge(5),face(1),center(),face(3)); break;
  case 3: sub_zone.init(node(3),edge(3),face(1),edge( 5),edge(4),face(2),center(),face(3)); break;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void GeomShapeView::
fillSubZoneHeptaedron10(HexaElementView sub_zone,Integer i)
{
  switch(i){
  case 0: sub_zone.init(node(0),edge( 0),face(0),edge( 4),edge(10),face(2),center(),face(6)); break;
  case 1: sub_zone.init(node(1),edge( 1),face(0),edge( 0),edge(11),face(3),center(),face(2)); break;
  case 2: sub_zone.init(node(2),edge( 2),face(0),edge( 1),edge(12),face(4),center(),face(3)); break;
  case 3: sub_zone.init(node(3),edge( 3),face(0),edge( 2),edge(13),face(5),center(),face(4)); break;
  case 4: sub_zone.init(node(4),edge( 4),face(0),edge( 3),edge(14),face(6),center(),face(5)); break;
  case 5: sub_zone.init(node(5),edge( 9),face(1),edge( 5),edge(10),face(6),center(),face(2)); break;
  case 6: sub_zone.init(node(6),edge( 5),face(1),edge( 6),edge(11),face(2),center(),face(3)); break;
  case 7: sub_zone.init(node(7),edge( 6),face(1),edge( 7),edge(12),face(3),center(),face(4)); break;
  case 8: sub_zone.init(node(8),edge( 7),face(1),edge( 8),edge(13),face(4),center(),face(5)); break;
  case 9: sub_zone.init(node(9),edge( 8),face(1),edge( 9),edge(14),face(5),center(),face(6)); break;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void GeomShapeView::
fillSubZoneOctaedron12(HexaElementView sub_zone,Integer i)
{
  switch(i){
  case 0:  sub_zone.init(node( 0),edge( 0),face(0),edge( 5),edge(12),face(2),center(),face(7)); break;
  case 1:  sub_zone.init(node( 1),edge( 1),face(0),edge( 0),edge(13),face(3),center(),face(2)); break;
  case 2:  sub_zone.init(node( 2),edge( 2),face(0),edge( 1),edge(14),face(4),center(),face(3)); break;
  case 3:  sub_zone.init(node( 3),edge( 3),face(0),edge( 2),edge(15),face(5),center(),face(4)); break;
  case 4:  sub_zone.init(node( 4),edge( 4),face(0),edge( 3),edge(16),face(6),center(),face(5)); break;
  case 5:  sub_zone.init(node( 5),edge( 5),face(0),edge( 4),edge(17),face(7),center(),face(6)); break;
  case 6:  sub_zone.init(node( 6),edge(11),face(1),edge( 6),edge(12),face(7),center(),face(2)); break;
  case 7:  sub_zone.init(node( 7),edge( 6),face(1),edge( 7),edge(13),face(2),center(),face(3)); break;
  case 8:  sub_zone.init(node( 8),edge( 7),face(1),edge( 8),edge(14),face(3),center(),face(4)); break;
  case 9:  sub_zone.init(node( 9),edge( 8),face(1),edge( 9),edge(15),face(4),center(),face(5)); break;
  case 10: sub_zone.init(node(10),edge( 9),face(1),edge(10),edge(16),face(5),center(),face(6)); break;
  case 11: sub_zone.init(node(11),edge(10),face(1),edge(11),edge(17),face(6),center(),face(7)); break;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief En 3D, calcule les psoitions à partir de l'hexaèdre \a hexa.
 */
void GeomShapeStdBuilder::
computeNodePositionFromHexa(HexaElementConstView hexa)
{
  const Real3 nul_vector = Real3(0.,0.,0.);

  // Calcule la position du centre.
  Real3 c = nul_vector;

  for( Integer i = 0; i<8; ++i ){
    setNode(i,hexa[i]);
    c += node(i);
  }

  setCenter(0.125 * c);

  // Calcul la position des centres des faces.
  _addFaceD(  0 , 0 , 3 , 2 , 1 );
  _addFaceD(  1 , 0 , 4 , 7 , 3 );
  _addFaceD(  2 , 0 , 1 , 5 , 4 );
  _addFaceD(  3 , 4 , 5 , 6 , 7 );
  _addFaceD(  4 , 1 , 2 , 6 , 5 );
  _addFaceD(  5 , 2 , 3 , 7 , 6 );
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief En 2D, calcule les positions à partir du quad \a quad.
 */
void GeomShapeStdBuilder::
computeNodePositionFromQuad(QuadElementConstView quad)
{
  const Real3 nul_vector = Real3(0.,0.,0.);

  // Calcule la position du centre.
  Real3 c = nul_vector;

  for( Integer i = 0; i<4; ++i )
  {
    setNode(i,quad[i]);
    c += node(i);
  }

  setCenter(0.25 * c);

  // Calcul la position des centres des faces.
  _addFace2D(  0 , 0 , 1  );
  _addFace2D(  1 , 1 , 2  );
  _addFace2D(  2 , 2 , 3  );
  _addFace2D(  3 , 3 , 0  );
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Hexaedron8ShapeView GeomShape::
initFromHexaedron8(Hexaedron8ElementConstView hexa)
{
  Hexaedron8ShapeView view;

  view._setNullItem(IT_Hexaedron8);

  GeomShapeStdBuilder s(toMutableView());
  s.computeNodePositionFromHexa(hexa);

  _setArray(view);

  //view.initFromHexa(hexa,*this);
  return view;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Quad4ShapeView GeomShape::
initFromQuad4(Quad4ElementConstView quad)
{
  Quad4ShapeView view;

  view._setNullItem(IT_Quad4);

  GeomShapeStdBuilder s(toMutableView());
  s.computeNodePositionFromQuad(quad);

  _setArray(view);
  
  return view;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

GEOMETRIC_END_NAMESPACE
ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
