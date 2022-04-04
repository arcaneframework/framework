// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CellConnectivity.cc                                         (C) 2000-2014 */
/*                                                                           */
/* Informations sur la connectivité d'une maille.                            */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//#include "arcane/geometric/IGeometric.h"
#include "arcane/geometric/CellConnectivity.h"
//#include "arcane/geometric/CellGeom.h"
//#include "arcane/geometric/CellGeomList.h"

#include "arcane/AbstractItemOperationByBasicType.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE
GEOMETRIC_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline void CellConnectivity::
_setEdgeDirectConnectic()
{
  if (m_edge_connectic){
    // Cas 3D
    for( Integer i=0, n=nbEdge(); i<n; ++i ){
      m_edge_direct_connectic[(i*2)] = m_edge_connectic[i].node(0);
      m_edge_direct_connectic[(i*2)+1] = m_edge_connectic[i].node(1);
    }
  }
  else{
    // Cas 2D.
    // Dans ce cas les arêtes sont numérotés en fonction des noeuds.
    for( Integer i=0, n=nbEdge(); i<n; ++i ){
      m_edge_direct_connectic[(i*2)] = i;
      m_edge_direct_connectic[(i*2)+1] = (i+1) % n;
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void NullConnectivity::
_init()
{
  _setEdgeDirectConnectic();
}

void VertexConnectivity::
_init()
{
  _setEdgeDirectConnectic();
}

void Line2Connectivity::
_init()
{
  _setEdgeDirectConnectic();
}

void Pentagon5Connectivity::
_init()
{
  _setEdgeDirectConnectic();
  // TODO
}

void Hexagon6Connectivity::
_init()
{
  _setEdgeDirectConnectic();
  // TODO
}

void Hexaedron8Connectivity::
_init()
{
  using namespace Arcane::geometric;

  m_nb_sub_zone = 8;
  m_node_connectic = hexa_node_connectic;
  m_edge_connectic = hexa_edge_connectic;
  m_face_connectic = hexa_face_connectic;
  m_node_association = hexa_node_association;

  m_svc_face_connectic = hexa_svc_face_connectic;
  m_nb_svc_face = sizeof(hexa_svc_face_connectic) / sizeof(hexa_svc_face_connectic[0]);

  m_edge_node_sub_zone_id[0] = 1;
  m_edge_node_sub_zone_id[1] = 3;
  m_edge_node_sub_zone_id[2] = 4;
  m_face_node_sub_zone_id[0] = 2;
  m_face_node_sub_zone_id[1] = 7;
  m_face_node_sub_zone_id[2] = 5;

  _setEdgeDirectConnectic();
}

void Pyramid5Connectivity::
_init()
{
  using namespace Arcane::geometric;

  m_nb_sub_zone = 8;
  m_node_connectic = pyra_node_connectic;
  m_edge_connectic = pyra_edge_connectic;
  m_face_connectic = pyra_face_connectic;
  m_node_association = pyra_node_association;
  m_svc_face_connectic = pyra_svc_face_connectic;
  m_nb_svc_face = sizeof(pyra_svc_face_connectic) / sizeof(pyra_svc_face_connectic[0]);
  m_edge_node_sub_zone_id[0] = 1;
  m_edge_node_sub_zone_id[1] = 3;
  m_edge_node_sub_zone_id[2] = 4;
  m_face_node_sub_zone_id[0] = 2;
  m_face_node_sub_zone_id[1] = 7;
  m_face_node_sub_zone_id[2] = 5;

  _setEdgeDirectConnectic();
}

void Pentaedron6Connectivity::
_init()
{
  using namespace Arcane::geometric;

  m_nb_sub_zone = 6;
  m_node_connectic = penta_node_connectic;
  m_edge_connectic = penta_edge_connectic;
  m_face_connectic = penta_face_connectic;
  m_node_association = penta_node_association;
  m_svc_face_connectic = penta_svc_face_connectic;
  m_nb_svc_face = sizeof(penta_svc_face_connectic) / sizeof(penta_svc_face_connectic[0]);
  m_edge_node_sub_zone_id[0] = 1;
  m_edge_node_sub_zone_id[1] = 3;
  m_edge_node_sub_zone_id[2] = 4;
  m_face_node_sub_zone_id[0] = 2;
  m_face_node_sub_zone_id[1] = 7;
  m_face_node_sub_zone_id[2] = 5;

  _setEdgeDirectConnectic();
}

void Tetraedron4Connectivity::
_init()
{
  using namespace Arcane::geometric;

  m_nb_sub_zone = 4;
  m_node_connectic = tetra_node_connectic;
  m_edge_connectic = tetra_edge_connectic;
  m_face_connectic = tetra_face_connectic;
  m_node_association = tetra_node_association;
  
  m_svc_face_connectic = tetra_svc_face_connectic;
  m_nb_svc_face = sizeof(tetra_svc_face_connectic) /sizeof(tetra_svc_face_connectic[0]);
  m_edge_node_sub_zone_id[0] = 1;
  m_edge_node_sub_zone_id[1] = 3;
  m_edge_node_sub_zone_id[2] = 4;
  m_face_node_sub_zone_id[0] = 2;
  m_face_node_sub_zone_id[1] = 7;
  m_face_node_sub_zone_id[2] = 5;

  _setEdgeDirectConnectic();
}

void Heptaedron10Connectivity::
_init()
{
  using namespace Arcane::geometric;

  m_nb_sub_zone = 10;
  m_node_connectic = wedge7_node_connectic;
  m_edge_connectic = wedge7_edge_connectic;
  m_face_connectic = wedge7_face_connectic;
  m_node_association = wedge7_node_association;

  m_svc_face_connectic = wedge7_svc_face_connectic;
  m_nb_svc_face = sizeof(wedge7_svc_face_connectic) / sizeof(wedge7_svc_face_connectic[0]);
  m_edge_node_sub_zone_id[0] = 1;
  m_edge_node_sub_zone_id[1] = 3;
  m_edge_node_sub_zone_id[2] = 4;
  m_face_node_sub_zone_id[0] = 2;
  m_face_node_sub_zone_id[1] = 7;
  m_face_node_sub_zone_id[2] = 5;

  _setEdgeDirectConnectic();
}

void Octaedron12Connectivity::
_init()
{
  using namespace Arcane::geometric;

  m_nb_sub_zone = 12;
  m_node_connectic = wedge8_node_connectic;
  m_edge_connectic = wedge8_edge_connectic;
  m_face_connectic = wedge8_face_connectic;
  m_node_association = wedge8_node_association;
      
  m_svc_face_connectic = wedge8_svc_face_connectic;
  m_nb_svc_face = sizeof(wedge8_svc_face_connectic) / sizeof(wedge8_svc_face_connectic[0]);
  m_edge_node_sub_zone_id[0] = 1;
  m_edge_node_sub_zone_id[1] = 3;
  m_edge_node_sub_zone_id[2] = 4;
  m_face_node_sub_zone_id[0] = 2;
  m_face_node_sub_zone_id[1] = 7;
  m_face_node_sub_zone_id[2] = 5;

  _setEdgeDirectConnectic();
}

void Quad4Connectivity::
_init()
{
  using namespace Arcane::geometric;

  m_nb_sub_zone = 4;
  m_node_connectic = quad_node_connectic;
  m_edge_connectic = 0;
  m_face_connectic = quad_face_connectic;
  m_node_association = quad_node_association;
      
  m_svc_face_connectic = quad_svc_face_connectic;
  m_nb_svc_face = sizeof(quad_svc_face_connectic) /sizeof(quad_svc_face_connectic[0]);
  m_edge_node_sub_zone_id[0] = 0;
  m_edge_node_sub_zone_id[1] = 0;
  m_edge_node_sub_zone_id[2] = 0;
  m_face_node_sub_zone_id[0] = 1;
  m_face_node_sub_zone_id[1] = 3;
  m_face_node_sub_zone_id[2] = 0;

  _setEdgeDirectConnectic();
}

void Triangle3Connectivity::
_init()
{
  using namespace Arcane::geometric;

  m_nb_sub_zone = 4;
  m_node_connectic = triangle_node_connectic;
  m_edge_connectic = 0;
  m_face_connectic = triangle_face_connectic;
  m_node_association = triangle_node_association;
      
  m_svc_face_connectic = triangle_svc_face_connectic;
  m_nb_svc_face = sizeof(triangle_svc_face_connectic)/sizeof(triangle_svc_face_connectic[0]);
  m_edge_node_sub_zone_id[0] = 0;
  m_edge_node_sub_zone_id[1] = 0;
  m_edge_node_sub_zone_id[2] = 0;
  m_face_node_sub_zone_id[0] = 1;
  m_face_node_sub_zone_id[1] = 3;
  m_face_node_sub_zone_id[2] = 0;

  _setEdgeDirectConnectic();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

GEOMETRIC_END_NAMESPACE
ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
