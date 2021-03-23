// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#ifndef _XDMF_DEFINES_H_
#define _XDMF_DEFINES_H_


#define XDMF_MAX_ORDER  10

// General Uniform Organization
#define XDMF_STRUCTURED     0
#define XDMF_UNSTRUCTURED   1

// Topologies
#define XDMF_NOTOPOLOGY     0x0
#define XDMF_POLYVERTEX     0x1		  // A Group of Points (Atoms)
#define XDMF_POLYLINE       0x2		  // Line Segments (Bonds)
#define XDMF_POLYGON        0x3		  // N Sided
#define XDMF_TRI            0x4		  // 3 Edge Polygon
#define XDMF_QUAD           0x5		  // 4 Edge Polygon
#define XDMF_TET            0x6		  // 4 Triangular Faces
#define XDMF_PYRAMID        0x7		  // 4 Triangles, QUADRILATERAL Base
#define XDMF_WEDGE          0x8		  // 2 Trianges, 2 QUADRILATERAL and QUADRILATERAL Base
#define XDMF_HEX            0x9		  // 6 QUADRILATERAL Faces
#define XDMF_EDGE_3         0x0022	  // 3 Node High Order Line
#define XDMF_TRI_6          0x0024	  // 6 Node High Order Triangle
#define XDMF_QUAD_8         0x0025	  // 8 Node High Order Quadrilateral
#define XDMF_TET_10         0x0026	  // 10 Node High Order Tetrahedron
#define XDMF_PYRAMID_13     0x0027	  // 13 Node High Order Pyramid
#define XDMF_WEDGE_15       0x0028	  // 15 Node High Order Wedge
#define XDMF_HEX_20         0x0029	  // 20 Node High Order Hexahedron

#define XDMF_MIXED          0x0070	  // A Mixture of Unstructured Base Topologies
#define XDMF_2DSMESH        0x0100	  // General ( Curved )
#define XDMF_2DRECTMESH     0x0101	  // Rectilinear
#define XDMF_2DCORECTMESH   0x0102	  // Co-Rectilinear
#define XDMF_3DSMESH        0x1100	  // Curvelinear Mesh
#define XDMF_3DRECTMESH     0x1101	  // VectorX, VectorY, VectorZ
#define XDMF_3DCORECTMESH   0x1102	  // Origin Dx, Dy, Dz 

#define XDMF_GEOMETRY_NONE          0
#define XDMF_GEOMETRY_XYZ           1
#define XDMF_GEOMETRY_XY            2
#define XDMF_GEOMETRY_X_Y_Z         3
#define XDMF_GEOMETRY_X_Y           4
#define XDMF_GEOMETRY_VXVYVZ        5
#define XDMF_GEOMETRY_ORIGIN_DXDYDZ 6


#define XDMF_SUCCESS  1
#define XDMF_FAIL  -1

#define XDMF_TRUE  1
#define XDMF_FALSE  0

#define XDMF_MAX_DIMENSION  10
#define XDMF_MAX_STRING_LENGTH  1024


#define  XDMF_UNKNOWN_TYPE   -1
#define  XDMF_INT8_TYPE 	  1
#define  XDMF_INT16_TYPE	  6
#define  XDMF_INT32_TYPE	  2
#define  XDMF_INT64_TYPE	  3
#define  XDMF_FLOAT32_TYPE   4
#define  XDMF_FLOAT64_TYPE   5
#define  XDMF_UINT8_TYPE	  7
#define  XDMF_UINT16_TYPE    8
#define  XDMF_UINT32_TYPE    9
#define	XDMF_COMPOUND_TYPE  0x10

#endif
