// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#ifndef _VTK_DEFINES_H_
#define _VTK_DEFINES_H_

#define VTK_EMPTY_CELL								0
#define VTK_VERTEX 									1
#define VTK_POLY_VERTEX 							2 	// set of 3D vertices
#define VTK_LINE 										3
#define VTK_POLY_LINE 								4	// a set of 1D lines
#define VTK_TRIANGLE 								5 

// 2D triangle strip
// A triangle strip is a compact representation of triangles connected edge to edge in strip fashion
#define VTK_TRIANGLE_STRIP 						6
#define VTK_POLYGON 									7
#define VTK_PIXEL 									8	// 2D orthogonal quadrilateral
#define VTK_QUAD 										9	// 2D quadrilateral
#define VTK_TETRA 									10	// 3D tetrahedron
#define VTK_VOXEL 									11	// 3D orthogonal parallelepiped
#define VTK_HEXAHEDRON 								12	// 3D hexahedron

// A wedge consists of two triangular and three quadrilateral faces
// and is defined by the six points.
#define VTK_WEDGE 									13	// 3D linear wedge

//A pyramid consists of a rectangular base with four triangular faces.
#define VTK_PYRAMID 									14	// 3D linear pyramid

#define VTK_PENTAGONAL_PRISM 						15	// 3D prism with pentagonal base (10 points)
#define VTK_HEXAGONAL_PRISM 						16	// 3D prism with hexagonal base (12 points)

#define VTK_QUADRATIC_EDGE 						21
#define VTK_QUADRATIC_TRIANGLE 					22
#define VTK_QUADRATIC_QUAD 						23
#define VTK_QUADRATIC_TETRA 						24
#define VTK_QUADRATIC_HEXAHEDRON 				25
#define VTK_QUADRATIC_WEDGE 						26
#define VTK_QUADRATIC_PYRAMID 					27
#define VTK_BIQUADRATIC_QUAD 						28
#define VTK_TRIQUADRATIC_HEXAHEDRON 			29
#define VTK_QUADRATIC_LINEAR_QUAD 				30
#define VTK_QUADRATIC_LINEAR_WEDGE 				31
#define VTK_BIQUADRATIC_QUADRATIC_WEDGE 		32
#define VTK_BIQUADRATIC_QUADRATIC_HEXAHEDRON 33


#endif
