// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VtkCellTypes.h                                              (C) 2000-2023 */
/*                                                                           */
/* Définitions des types de maille de VTK.                                   */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_STD_INTERNAL_VTKCELLTYPES_H
#define ARCANE_STD_INTERNAL_VTKCELLTYPES_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcaneGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::VtkUtils
{

// Les valeurs de 'VTK_*' sont issues des sources de VTK 9.2.
// Elles sont définies dans le fichier 'Common/DataModel/vtkCellType'.

// Linear cells
const int VTK_EMPTY_CELL = 0;
const int VTK_VERTEX = 1;
const int VTK_POLY_VERTEX = 2;
const int VTK_LINE = 3;
const int VTK_POLY_LINE = 4;
const int VTK_TRIANGLE = 5;
const int VTK_TRIANGLE_STRIP = 6;
const int VTK_POLYGON = 7;
const int VTK_PIXEL = 8;
const int VTK_QUAD = 9;
const int VTK_TETRA = 10;
const int VTK_VOXEL = 11;
const int VTK_HEXAHEDRON = 12;
const int VTK_WEDGE = 13;
const int VTK_PYRAMID = 14;
const int VTK_PENTAGONAL_PRISM = 15;
const int VTK_HEXAGONAL_PRISM = 16;

// Quadratic, isoparametric cells
const int VTK_QUADRATIC_EDGE = 21;
const int VTK_QUADRATIC_TRIANGLE = 22;
const int VTK_QUADRATIC_QUAD = 23;
const int VTK_QUADRATIC_POLYGON = 36;
const int VTK_QUADRATIC_TETRA = 24;
const int VTK_QUADRATIC_HEXAHEDRON = 25;
const int VTK_QUADRATIC_WEDGE = 26;
const int VTK_QUADRATIC_PYRAMID = 27;
const int VTK_BIQUADRATIC_QUAD = 28;
const int VTK_TRIQUADRATIC_HEXAHEDRON = 29;
const int VTK_TRIQUADRATIC_PYRAMID = 37;
const int VTK_QUADRATIC_LINEAR_QUAD = 30;
const int VTK_QUADRATIC_LINEAR_WEDGE = 31;
const int VTK_BIQUADRATIC_QUADRATIC_WEDGE = 32;
const int VTK_BIQUADRATIC_QUADRATIC_HEXAHEDRON = 33;
const int VTK_BIQUADRATIC_TRIANGLE = 34;

// Cubic, isoparametric cell
const int VTK_CUBIC_LINE = 35;

// Special class of cells formed by convex group of points
const int VTK_CONVEX_POINT_SET = 41;

// Polyhedron cell (consisting of polygonal faces)
const int VTK_POLYHEDRON = 42;

// Higher order cells in parametric form
const int VTK_PARAMETRIC_CURVE = 51;
const int VTK_PARAMETRIC_SURFACE = 52;
const int VTK_PARAMETRIC_TRI_SURFACE = 53;
const int VTK_PARAMETRIC_QUAD_SURFACE = 54;
const int VTK_PARAMETRIC_TETRA_REGION = 55;
const int VTK_PARAMETRIC_HEX_REGION = 56;

// Higher order cells
const int VTK_HIGHER_ORDER_EDGE = 60;
const int VTK_HIGHER_ORDER_TRIANGLE = 61;
const int VTK_HIGHER_ORDER_QUAD = 62;
const int VTK_HIGHER_ORDER_POLYGON = 63;
const int VTK_HIGHER_ORDER_TETRAHEDRON = 64;
const int VTK_HIGHER_ORDER_WEDGE = 65;
const int VTK_HIGHER_ORDER_PYRAMID = 66;
const int VTK_HIGHER_ORDER_HEXAHEDRON = 67;

// Arbitrary order Lagrange elements (formulated separated from generic higher order cells)
const int VTK_LAGRANGE_CURVE = 68;
const int VTK_LAGRANGE_TRIANGLE = 69;
const int VTK_LAGRANGE_QUADRILATERAL = 70;
const int VTK_LAGRANGE_TETRAHEDRON = 71;
const int VTK_LAGRANGE_HEXAHEDRON = 72;
const int VTK_LAGRANGE_WEDGE = 73;
const int VTK_LAGRANGE_PYRAMID = 74;

// Arbitrary order Bezier elements (formulated separated from generic higher order cells)
const int VTK_BEZIER_CURVE = 75;
const int VTK_BEZIER_TRIANGLE = 76;
const int VTK_BEZIER_QUADRILATERAL = 77;
const int VTK_BEZIER_TETRAHEDRON = 78;
const int VTK_BEZIER_HEXAHEDRON = 79;
const int VTK_BEZIER_WEDGE = 80;
const int VTK_BEZIER_PYRAMID = 81;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::VtkUtils

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
