// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VtkCellTypes.h                                              (C) 2000-2025 */
/*                                                                           */
/* Définitions des types de maille de VTK.                                   */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_INTERNAL_VTKCELLTYPES_H
#define ARCANE_CORE_INTERNAL_VTKCELLTYPES_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class ItemTypeInfo;
}

namespace Arcane::VtkUtils
{

// Les valeurs de 'VTK_*' sont issues des sources de VTK 9.2.
// Elles sont définies dans le fichier 'Common/DataModel/vtkCellType'.

// Linear cells
const unsigned char VTK_EMPTY_CELL = 0;
const unsigned char VTK_VERTEX = 1;
const unsigned char VTK_POLY_VERTEX = 2;
const unsigned char VTK_LINE = 3;
const unsigned char VTK_POLY_LINE = 4;
const unsigned char VTK_TRIANGLE = 5;
const unsigned char VTK_TRIANGLE_STRIP = 6;
const unsigned char VTK_POLYGON = 7;
const unsigned char VTK_PIXEL = 8;
const unsigned char VTK_QUAD = 9;
const unsigned char VTK_TETRA = 10;
const unsigned char VTK_VOXEL = 11;
const unsigned char VTK_HEXAHEDRON = 12;
const unsigned char VTK_WEDGE = 13;
const unsigned char VTK_PYRAMID = 14;
const unsigned char VTK_PENTAGONAL_PRISM = 15;
const unsigned char VTK_HEXAGONAL_PRISM = 16;

// Quadratic, isoparametric cells
const unsigned char VTK_QUADRATIC_EDGE = 21;
const unsigned char VTK_QUADRATIC_TRIANGLE = 22;
const unsigned char VTK_QUADRATIC_QUAD = 23;
const unsigned char VTK_QUADRATIC_POLYGON = 36;
const unsigned char VTK_QUADRATIC_TETRA = 24;
const unsigned char VTK_QUADRATIC_HEXAHEDRON = 25;
const unsigned char VTK_QUADRATIC_WEDGE = 26;
const unsigned char VTK_QUADRATIC_PYRAMID = 27;
const unsigned char VTK_BIQUADRATIC_QUAD = 28;
const unsigned char VTK_TRIQUADRATIC_HEXAHEDRON = 29;
const unsigned char VTK_TRIQUADRATIC_PYRAMID = 37;
const unsigned char VTK_QUADRATIC_LINEAR_QUAD = 30;
const unsigned char VTK_QUADRATIC_LINEAR_WEDGE = 31;
const unsigned char VTK_BIQUADRATIC_QUADRATIC_WEDGE = 32;
const unsigned char VTK_BIQUADRATIC_QUADRATIC_HEXAHEDRON = 33;
const unsigned char VTK_BIQUADRATIC_TRIANGLE = 34;

// Cubic, isoparametric cell
const unsigned char VTK_CUBIC_LINE = 35;

// Special class of cells formed by convex group of points
const unsigned char VTK_CONVEX_POINT_SET = 41;

// Polyhedron cell (consisting of polygonal faces)
const unsigned char VTK_POLYHEDRON = 42;

// Higher order cells in parametric form
const unsigned char VTK_PARAMETRIC_CURVE = 51;
const unsigned char VTK_PARAMETRIC_SURFACE = 52;
const unsigned char VTK_PARAMETRIC_TRI_SURFACE = 53;
const unsigned char VTK_PARAMETRIC_QUAD_SURFACE = 54;
const unsigned char VTK_PARAMETRIC_TETRA_REGION = 55;
const unsigned char VTK_PARAMETRIC_HEX_REGION = 56;

// Higher order cells
const unsigned char VTK_HIGHER_ORDER_EDGE = 60;
const unsigned char VTK_HIGHER_ORDER_TRIANGLE = 61;
const unsigned char VTK_HIGHER_ORDER_QUAD = 62;
const unsigned char VTK_HIGHER_ORDER_POLYGON = 63;
const unsigned char VTK_HIGHER_ORDER_TETRAHEDRON = 64;
const unsigned char VTK_HIGHER_ORDER_WEDGE = 65;
const unsigned char VTK_HIGHER_ORDER_PYRAMID = 66;
const unsigned char VTK_HIGHER_ORDER_HEXAHEDRON = 67;

// Arbitrary order Lagrange elements (formulated separated from generic higher order cells)
const unsigned char VTK_LAGRANGE_CURVE = 68;
const unsigned char VTK_LAGRANGE_TRIANGLE = 69;
const unsigned char VTK_LAGRANGE_QUADRILATERAL = 70;
const unsigned char VTK_LAGRANGE_TETRAHEDRON = 71;
const unsigned char VTK_LAGRANGE_HEXAHEDRON = 72;
const unsigned char VTK_LAGRANGE_WEDGE = 73;
const unsigned char VTK_LAGRANGE_PYRAMID = 74;

// Arbitrary order Bezier elements (formulated separated from generic higher order cells)
const unsigned char VTK_BEZIER_CURVE = 75;
const unsigned char VTK_BEZIER_TRIANGLE = 76;
const unsigned char VTK_BEZIER_QUADRILATERAL = 77;
const unsigned char VTK_BEZIER_TETRAHEDRON = 78;
const unsigned char VTK_BEZIER_HEXAHEDRON = 79;
const unsigned char VTK_BEZIER_WEDGE = 80;
const unsigned char VTK_BEZIER_PYRAMID = 81;

// Invalid value to detect unsupported Arcane type
const unsigned char VTK_BAD_ARCANE_TYPE = 255;

extern "C++" ARCANE_CORE_EXPORT Int16
vtkToArcaneCellType(int vtk_type, Int32 nb_node);

extern "C++" ARCANE_CORE_EXPORT unsigned char
arcaneToVtkCellType(Int16 arcane_type);

extern "C++" ARCANE_CORE_EXPORT unsigned char
arcaneToVtkCellType(const ItemTypeInfo* arcane_type);

// Les valeurs pour les types 'CellGhostTypes' et 'PointGhostTypes' sont définies
// dans le fichier vtkDataSetAttributes.h.
enum CellGhostTypes
{
  DUPLICATECELL = 1, // the cell is present on multiple processors
  HIGHCONNECTIVITYCELL = 2, // the cell has more neighbors than in a regular mesh
  LOWCONNECTIVITYCELL = 4, // the cell has less neighbors than in a regular mesh
  REFINEDCELL = 8, // other cells are present that refines it.
  EXTERIORCELL = 16, // the cell is on the exterior of the data set
  HIDDENCELL = 32 // the cell is needed to maintain connectivity, but the data values should be ignored.
};

enum PointGhostTypes
{
  DUPLICATEPOINT = 1, // the point is present on multiple processors
  HIDDENPOINT = 2 // the point is needed to maintain connectivity, but the data values should be ignored.
};


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::VtkUtils

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
