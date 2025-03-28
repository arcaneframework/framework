// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VtkCellTypes.cc                                             (C) 2000-2025 */
/*                                                                           */
/* Définitions des types de maille de VTK.                                   */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/internal/VtkCellTypes.h"

#include "arcane/utils/IOException.h"
#include "arcane/utils/FatalErrorException.h"

#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int16 VtkUtils::
vtkToArcaneCellType(int vtk_type, Int32 nb_node)
{
  switch (vtk_type) {
  case VTK_EMPTY_CELL:
    return IT_NullType;
  case VTK_VERTEX:
    return IT_Vertex;
  case VTK_LINE:
    return IT_Line2;
  case VTK_QUADRATIC_EDGE:
    return IT_Line3;
  case VTK_TRIANGLE:
    return IT_Triangle3;
  case VTK_QUAD:
    return IT_Quad4;
  case VTK_QUADRATIC_QUAD:
    return IT_Quad8;
  case VTK_POLYGON: // VTK_POLYGON (a tester...)
    if (nb_node == 5)
      return IT_Pentagon5;
    if (nb_node == 6)
      return IT_Hexagon6;
    ARCANE_THROW(IOException, "Unsupported VtkCellType VTK_POLYGON with nb_node={0}", nb_node);
  case VTK_TETRA:
    return IT_Tetraedron4;
  case VTK_QUADRATIC_TETRA:
    return IT_Tetraedron10;
  case VTK_PYRAMID:
    return IT_Pyramid5;
  case VTK_WEDGE:
    return IT_Pentaedron6;
  case VTK_HEXAHEDRON:
    return IT_Hexaedron8;
  case VTK_QUADRATIC_HEXAHEDRON:
    return IT_Hexaedron20;
  case VTK_PENTAGONAL_PRISM:
    return IT_Heptaedron10;
  case VTK_HEXAGONAL_PRISM:
    return IT_Octaedron12;
    // NOTE GG: les types suivants ne sont pas bon pour VTK.
    //case 27: it = IT_Enneedron14; break; //
    //case 28: it = IT_Decaedron16; break; // VTK_HEXAGONAL_PRISM
    //case 29: it = IT_Heptagon7; break; // VTK_HEPTAGON
    //case 30: it = IT_Octogon8; break; // VTK_OCTAGON
  default:
    ARCANE_THROW(IOException, "Unsupported VtkCellType '{0}'", vtk_type);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

unsigned char VtkUtils::
arcaneToVtkCellType(Int16 arcane_type)
{
  switch (arcane_type) {
  case IT_NullType:
    return VTK_EMPTY_CELL;
  case IT_Vertex:
  case IT_FaceVertex:
    return VTK_VERTEX;
  case IT_Line2:
  case IT_CellLine2:
  case IT_Cell3D_Line2:
    return VTK_LINE;
  case IT_Line3:
    return VTK_QUADRATIC_EDGE;
  case IT_Triangle3:
  case IT_Cell3D_Triangle3:
    return VTK_TRIANGLE;
  case IT_Triangle6:
    return VTK_QUADRATIC_TRIANGLE;
  case IT_Quad4:
  case IT_Cell3D_Quad4:
    return VTK_QUAD;
  case IT_Quad8:
    return VTK_QUADRATIC_QUAD;
  case IT_Pentagon5:
    return VTK_POLYGON;
    // VTK_POLYGON (a tester...)
  case IT_Hexagon6:
    return VTK_POLYGON;
    // VTK_POLYGON (a tester ...)
  case IT_Tetraedron4:
    return VTK_TETRA;
  case IT_Tetraedron10:
    return VTK_QUADRATIC_TETRA;
  case IT_Pyramid5:
    return VTK_PYRAMID;
  case IT_Pentaedron6:
    return VTK_WEDGE;
  case IT_Hexaedron8:
    return VTK_HEXAHEDRON;
  case IT_Hexaedron20:
    return VTK_QUADRATIC_HEXAHEDRON;
  case IT_Heptaedron10:
    return VTK_PENTAGONAL_PRISM;
  case IT_Octaedron12:
    return VTK_HEXAGONAL_PRISM;
  default:
    ARCANE_FATAL("Unsuported item type for VtkWriter type={0}", arcane_type);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::VtkUtils

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
