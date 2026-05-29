// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* GeomType.h                                                  (C) 2000-2026 */
/*                                                                           */
/* Enumeration specifying the type of polygon or polyhedron.                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_GEOMETRIC_GEOMTYPE_H
#define ARCANE_GEOMETRIC_GEOMTYPE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/geometry/GeometricGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::geometric
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Doxygen does not support (in 1.8.7) documentation for enum classes.
// We therefore use a fake class to maintain the same concept

/*!
 * \ingroup ArcaneGeometric
 * \brief Enumeration specifying the type of polygon or polyhedron associated
 * with a geometric element or shape.
 *
 * NOTE: The values of this enumeration must correspond to 
 * those of IT_* defined in ArcaneTypes.h
 */

#ifdef DOXYGEN_DOC
class GeomType
{
 public:

  enum{
    //! Null element
    NullType = IT_NullType,
    //! Vertex
    Vertex = IT_Vertex,
    //! Line
    Line2 = IT_Line2,
    //! Triangle
    Triangle3 = IT_Triangle3,
    //! Quadrangle
    Quad4 = IT_Quad4,
    //! Pentagon
    Pentagon5 = IT_Pentagon5,
    //! Hexagon
    Hexagon6 = IT_Hexagon6,
    //! Tetrahedron
    Tetraedron4 = IT_Tetraedron4,
    //! Pyramid
    Pyramid5 = IT_Pyramid5,
    //! Prism
    Pentaedron6 = IT_Pentaedron6,
    //! Hexahedron
    Hexaedron8 = IT_Hexaedron8,
    //! Pentagonal prism
    Heptaedron10 = IT_Heptaedron10,
    //! Hexagonal prism
    Octaedron12 = IT_Octaedron12
  }
};
#else
enum class GeomType
{
  //! Null element
  NullType = IT_NullType,
  //! Vertex
  Vertex = IT_Vertex,
  //! Line
  Line2 = IT_Line2,
  //! Triangle
  Triangle3 = IT_Triangle3,
  //! Quadrangle
  Quad4 = IT_Quad4,
  //! Pentagon
  Pentagon5 = IT_Pentagon5,
  //! Hexagon
  Hexagon6 = IT_Hexagon6,
  //! Tetrahedron
  Tetraedron4 = IT_Tetraedron4,
  //! Pyramid
  Pyramid5 = IT_Pyramid5,
  //! Prism
  Pentaedron6 = IT_Pentaedron6,
  //! Hexahedron
  Hexaedron8 = IT_Hexaedron8,
  //! Pentagonal prism
  Heptaedron10 = IT_Heptaedron10,
  //! Hexagonal prism
  Octaedron12 = IT_Octaedron12
};
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifdef ARCANE_CHECK
#define ARCANE_GEOMETRIC_CHECKTYPE(a,b) ::Arcane::geometric::_arcaneCheckType(a,b);
#else
#define ARCANE_GEOMETRIC_CHECKTYPE(a,b)
#endif

extern "C++" void
_arcaneBadType(GeomType type,GeomType wanted_type);

inline void
_arcaneCheckType(GeomType type,GeomType wanted_type)
{
  if (type!=wanted_type)
    _arcaneBadType(type,wanted_type);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::geometric

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
