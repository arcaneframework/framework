// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* GeometricGlobal.h                                           (C) 2000-2016 */
/*                                                                           */
/* Déclarations globales pour la composante géométrique.                     */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_GEOMETRIC_GEOMETRICGLOBAL_H
#define ARCANE_GEOMETRIC_GEOMETRICGLOBAL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/ArcaneTypes.h"

#define GEOMETRIC_BEGIN_NAMESPACE  namespace geometric {
#define GEOMETRIC_END_NAMESPACE    }

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifdef ARCANE_COMPONENT_arcane_cea_geometric
#define ARCANE_CEA_GEOMETRIC_EXPORT ARCANE_EXPORT
#else
#define ARCANE_CEA_GEOMETRIC_EXPORT ARCANE_IMPORT
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE
GEOMETRIC_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class GeomShapeView;
class GeomShape;
class GeomShapeMng;

class Triangle3ShapeView;
class Quad4ShapeView;
class Pentagon5ShapeView;
class Hexagon6ShapeView;
class Tetraedron4ShapeView;
class Pyramid5ShapeView;
class Pentaedron6ShapeView;
class Hexaedron8ShapeView;
class Heptaedron10ShapeView;
class Octaedron12ShapeView;

class Triangle3Element;
class Quad4Element;
class Pentagon5Element;
class Hexagon6Element;
class Tetraedron4Element;
class Pyramid5Element;
class Pentaedron6Element;
class Hexaedron8Element;
class Heptaedron10Element;
class Octaedron12Element;

class Triangle3ElementView;
class Quad4ElementView;
class Pentagon5ElementView;
class Hexagon6ElementView;
class Tetraedron4ElementView;
class Pyramid5ElementView;
class Pentaedron6ElementView;
class Hexaedron8ElementView;
class Heptaedron10ElementView;
class Octaedron12ElementView;

class Triangle3ElementConstView;
class Quad4ElementConstView;
class Pentagon5ElementConstView;
class Hexagon6ElementConstView;
class Tetraedron4ElementConstView;
class Pyramid5ElementConstView;
class Pentaedron6ElementConstView;
class Hexaedron8ElementConstView;
class Heptaedron10ElementConstView;
class Octaedron12ElementConstView;

typedef Triangle3Element TriangleElement;
typedef Quad4Element QuadElement;
typedef Pentagon5Element PentagonElement;
typedef Hexagon6Element HexagonElement;
typedef Tetraedron4Element TetraElement;
typedef Pyramid5Element PyramidElement;
typedef Pentaedron6Element PentaElement;
typedef Hexaedron8Element HexaElement;
typedef Heptaedron10Element Wedge7Element;
typedef Octaedron12Element Wedge8Element;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

GEOMETRIC_END_NAMESPACE
ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
