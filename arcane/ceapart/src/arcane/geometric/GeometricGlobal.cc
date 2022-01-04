// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* GeometricGlobal.h                                           (C) 2000-2014 */
/*                                                                           */
/* Déclarations globales pour la composante géométrique.                     */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/TraceInfo.h"

#include "arcane/ArcaneTypes.h"

#include "arcane/geometric/GeometricGlobal.h"
#include "arcane/geometric/GeomType.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE
GEOMETRIC_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

static const char*
_geometricTypeToName(GeomType type)
{
  switch(type){
  case GeomType::NullType: return "Null";
  case GeomType::Vertex: return "Vertex";
  case GeomType::Line2: return "Line2";
  case GeomType::Triangle3: return "Triangle3";
  case GeomType::Quad4: return "Quad4";
  case GeomType::Pentagon5: return "Pentagon5";
  case GeomType::Hexagon6: return "Hexagon6";
  case GeomType::Tetraedron4: return "Tetraedron4";
  case GeomType::Pyramid5: return "Pyramid5";
  case GeomType::Pentaedron6: return "Pentaedron6";
  case GeomType::Hexaedron8: return "Hexaedron8";
  case GeomType::Heptaedron10: return "Heptaedron10";
  case GeomType::Octaedron12: return "Octaedron12";
  }
  return "Unknown";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" void
_arcaneBadType(GeomType type,GeomType wanted_type)
{
  const char* t1 = _geometricTypeToName(type);
  const char* t2 = _geometricTypeToName(wanted_type);
  throw FatalErrorException(A_FUNCINFO,
                            String::format("Bad geometric type type={0}, expected={1}",
                                           t1,t2));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

GEOMETRIC_END_NAMESPACE
ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

