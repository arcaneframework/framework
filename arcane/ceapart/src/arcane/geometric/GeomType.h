// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* GeomType.h                                                  (C) 2000-2014 */
/*                                                                           */
/* Enumération spécifiant le type de polygone ou polyèdre.                   */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_GEOMETRIC_GEOMTYPE_H
#define ARCANE_GEOMETRIC_GEOMTYPE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/geometric/GeometricGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE
GEOMETRIC_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Doxygen ne comprend pas (dans la 1.8.7) la documentation des enum class.
// On fait donc une fausse classe pour avoir la même notion

/*!
 * \ingroup ArcaneGeometric
 * \brief Enumération spécifiant le type de polygone ou polyèdre associé
 * à un élément ou une forme géométrique.
 *
 * NOTE: Les valeurs de cette énumération doivent correspondre à 
 * celles des IT_* définis dans ArcaneTypes.h
 */

#ifdef DOXYGEN_DOC
class GeomType
{
 public:

  enum{
    //! Élément nul
    NullType = IT_NullType,
    //! Sommet
    Vertex = IT_Vertex,
    //! Ligne
    Line2 = IT_Line2,
    //! Triangle
    Triangle3 = IT_Triangle3,
    //! Quadrangle
    Quad4 = IT_Quad4,
    //! Pentagone
    Pentagon5 = IT_Pentagon5,
    //! Hexagone
    Hexagon6 = IT_Hexagon6,
    //! Tétraèdre
    Tetraedron4 = IT_Tetraedron4,
    //! Pyramide
    Pyramid5 = IT_Pyramid5,
    //! Prisme
    Pentaedron6 = IT_Pentaedron6,
    //! Hexaèdre
    Hexaedron8 = IT_Hexaedron8,
    //! Prisme à base pentagonale
    Heptaedron10 = IT_Heptaedron10,
    //! Prisme à base hexagonale
    Octaedron12 = IT_Octaedron12
  }
};
#else
enum class GeomType
{
  //! Élément nul
  NullType = IT_NullType,
  //! Sommet
  Vertex = IT_Vertex,
  //! Ligne
  Line2 = IT_Line2,
  //! Triangle
  Triangle3 = IT_Triangle3,
  //! Quadrangle
  Quad4 = IT_Quad4,
  //! Pentagone
  Pentagon5 = IT_Pentagon5,
  //! Hexagone
  Hexagon6 = IT_Hexagon6,
  //! Tétraèdre
  Tetraedron4 = IT_Tetraedron4,
  //! Pyramide
  Pyramid5 = IT_Pyramid5,
  //! Prisme
  Pentaedron6 = IT_Pentaedron6,
  //! Hexaèdre
  Hexaedron8 = IT_Hexaedron8,
  //! Prisme à base pentagonale
  Heptaedron10 = IT_Heptaedron10,
  //! Prisme à base hexagonale
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

GEOMETRIC_END_NAMESPACE
ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

