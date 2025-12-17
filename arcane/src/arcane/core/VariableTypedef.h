// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VariableTypedef.h                                           (C) 2000-2025 */
/*                                                                           */
/* Déclarations des typedefs des variables.                                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_VARIABLETYPEDEF_H
#define ARCANE_CORE_VARIABLETYPEDEF_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"
#include "arcane/core/ItemTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \file VariableTypedef.h
 *
 * Déclarations des types des variables.
 */

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename DataType> class ItemVariableScalarRefT;
template <typename ItemType, typename DataType> class MeshVariableScalarRefT;
template <typename ItemType, typename DataType> class MeshVariableArrayRefT;
template <typename DataType> class ItemPartialVariableScalarRefT;
template <typename DataType> class ItemPartialVariableArrayRefT;
template <typename ItemType, typename DataType> class MeshPartialVariableScalarRefT;
template <typename ItemType, typename DataType> class MeshPartialVariableArrayRefT;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
  \ingroup Variable
  \brief Variable scalaire de type byte.
*/
typedef VariableRefScalarT<Byte> VariableScalarByte;

/*!
  \ingroup Variable
  \brief Variable scalaire de type réel
*/
typedef VariableRefScalarT<Real> VariableScalarReal;

/*!
  \ingroup Variable
  \brief Variable scalaire de type entier 16 bits
*/
typedef VariableRefScalarT<Int16> VariableScalarInt16;

/*!
  \ingroup Variable
  \brief Variable scalaire de type entier 32 bits
*/
typedef VariableRefScalarT<Int32> VariableScalarInt32;

/*!
  \ingroup Variable
  \brief Variable scalaire de type entier 64 bits
*/
typedef VariableRefScalarT<Int64> VariableScalarInt64;

/*!
  \ingroup Variable
  \brief Variable scalaire de type entier
*/
typedef VariableRefScalarT<Integer> VariableScalarInteger;

/*!
  \ingroup Variable
  \brief Variable scalaire de type dimension
  \deprecated Utiliser #VariableScalarInteger à la place
*/
typedef VariableRefScalarT<Integer> VariableScalarInteger;

/*!
  \ingroup Variable
  \brief Variable scalaire de type chaine de caractère
*/
typedef VariableRefScalarT<String> VariableScalarString;

/*!
  \ingroup Variable
  \brief Variable scalaire de type coordonnée (x,y,z)
*/
typedef VariableRefScalarT<Real3> VariableScalarReal3;

/*!
  \ingroup Variable
  \brief Variable scalaire de type tenseur 3x3
*/
typedef VariableRefScalarT<Real3x3> VariableScalarReal3x3;

/*!
  \ingroup Variable
  \brief Variable scalaire de type coordonnée (x,y)
*/
typedef VariableRefScalarT<Real2> VariableScalarReal2;

/*!
  \ingroup Variable
  \brief Variable scalaire de type tenseur 2x2
*/
typedef VariableRefScalarT<Real2x2> VariableScalarReal2x2;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
  \ingroup Variable
  \brief Variable tableau de type byte
*/
typedef VariableRefArrayT<Byte> VariableArrayByte;

/*!
  \ingroup Variable
  \brief Variable tableau de type réels
*/
typedef VariableRefArrayT<Real> VariableArrayReal;

/*!
  \ingroup Variable
  \brief Variable tableau de type entier 16 bits
*/
typedef VariableRefArrayT<Int16> VariableArrayInt16;

/*!
  \ingroup Variable
  \brief Variable tableau de type entier 32 bits
*/
typedef VariableRefArrayT<Int32> VariableArrayInt32;

/*!
  \ingroup Variable
  \brief Variable tableau de type entier 64 bits
*/
typedef VariableRefArrayT<Int64> VariableArrayInt64;

/*!
  \ingroup Variable
  \brief Variable tableau de type entier
*/
typedef VariableRefArrayT<Integer> VariableArrayInteger;

/*!
  \ingroup Variable
  \brief Variable tableau de type indices
  \deprecated Utiliser #VariableArrayInteger à la place
*/
typedef VariableRefArrayT<Integer> VariableArrayInteger;

/*!
  \ingroup Variable
  \brief Variable tableau de type chaîne de caractères
*/
typedef VariableRefArrayT<String> VariableArrayString;

/*!
  \ingroup Variable
  \brief Variable tableau de type coordonnées (x,y,z)
*/
typedef VariableRefArrayT<Real3> VariableArrayReal3;

/*!
  \ingroup Variable
  \brief Variable tableau de type tenseur de réels
*/
typedef VariableRefArrayT<Real3x3> VariableArrayReal3x3;

/*!
  \ingroup Variable
  \brief Variable tableau de type coordonnées (x,y)
*/
typedef VariableRefArrayT<Real2> VariableArrayReal2;

/*!
  \ingroup Variable
  \brief Variable tableau de type tenseur 2x2
*/
typedef VariableRefArrayT<Real2x2> VariableArrayReal2x2;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
  \ingroup Variable
  \brief Variable tableau à deux dimensions de type byte
*/
typedef VariableRefArray2T<Byte> VariableArray2Byte;

/*!
  \ingroup Variable
  \brief Variable tableau à deux dimensions de type réels
*/
typedef VariableRefArray2T<Real> VariableArray2Real;

/*!
  \ingroup Variable
  \brief Variable tableau à deux dimensions de type entier
*/
typedef VariableRefArray2T<Integer> VariableArray2Integer;

/*!
  \ingroup Variable
  \brief Variable tableau à deux dimensions de type entier 16 bits
*/
typedef VariableRefArray2T<Int16> VariableArray2Int16;

/*!
  \ingroup Variable
  \brief Variable tableau à deux dimensions de type entier 32 bits
*/
typedef VariableRefArray2T<Int32> VariableArray2Int32;

/*!
  \ingroup Variable
  \brief Variable tableau à deux dimensions de type entier 64 bits
*/
typedef VariableRefArray2T<Int64> VariableArray2Int64;

/*!
  \ingroup Variable
  \brief Variable tableau à deux dimensions de type coordonnées (x,y,z)
*/
typedef VariableRefArray2T<Real3> VariableArray2Real3;

/*!
  \ingroup Variable
  \brief Variable tableau à deux dimensions de type tenseur de réels 3x3
*/
typedef VariableRefArray2T<Real3x3> VariableArray2Real3x3;

/*!
  \ingroup Variable
  \brief Variable tableau à deux dimensions de type coordonnées (x,y)
*/
typedef VariableRefArray2T<Real2> VariableArray2Real2;

/*!
  \ingroup Variable
  \brief Variable tableau à deux dimensions de type tenseur de réels 3x3
*/
typedef VariableRefArray2T<Real2x2> VariableArray2Real2x2;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
  \ingroup Variable
  \brief Grandeur de type réel
*/
typedef ItemVariableScalarRefT<Real> VariableItemReal;

/*!
  \ingroup Variable
  \brief Grandeur au noeud de type réel
*/
typedef MeshVariableScalarRefT<Node, Real> VariableNodeReal;

/*!
  \ingroup Variable
  \brief Grandeur au noeud de type réel
*/
typedef MeshVariableScalarRefT<Edge, Real> VariableEdgeReal;

/*!
  \ingroup Variable
  \brief Grandeur aux faces de type réel
*/
typedef MeshVariableScalarRefT<Face, Real> VariableFaceReal;

/*!
  \ingroup Variable
  \brief Grandeur au centre des mailles de type réel
*/
typedef MeshVariableScalarRefT<Cell, Real> VariableCellReal;

/*!
  \ingroup Variable
  \brief Grandeur particulaire de type réel
*/
typedef MeshVariableScalarRefT<Particle, Real> VariableParticleReal;

/*!
  \ingroup Variable
  \brief Grandeur de DDL de type reel
*/
typedef MeshVariableScalarRefT<DoF, Real> VariableDoFReal;
/*!
  \ingroup Variable
  \brief Grandeur au centre des mailles de type rel
*/
typedef MeshVariableScalarRefT<Cell, Real> VariableCellReal;
/*!
  \ingroup Variable
  \brief Grandeur de type coordonn?es 2D
*/
typedef ItemVariableScalarRefT<Real2> VariableItemReal2;

/*!
  \ingroup Variable
  \brief Grandeur au noeud de type coordonnées
*/
typedef MeshVariableScalarRefT<Node, Real2> VariableNodeReal2;

/*!
  \ingroup Variable
  \brief Grandeur au noeud de type coordonnées
*/
typedef MeshVariableScalarRefT<Edge, Real2> VariableEdgeReal2;

/*!
  \ingroup Variable
  \brief Grandeur aux faces de type coordonnées
*/
typedef MeshVariableScalarRefT<Face, Real2> VariableFaceReal2;

/*!
  \ingroup Variable
  \brief Grandeur au centre des mailles de type coordonnées
*/
typedef MeshVariableScalarRefT<Cell, Real2> VariableCellReal2;

/*!
  \ingroup Variable
  \brief Grandeur particulaire de type coordonnées
*/
typedef MeshVariableScalarRefT<Particle, Real2> VariableParticleReal2;

/*!
  \ingroup Variable
  \brief Grandeur de DDL de type coordonnees
*/
typedef MeshVariableScalarRefT<DoF, Real2> VariableDoFReal2;
/*!
  \ingroup Variable
  \brief Grandeur de type coordonn?es 3D
*/
typedef ItemVariableScalarRefT<Real3> VariableItemReal3;

/*!
  \ingroup Variable
  \brief Grandeur au noeud de type coordonnées
*/
typedef MeshVariableScalarRefT<Node, Real3> VariableNodeReal3;

/*!
  \ingroup Variable
  \brief Grandeur au noeud de type coordonnées
*/
typedef MeshVariableScalarRefT<Edge, Real3> VariableEdgeReal3;

/*!
  \ingroup Variable
  \brief Grandeur aux faces de type coordonnées
*/
typedef MeshVariableScalarRefT<Face, Real3> VariableFaceReal3;

/*!
  \ingroup Variable
  \brief Grandeur au centre des mailles de type coordonnées
*/
typedef MeshVariableScalarRefT<Cell, Real3> VariableCellReal3;

/*!
  \ingroup Variable
  \brief Grandeur particulaire de type coordonnées
*/
typedef MeshVariableScalarRefT<Particle, Real3> VariableParticleReal3;

/*!
  \ingroup Variable
  \brief Grandeur de DDL de type coordonnees
*/
typedef MeshVariableScalarRefT<DoF, Real3> VariableDoFReal3;
/*!
  \ingroup Variable
  \brief Grandeur de type tenseur de r?els
*/
typedef ItemVariableScalarRefT<Real2x2> VariableItemReal2x2;

/*!
  \ingroup Variable
  \brief Grandeur au noeud de type tenseur de réels
*/
typedef MeshVariableScalarRefT<Node, Real2x2> VariableNodeReal2x2;

/*!
  \ingroup Variable
  \brief Grandeur au noeud de type tenseur de réels
*/
typedef MeshVariableScalarRefT<Edge, Real2x2> VariableEdgeReal2x2;

/*!
  \ingroup Variable
  \brief Grandeur aux faces de type tenseur de réels
*/
typedef MeshVariableScalarRefT<Face, Real2x2> VariableFaceReal2x2;

/*!
  \ingroup Variable
  \brief Grandeur au centre des mailles de type tenseur de réels
*/
typedef MeshVariableScalarRefT<Cell, Real2x2> VariableCellReal2x2;

/*!
  \ingroup Variable
  \brief Grandeur particulaire de type tenseur de réels
*/
typedef MeshVariableScalarRefT<Particle, Real2x2> VariableParticleReal2x2;

/*!
  \ingroup Variable
  \brief Grandeur de DDL de type tenseur de reels
*/
typedef MeshVariableScalarRefT<DoF, Real2x2> VariableDoFReal2x2;
/*!
  \ingroup Variable
  \brief Grandeur de type tenseur de r?els
*/
typedef ItemVariableScalarRefT<Real3x3> VariableItemReal3x3;

/*!
  \ingroup Variable
  \brief Grandeur au noeud de type tenseur de réels
*/
typedef MeshVariableScalarRefT<Node, Real3x3> VariableNodeReal3x3;

/*!
  \ingroup Variable
  \brief Grandeur au noeud de type tenseur de réels
*/
typedef MeshVariableScalarRefT<Edge, Real3x3> VariableEdgeReal3x3;

/*!
  \ingroup Variable
  \brief Grandeur aux faces de type tenseur de réels
*/
typedef MeshVariableScalarRefT<Face, Real3x3> VariableFaceReal3x3;

/*!
  \ingroup Variable
  \brief Grandeur au centre des mailles de type tenseur de réels
*/
typedef MeshVariableScalarRefT<Cell, Real3x3> VariableCellReal3x3;

/*!
  \ingroup Variable
  \brief Grandeur particulaire de type tenseur de réels
*/
typedef MeshVariableScalarRefT<Particle, Real3x3> VariableParticleReal3x3;

/*!
  \ingroup Variable
  \brief Grandeur de DDL de type tenseur de reels
*/
typedef MeshVariableScalarRefT<DoF, Real3x3> VariableDoFReal3x3;
/*!
  \ingroup Variable
  \brief Grandeur de type entier
*/
typedef ItemVariableScalarRefT<Integer> VariableItemInteger;

/*!
  \ingroup Variable
  \brief Grandeur au noeud de type entier
*/
typedef MeshVariableScalarRefT<Node, Integer> VariableNodeInteger;

/*!
  \ingroup Variable
  \brief Grandeur au noeud de type entier
*/
typedef MeshVariableScalarRefT<Edge, Integer> VariableEdgeInteger;

/*!
  \ingroup Variable
  \brief Grandeur aux faces de type entier
*/
typedef MeshVariableScalarRefT<Face, Integer> VariableFaceInteger;

/*!
  \ingroup Variable
  \brief Grandeur au centre des mailles de type entier
*/
typedef MeshVariableScalarRefT<Cell, Integer> VariableCellInteger;

/*!
  \ingroup Variable
  \brief Grandeur particulaire de type entier
*/
typedef MeshVariableScalarRefT<Particle, Integer> VariableParticleInteger;

/*!
  \ingroup Variable
  \brief Grandeur de DDL de type entier
*/
typedef MeshVariableScalarRefT<DoF, Integer> VariableDoFInteger;
/*!
  \ingroup Variable
  \brief Grandeur de type entier 32 bits
*/
typedef ItemVariableScalarRefT<Int16> VariableItemInt16;
/*!
  \ingroup Variable
  \brief Grandeur de type entier 32 bits
*/
typedef ItemVariableScalarRefT<Int32> VariableItemInt32;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
  \ingroup Variable
  \brief Grandeur au noeud de type entier 16 bits
*/
typedef MeshVariableScalarRefT<Node, Int16> VariableNodeInt16;

/*!
  \ingroup Variable
  \brief  Grandeur au noeud de type entier 16 bits
*/
typedef MeshVariableScalarRefT<Edge, Int16> VariableEdgeInt16;

/*!
  \ingroup Variable
  \brief  Grandeur aux faces de type entier 16 bits
*/
typedef MeshVariableScalarRefT<Face, Int16> VariableFaceInt16;

/*!
  \ingroup Variable
  \brief  Grandeur au centre des mailles de type entier 16 bits
*/
typedef MeshVariableScalarRefT<Cell, Int16> VariableCellInt16;

/*!
  \ingroup Variable
  \brief  Grandeur particulaire de type entier 16 bits
*/
typedef MeshVariableScalarRefT<Particle, Int16> VariableParticleInt16;

/*!
  \ingroup Variable
  \brief Grandeur de DDL de type entier 16 bits
*/
typedef MeshVariableScalarRefT<DoF, Int16> VariableDoFInt16;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
  \ingroup Variable
  \brief Grandeur au noeud de type entier 32 bits
*/
typedef MeshVariableScalarRefT<Node, Int32> VariableNodeInt32;

/*!
  \ingroup Variable
  \brief  Grandeur au noeud de type entier 32 bits
*/
typedef MeshVariableScalarRefT<Edge, Int32> VariableEdgeInt32;

/*!
  \ingroup Variable
  \brief  Grandeur aux faces de type entier 32 bits
*/
typedef MeshVariableScalarRefT<Face, Int32> VariableFaceInt32;

/*!
  \ingroup Variable
  \brief  Grandeur au centre des mailles de type entier 32 bits
*/
typedef MeshVariableScalarRefT<Cell, Int32> VariableCellInt32;

/*!
  \ingroup Variable
  \brief  Grandeur particulaire de type entier 32 bits
*/
typedef MeshVariableScalarRefT<Particle, Int32> VariableParticleInt32;

/*!
  \ingroup Variable
  \brief Grandeur de DDL de type entier 32 bits
*/
typedef MeshVariableScalarRefT<DoF, Int32> VariableDoFInt32;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
  \ingroup Variable
  \brief  Grandeur de type entier 64 bits
*/
typedef ItemVariableScalarRefT<Int64> VariableItemInt64;

/*!
  \ingroup Variable
  \brief  Grandeur au noeud de type entier 64 bits
*/
typedef MeshVariableScalarRefT<Node, Int64> VariableNodeInt64;

/*!
  \ingroup Variable
  \brief  Grandeur au noeud de type entier 64 bits
*/
typedef MeshVariableScalarRefT<Edge, Int64> VariableEdgeInt64;

/*!
  \ingroup Variable
  \brief  Grandeur aux faces de type entier 64 bits
*/
typedef MeshVariableScalarRefT<Face, Int64> VariableFaceInt64;

/*!
  \ingroup Variable
  \brief  Grandeur au centre des mailles de type entier 64 bits
*/
typedef MeshVariableScalarRefT<Cell, Int64> VariableCellInt64;

/*!
  \ingroup Variable
  \brief  Grandeur particulaire de type entier 64 bits
*/
typedef MeshVariableScalarRefT<Particle, Int64> VariableParticleInt64;

/*!
  \ingroup Variable
  \brief Grandeur de DDL de type entier 64 bits
*/
typedef MeshVariableScalarRefT<DoF, Int64> VariableDoFInt64;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
  \ingroup Variable
  \brief  Grandeur au noeud de type entier naturel
  \deprecated Utiliser #VariableNodeInteger à la place
*/
typedef MeshVariableScalarRefT<Node, Integer> VariableNodeInteger;

/*!
  \ingroup Variable
  \brief  Grandeur aux faces de type entier naturel
  \deprecated Utiliser #VariableFaceInteger à la place
*/
typedef MeshVariableScalarRefT<Face, Integer> VariableFaceInteger;

/*!
  \ingroup Variable
  \brief  Grandeur au centre des mailles de type entier naturel
  \deprecated Utiliser #VariableCellInteger à la place
*/
typedef MeshVariableScalarRefT<Cell, Integer> VariableCellInteger;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
  \ingroup Variable
  \brief  Grandeur de type entier 64 bits
*/
typedef ItemVariableScalarRefT<Byte> VariableItemByte;

/*!
  \ingroup Variable
  \brief  Grandeur au noeud de type octet
*/
typedef MeshVariableScalarRefT<Node, Byte> VariableNodeByte;

/*!
  \ingroup Variable
  \brief  Grandeur au noeud de type octet
*/
typedef MeshVariableScalarRefT<Edge, Byte> VariableEdgeByte;

/*!
  \ingroup Variable
  \brief  Grandeur aux faces de type octet
*/
typedef MeshVariableScalarRefT<Face, Byte> VariableFaceByte;

/*!
  \ingroup Variable
  \brief  Grandeur au centre des mailles de type octet
*/
typedef MeshVariableScalarRefT<Cell, Byte> VariableCellByte;

/*!
  \ingroup Variable
  \brief  Grandeur particulaire de type octet
*/
typedef MeshVariableScalarRefT<Particle, Byte> VariableParticleByte;

/*!
  \ingroup Variable
  \brief Grandeur de DDL de type octet
*/
typedef MeshVariableScalarRefT<DoF, Byte> VariableDoFByte;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
  \ingroup Variable
  \brief  Grandeur au noeud de type booléen
*/
typedef MeshVariableScalarRefT<Node, Byte> VariableNodeBool;

/*!
  \ingroup Variable
  \brief  Grandeur au noeud de type booléen
*/
typedef MeshVariableScalarRefT<Edge, Byte> VariableEdgeBool;

/*!
  \ingroup Variable
  \brief  Grandeur aux faces de type booléen
*/
typedef MeshVariableScalarRefT<Face, Byte> VariableFaceBool;

/*!
  \ingroup Variable
  \brief  Grandeur au centre des mailles de type booléen
*/
typedef MeshVariableScalarRefT<Cell, Byte> VariableCellBool;

/*!
  \ingroup Variable
  \brief  Grandeur particulaire de type booléen
*/
typedef MeshVariableScalarRefT<Particle, Byte> VariableParticleBool;

/*!
  \ingroup Variable
  \brief Grandeur de DDL de type booleen
*/
typedef MeshVariableScalarRefT<DoF, Byte> VariableDoFBool;

/*!---------------------------------------------------------------------------*/
/*!---------------------------------------------------------------------------*/

/*!
  \ingroup Variable
  \brief  Grandeur au noeud de type tableau de réel
*/
typedef MeshVariableArrayRefT<Node, Real> VariableNodeArrayReal;

/*!
  \ingroup Variable
  \brief  Grandeur au noeud de type tableau de réel
*/
typedef MeshVariableArrayRefT<Edge, Real> VariableEdgeArrayReal;

/*!
  \ingroup Variable
  \brief  Grandeur aux faces de type tableau de réel
*/
typedef MeshVariableArrayRefT<Face, Real> VariableFaceArrayReal;

/*!
  \ingroup Variable
  \brief  Grandeur au centre des mailles de type tableau de réel
*/
typedef MeshVariableArrayRefT<Cell, Real> VariableCellArrayReal;

/*!
  \ingroup Variable
  \brief  Grandeur particulaire de type tableau de r?el
*/
typedef MeshVariableArrayRefT<Particle, Real> VariableParticleArrayReal;

/*!
  \ingroup Variable
  \brief Grandeur de DDL de type tableau de reels
*/
typedef MeshVariableArrayRefT<DoF, Real> VariableDoFArrayReal;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
  \ingroup Variable
  \brief  Grandeur au noeud de type tableau de coordonnées
*/
typedef MeshVariableArrayRefT<Node, Real2> VariableNodeArrayReal2;

/*!
  \ingroup Variable
  \brief  Grandeur au noeud de type tableau de coordonnées
*/
typedef MeshVariableArrayRefT<Edge, Real2> VariableEdgeArrayReal2;

/*!
  \ingroup Variable
  \brief  Grandeur aux faces de type tableau de coordonnées
*/
typedef MeshVariableArrayRefT<Face, Real2> VariableFaceArrayReal2;

/*!
  \ingroup Variable
  \brief  Grandeur au centre des mailles de type tableau de coordonnées
*/
typedef MeshVariableArrayRefT<Cell, Real2> VariableCellArrayReal2;

/*!
  \ingroup Variable
  \brief  Grandeur particulaire de type tableau de coordonnées
*/
typedef MeshVariableArrayRefT<Particle, Real2> VariableParticleArrayReal2;

/*!
  \ingroup Variable
  \brief Grandeur de DDL de type tableau de coordonnees
*/
typedef MeshVariableArrayRefT<DoF, Real2> VariableDoFArrayReal2;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
  \ingroup Variable
  \brief  Grandeur au noeud de type tableau de coordonnées
*/
typedef MeshVariableArrayRefT<Node, Real3> VariableNodeArrayReal3;

/*!
  \ingroup Variable
  \brief  Grandeur aux faces de type tableau de coordonnées
*/
typedef MeshVariableArrayRefT<Face, Real3> VariableFaceArrayReal3;

/*!
  \ingroup Variable
  \brief  Grandeur au centre des mailles de type tableau de coordonnées
*/
typedef MeshVariableArrayRefT<Cell, Real3> VariableCellArrayReal3;

/*!
  \ingroup Variable
  \brief  Grandeur particulaire de type tableau de coordonnées
*/
typedef MeshVariableArrayRefT<Particle, Real3> VariableParticleArrayReal3;

/*!
  \ingroup Variable
  \brief Grandeur de DDL de type tableau de reels
*/
typedef MeshVariableArrayRefT<DoF, Real3> VariableDoFArrayReal3;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
  \ingroup Variable
  \brief  Grandeur au noeud de type tableau de tenseur de réels
*/
typedef MeshVariableArrayRefT<Node, Real2x2> VariableNodeArrayReal2x2;

/*!
  \ingroup Variable
  \brief  Grandeur aux faces de type tableau de tenseur de réels
*/
typedef MeshVariableArrayRefT<Face, Real2x2> VariableFaceArrayReal2x2;

/*!
  \ingroup Variable
  \brief  Grandeur au centre des mailles de type tableau de tenseur de réels
*/
typedef MeshVariableArrayRefT<Cell, Real2x2> VariableCellArrayReal2x2;

/*!
  \ingroup Variable
  \brief  Grandeur particulaire de type tableau de tenseur de r?els
*/
typedef MeshVariableArrayRefT<Particle, Real2x2> VariableParticleArrayReal2x2;

/*!
  \ingroup Variable
  \brief Grandeur de DDL de type tableau de tenseur de reels
*/
typedef MeshVariableArrayRefT<DoF, Real2x2> VariableDoFArrayReal2x2;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
  \ingroup Variable
  \brief  Grandeur au noeud de type tableau de tenseur de réels
*/
typedef MeshVariableArrayRefT<Node, Real3x3> VariableNodeArrayReal3x3;

/*!
  \ingroup Variable
  \brief  Grandeur aux faces de type tableau de tenseur de réels
*/
typedef MeshVariableArrayRefT<Face, Real3x3> VariableFaceArrayReal3x3;

/*!
  \ingroup Variable
  \brief  Grandeur au centre des mailles de type tableau de tenseur de réels
*/
typedef MeshVariableArrayRefT<Cell, Real3x3> VariableCellArrayReal3x3;

/*!
  \ingroup Variable
  \brief  Grandeur particulaire de type tableau de tenseur de réels
*/
typedef MeshVariableArrayRefT<Particle, Real3x3> VariableParticleArrayReal3x3;

/*!
  \ingroup Variable
  \brief Grandeur de DDL de type tableau de tenseur de reels
*/
typedef MeshVariableArrayRefT<DoF, Real3x3> VariableDoFArrayReal3x3;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
  \ingroup Variable
  \brief  Grandeur au noeud de type tableau d'entiers
*/
typedef MeshVariableArrayRefT<Node, Integer> VariableNodeArrayInteger;

/*!
  \ingroup Variable
  \brief  Grandeur aux faces de type tableau d'entiers
*/
typedef MeshVariableArrayRefT<Face, Integer> VariableFaceArrayInteger;

/*!
  \ingroup Variable
  \brief  Grandeur au centre des mailles de type tableau d'entiers
*/
typedef MeshVariableArrayRefT<Cell, Integer> VariableCellArrayInteger;

/*!
  \ingroup Variable
  \brief  Grandeur particulaire de type tableau d'entiers
*/
typedef MeshVariableArrayRefT<Particle, Integer> VariableParticleArrayInteger;

/*!
  \ingroup Variable
  \brief Grandeur de DDL de type tableau d'entiers
*/
typedef MeshVariableArrayRefT<DoF, Integer> VariableDoFArrayInteger;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
  \ingroup Variable
  \brief  Grandeur au noeud de type tableau d'entiers 16 bits
*/
typedef MeshVariableArrayRefT<Node, Int16> VariableNodeArrayInt16;

/*!
  \ingroup Variable
  \brief  Grandeur aux faces de type tableau d'entiers 16 bits
*/
typedef MeshVariableArrayRefT<Face, Int16> VariableFaceArrayInt16;

/*!
  \ingroup Variable
  \brief  Grandeur au centre des mailles de type tableau d'entiers 16 bits
*/
typedef MeshVariableArrayRefT<Cell, Int16> VariableCellArrayInt16;

/*!
  \ingroup Variable
  \brief  Grandeur particulaire de type tableau d'entiers 16 bits
*/
typedef MeshVariableArrayRefT<Particle, Int16> VariableParticleArrayInt16;

/*!
  \ingroup Variable
  \brief Grandeur de DDL de type tableau d'entiers 16 bits
*/
typedef MeshVariableArrayRefT<DoF, Int32> VariableDoFArrayInt32;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
  \ingroup Variable
  \brief  Grandeur au noeud de type tableau d'entiers
*/
typedef MeshVariableArrayRefT<Node, Int32> VariableNodeArrayInt32;

/*!
  \ingroup Variable
  \brief  Grandeur aux faces de type tableau d'entiers
*/
typedef MeshVariableArrayRefT<Face, Int32> VariableFaceArrayInt32;

/*!
  \ingroup Variable
  \brief  Grandeur au centre des mailles de type tableau d'entiers
*/
typedef MeshVariableArrayRefT<Cell, Int32> VariableCellArrayInt32;

/*!
  \ingroup Variable
  \brief  Grandeur particulaire de type tableau d'entiers
*/
typedef MeshVariableArrayRefT<Particle, Int32> VariableParticleArrayInt32;

/*!
  \ingroup Variable
  \brief Grandeur de DDL de type tableau d'entiers 32 bits
*/
typedef MeshVariableArrayRefT<DoF, Int32> VariableDoFArrayInt32;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
  \ingroup Variable
  \brief  Grandeur au noeud de type tableau d'entiers
*/
typedef MeshVariableArrayRefT<Node, Int64> VariableNodeArrayInt64;

/*!
  \ingroup Variable
  \brief  Grandeur aux faces de type tableau d'entiers
*/
typedef MeshVariableArrayRefT<Face, Int64> VariableFaceArrayInt64;

/*!
  \ingroup Variable
  \brief  Grandeur au centre des mailles de type tableau d'entiers
*/
typedef MeshVariableArrayRefT<Cell, Int64> VariableCellArrayInt64;

/*!
  \ingroup Variable
  \brief  Grandeur particulaire de type tableau d'entiers
*/
typedef MeshVariableArrayRefT<Particle, Int64> VariableParticleArrayInt64;

/*!
  \ingroup Variable
  \brief Grandeur de DDL de type tableau d'entiers 64 bits
*/
typedef MeshVariableArrayRefT<DoF, Int64> VariableDoFArrayInt64;

/*!
  \ingroup Variable
  \brief Grandeur de DDL de type tableau d'entiers 64 bits
*/
typedef MeshVariableArrayRefT<DoF, Int64> VariableDoFArrayInt64;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
  \ingroup Variable
  \brief  Grandeur au noeud de type tableau d'octet
*/
typedef MeshVariableArrayRefT<Node, Byte> VariableNodeArrayByte;

/*!
  \ingroup Variable
  \brief  Grandeur aux faces de type tableau d'octet
*/
typedef MeshVariableArrayRefT<Face, Byte> VariableFaceArrayByte;

/*!
  \ingroup Variable
  \brief  Grandeur au centre des mailles de type tableau d'octet
*/
typedef MeshVariableArrayRefT<Cell, Byte> VariableCellArrayByte;

/*!
  \ingroup Variable
  \brief  Grandeur particulaire de type tableau d'octets
*/
typedef MeshVariableArrayRefT<Particle, Byte> VariableParticleArrayByte;

/*!
  \ingroup Variable
  \brief Grandeur de DDL de type tableau d'octets
*/
typedef MeshVariableArrayRefT<DoF, Byte> VariableDoFArrayByte;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
  \ingroup Variable
  \brief  Grandeur au noeud de type tableau de booléen
*/
typedef MeshVariableArrayRefT<Node, Byte> VariableNodeArrayBool;

/*!
  \ingroup Variable
  \brief  Grandeur aux faces de type tableau de booléen
*/
typedef MeshVariableArrayRefT<Face, Byte> VariableFaceArrayBool;

/*!
  \ingroup Variable
  \brief  Grandeur au centre des mailles de type tableau de booléen
*/
typedef MeshVariableArrayRefT<Cell, Byte> VariableCellArrayBool;

/*!
  \ingroup Variable
  \brief  Grandeur particulaire de type tableau de booléen
*/
typedef MeshVariableArrayRefT<Particle, Byte> VariableParticleArrayBool;

/*!
  \ingroup Variable
  \brief Grandeur de DDL de type tableau de booleens
*/
typedef MeshVariableArrayRefT<DoF, Byte> VariableDoFArrayBool;

/*!---------------------------------------------------------------------------*/
/*!---------------------------------------------------------------------------*/

/*!---------------------------------------------------------------------------*/
/*!---------------------------------------------------------------------------*/

/*!
  \ingroup Variable partielle
  \brief  Grandeur de type réel
*/
typedef ItemPartialVariableScalarRefT<Real> PartialVariableItemReal;

/*!
  \ingroup Variable partielle
  \brief  Grandeur au noeud de type réel
*/
typedef MeshPartialVariableScalarRefT<Node, Real> PartialVariableNodeReal;

/*!
  \ingroup Variable partielle
  \brief  Grandeur au noeud de type réel
*/
typedef MeshPartialVariableScalarRefT<Edge, Real> PartialVariableEdgeReal;

/*!
  \ingroup Variable partielle
  \brief  Grandeur aux faces de type réel
*/
typedef MeshPartialVariableScalarRefT<Face, Real> PartialVariableFaceReal;

/*!
  \ingroup Variable partielle
  \brief  Grandeur au centre des mailles de type réel
*/
typedef MeshPartialVariableScalarRefT<Cell, Real> PartialVariableCellReal;

/*!
  \ingroup Variable partielle
  \brief  Grandeur particulaire de type réel
*/
typedef MeshPartialVariableScalarRefT<Particle, Real> PartialVariableParticleReal;

/*!
  \ingroup Variable partielle
  \brief Grandeur de DDL de type reel
*/
typedef MeshPartialVariableScalarRefT<DoF, Real> PartialVariableDoFReal;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
  \ingroup Variable partielle
  \brief  Grandeur de type coordonnées 2D
*/
typedef ItemPartialVariableScalarRefT<Real2> PartialVariableItemReal2;

/*!
  \ingroup Variable partielle
  \brief  Grandeur au noeud de type coordonnées
*/
typedef MeshPartialVariableScalarRefT<Node, Real2> PartialVariableNodeReal2;

/*!
  \ingroup Variable partielle
  \brief  Grandeur au noeud de type coordonnées
*/
typedef MeshPartialVariableScalarRefT<Edge, Real2> PartialVariableEdgeReal2;

/*!
  \ingroup Variable partielle
  \brief  Grandeur aux faces de type coordonnées
*/
typedef MeshPartialVariableScalarRefT<Face, Real2> PartialVariableFaceReal2;

/*!
  \ingroup Variable partielle
  \brief  Grandeur au centre des mailles de type coordonnées
*/
typedef MeshPartialVariableScalarRefT<Cell, Real2> PartialVariableCellReal2;

/*!
  \ingroup Variable partielle
  \brief  Grandeur particulaire de type coordonnées
*/
typedef MeshPartialVariableScalarRefT<Particle, Real2> PartialVariableParticleReal2;

/*!
  \ingroup Variable partielle
  \brief Grandeur de DDL de type coordonnees
*/
typedef MeshPartialVariableScalarRefT<DoF, Real2> PartialVariableDoFReal2;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
  \ingroup Variable partielle
  \brief  Grandeur de type coordonnées 3D
*/
typedef ItemPartialVariableScalarRefT<Real3> PartialVariableItemReal3;

/*!
  \ingroup Variable partielle
  \brief  Grandeur au noeud de type coordonnées
*/
typedef MeshPartialVariableScalarRefT<Node, Real3> PartialVariableNodeReal3;

/*!
  \ingroup Variable partielle
  \brief  Grandeur au noeud de type coordonnées
*/
typedef MeshPartialVariableScalarRefT<Edge, Real3> PartialVariableEdgeReal3;

/*!
  \ingroup Variable partielle
  \brief  Grandeur aux faces de type coordonnées
*/
typedef MeshPartialVariableScalarRefT<Face, Real3> PartialVariableFaceReal3;

/*!
  \ingroup Variable partielle
  \brief  Grandeur au centre des mailles de type coordonnées
*/
typedef MeshPartialVariableScalarRefT<Cell, Real3> PartialVariableCellReal3;

/*!
  \ingroup Variable partielle
  \brief  Grandeur particulaire de type coordonnées
*/
typedef MeshPartialVariableScalarRefT<Particle, Real3> PartialVariableParticleReal3;

/*!
  \ingroup Variable partielle
  \brief Grandeur de DDL de type coordonnees
*/
typedef MeshPartialVariableScalarRefT<DoF, Real3> PartialVariableDoFReal3;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
  \ingroup Variable partielle
  \brief  Grandeur de type tenseur de réels
*/
typedef ItemPartialVariableScalarRefT<Real2x2> PartialVariableItemReal2x2;

/*!
  \ingroup Variable partielle
  \brief  Grandeur au noeud de type tenseur de réels
*/
typedef MeshPartialVariableScalarRefT<Node, Real2x2> PartialVariableNodeReal2x2;

/*!
  \ingroup Variable partielle
  \brief  Grandeur au noeud de type tenseur de réels
*/
typedef MeshPartialVariableScalarRefT<Edge, Real2x2> PartialVariableEdgeReal2x2;

/*!
  \ingroup Variable partielle
  \brief  Grandeur aux faces de type tenseur de réels
*/
typedef MeshPartialVariableScalarRefT<Face, Real2x2> PartialVariableFaceReal2x2;

/*!
  \ingroup Variable partielle
  \brief  Grandeur au centre des mailles de type tenseur de réels
*/
typedef MeshPartialVariableScalarRefT<Cell, Real2x2> PartialVariableCellReal2x2;

/*!
  \ingroup Variable partielle
  \brief  Grandeur particulaire de type tenseur de réels
*/
typedef MeshPartialVariableScalarRefT<Particle, Real2x2> PartialVariableParticleReal2x2;

/*!
  \ingroup Variable partielle
  \brief Grandeur de DDL de type tenseur de reels
*/
typedef MeshPartialVariableScalarRefT<DoF, Real2x2> PartialVariableDoFReal2x2;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
  \ingroup Variable partielle
  \brief  Grandeur de type tenseur de réels
*/
typedef ItemPartialVariableScalarRefT<Real3x3> PartialVariableItemReal3x3;

/*!
  \ingroup Variable partielle
  \brief  Grandeur au noeud de type tenseur de réels
*/
typedef MeshPartialVariableScalarRefT<Node, Real3x3> PartialVariableNodeReal3x3;

/*!
  \ingroup Variable partielle
  \brief  Grandeur au noeud de type tenseur de réels
*/
typedef MeshPartialVariableScalarRefT<Edge, Real3x3> PartialVariableEdgeReal3x3;

/*!
  \ingroup Variable partielle
  \brief  Grandeur aux faces de type tenseur de réels
*/
typedef MeshPartialVariableScalarRefT<Face, Real3x3> PartialVariableFaceReal3x3;

/*!
  \ingroup Variable partielle
  \brief  Grandeur au centre des mailles de type tenseur de réels
*/
typedef MeshPartialVariableScalarRefT<Cell, Real3x3> PartialVariableCellReal3x3;

/*!
  \ingroup Variable partielle
  \brief  Grandeur particulaire de type tenseur de réels
*/
typedef MeshPartialVariableScalarRefT<Particle, Real3x3> PartialVariableParticleReal3x3;

/*!
  \ingroup Variable partielle
  \brief Grandeur de DDL de type tenseur de reels
*/
typedef MeshPartialVariableScalarRefT<DoF, Real3x3> PartialVariableDoFReal3x3;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
  \ingroup Variable partielle
  \brief  Grandeur de type entier
*/
typedef ItemPartialVariableScalarRefT<Integer> PartialVariableItemInteger;

/*!
  \ingroup Variable partielle
  \brief  Grandeur au noeud de type entier
*/
typedef MeshPartialVariableScalarRefT<Node, Integer> PartialVariableNodeInteger;

/*!
  \ingroup Variable partielle
  \brief  Grandeur au noeud de type entier
*/
typedef MeshPartialVariableScalarRefT<Edge, Integer> PartialVariableEdgeInteger;

/*!
  \ingroup Variable partielle
  \brief  Grandeur aux faces de type entier
*/
typedef MeshPartialVariableScalarRefT<Face, Integer> PartialVariableFaceInteger;

/*!
  \ingroup Variable partielle
  \brief  Grandeur au centre des mailles de type entier
*/
typedef MeshPartialVariableScalarRefT<Cell, Integer> PartialVariableCellInteger;

/*!
  \ingroup Variable partielle
  \brief  Grandeur particulaire de type entier
*/
typedef MeshPartialVariableScalarRefT<Particle, Integer> PartialVariableParticleInteger;

/*!
  \ingroup Variable partielle
  \brief Grandeur de DDL de type entier
*/
typedef MeshPartialVariableScalarRefT<DoF, Integer> PartialVariableDoFInteger;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
  \ingroup Variable partielle
  \brief  Grandeur de type entier 32 bits
*/
typedef ItemPartialVariableScalarRefT<Int32> PartialVariableItemInt32;

/*!
  \ingroup Variable partielle
  \brief  Grandeur au noeud de type entier 32 bits
*/
typedef MeshPartialVariableScalarRefT<Node, Int32> PartialVariableNodeInt32;

/*!
  \ingroup Variable partielle
  \brief  Grandeur au noeud de type entier 32 bits
*/
typedef MeshPartialVariableScalarRefT<Edge, Int32> PartialVariableEdgeInt32;

/*!
  \ingroup Variable partielle
  \brief  Grandeur aux faces de type entier 32 bits
*/
typedef MeshPartialVariableScalarRefT<Face, Int32> PartialVariableFaceInt32;

/*!
  \ingroup Variable partielle
  \brief  Grandeur au centre des mailles de type entier 32 bits
*/
typedef MeshPartialVariableScalarRefT<Cell, Int32> PartialVariableCellInt32;

/*!
  \ingroup Variable partielle
  \brief  Grandeur particulaire de type entier 32 bits
*/
typedef MeshPartialVariableScalarRefT<Particle, Int32> PartialVariableParticleInt32;

/*!
  \ingroup Variable partielle
  \brief Grandeur de DDL de type entier 32 bits
*/
typedef MeshPartialVariableScalarRefT<DoF, Int32> PartialVariableDoFInt32;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
  \ingroup Variable partielle
  \brief  Grandeur de type entier 16 bits
*/
typedef ItemPartialVariableScalarRefT<Int16> PartialVariableItemInt16;

/*!
  \ingroup Variable partielle
  \brief  Grandeur au noeud de type entier 16 bits
*/
typedef MeshPartialVariableScalarRefT<Node, Int16> PartialVariableNodeInt16;

/*!
  \ingroup Variable partielle
  \brief  Grandeur au noeud de type entier 16 bits
*/
typedef MeshPartialVariableScalarRefT<Edge, Int16> PartialVariableEdgeInt16;

/*!
  \ingroup Variable partielle
  \brief  Grandeur aux faces de type entier 16 bits
*/
typedef MeshPartialVariableScalarRefT<Face, Int16> PartialVariableFaceInt16;

/*!
  \ingroup Variable partielle
  \brief  Grandeur au centre des mailles de type entier 16 bits
*/
typedef MeshPartialVariableScalarRefT<Cell, Int16> PartialVariableCellInt16;

/*!
  \ingroup Variable partielle
  \brief  Grandeur particulaire de type entier 16 bits
*/
typedef MeshPartialVariableScalarRefT<Particle, Int16> PartialVariableParticleInt16;

/*!
  \ingroup Variable partielle
  \brief Grandeur de DDL de type entier 16 bits
*/
typedef MeshPartialVariableScalarRefT<DoF, Int16> PartialVariableDoFInt16;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
  \ingroup Variable
  \brief  Grandeur de type entier 64 bits
*/
typedef ItemPartialVariableScalarRefT<Int64> PartialVariableItemInt64;

/*!
  \ingroup Variable partielle
  \brief  Grandeur au noeud de type entier 64 bits
*/
typedef MeshPartialVariableScalarRefT<Node, Int64> PartialVariableNodeInt64;

/*!
  \ingroup Variable partielle
  \brief  Grandeur au noeud de type entier 64 bits
*/
typedef MeshPartialVariableScalarRefT<Edge, Int64> PartialVariableEdgeInt64;

/*!
  \ingroup Variable partielle
  \brief  Grandeur aux faces de type entier 64 bits
*/
typedef MeshPartialVariableScalarRefT<Face, Int64> PartialVariableFaceInt64;

/*!
  \ingroup Variable partielle
  \brief  Grandeur au centre des mailles de type entier 64 bits
*/
typedef MeshPartialVariableScalarRefT<Cell, Int64> PartialVariableCellInt64;

/*!
  \ingroup Variable partielle
  \brief  Grandeur particulaire de type entier 64 bits
*/
typedef MeshPartialVariableScalarRefT<Particle, Int64> PartialVariableParticleInt64;

/*!
  \ingroup Variable partielle
  \brief Grandeur de DDL de type entier 64 bits
*/
typedef MeshPartialVariableScalarRefT<DoF, Int64> PartialVariableDoFInt64;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
  \ingroup Variable partielle
  \brief  Grandeur de type entier 64 bits
*/
typedef ItemPartialVariableScalarRefT<Byte> PartialVariableItemByte;

/*!
  \ingroup Variable partielle
  \brief  Grandeur au noeud de type octet
*/
typedef MeshPartialVariableScalarRefT<Node, Byte> PartialVariableNodeByte;

/*!
  \ingroup Variable partielle
  \brief  Grandeur au noeud de type octet
*/
typedef MeshPartialVariableScalarRefT<Edge, Byte> PartialVariableEdgeByte;

/*!
  \ingroup Variable partielle
  \brief  Grandeur aux faces de type octet
*/
typedef MeshPartialVariableScalarRefT<Face, Byte> PartialVariableFaceByte;

/*!
  \ingroup Variable partielle
  \brief  Grandeur au centre des mailles de type octet
*/
typedef MeshPartialVariableScalarRefT<Cell, Byte> PartialVariableCellByte;

/*!
  \ingroup Variable partielle
  \brief  Grandeur particulaire de type octet
*/
typedef MeshPartialVariableScalarRefT<Particle, Byte> PartialVariableParticleByte;

/*!
  \ingroup Variable partielle
  \brief Grandeur de DDL de type octet
*/
typedef MeshPartialVariableScalarRefT<DoF, Byte> PartialVariableDoFByte;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
  \ingroup Variable partielle
  \brief  Grandeur au noeud de type booléen
*/
typedef MeshPartialVariableScalarRefT<Node, Byte> PartialVariableNodeBool;

/*!
  \ingroup Variable partielle
  \brief  Grandeur au noeud de type booléen
*/
typedef MeshPartialVariableScalarRefT<Edge, Byte> PartialVariableEdgeBool;

/*!
  \ingroup Variable partielle
  \brief  Grandeur aux faces de type booléen
*/
typedef MeshPartialVariableScalarRefT<Face, Byte> PartialVariableFaceBool;

/*!
  \ingroup Variable partielle
  \brief  Grandeur au centre des mailles de type booléen
*/
typedef MeshPartialVariableScalarRefT<Cell, Byte> PartialVariableCellBool;

/*!
  \ingroup Variable partielle
  \brief  Grandeur particulaire de type booléen
*/
typedef MeshPartialVariableScalarRefT<Particle, Byte> PartialVariableParticleBool;

/*!
  \ingroup Variable partielle
  \brief Grandeur de DDL de type booleen
*/
typedef MeshPartialVariableScalarRefT<DoF, Byte> PartialVariableDoFBool;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
  \ingroup Variable partielle
  \brief  Grandeur de type tableau de réels
*/
typedef ItemPartialVariableArrayRefT<Real> PartialVariableItemArrayReal;

/*!
  \ingroup Variable partielle
  \brief  Grandeur au noeud de type tableau de réels
*/
typedef MeshPartialVariableArrayRefT<Node, Real> PartialVariableNodeArrayReal;

/*!
  \ingroup Variable partielle
  \brief  Grandeur au noeud de type tableau de réels
*/
typedef MeshPartialVariableArrayRefT<Edge, Real> PartialVariableEdgeArrayReal;

/*!
  \ingroup Variable partielle
  \brief  Grandeur aux faces de type tableau de réels
*/
typedef MeshPartialVariableArrayRefT<Face, Real> PartialVariableFaceArrayReal;

/*!
  \ingroup Variable partielle
  \brief  Grandeur au centre des mailles de type tableau de réels
*/
typedef MeshPartialVariableArrayRefT<Cell, Real> PartialVariableCellArrayReal;

/*!
  \ingroup Variable partielle
  \brief  Grandeur particulaire de type tableau de réels
*/
typedef MeshPartialVariableArrayRefT<Particle, Real> PartialVariableParticleArrayReal;

/*!
  \ingroup Variable partielle
  \brief Grandeur de DDL de type tableau de reels
*/
typedef MeshPartialVariableArrayRefT<DoF, Real> PartialVariableDoFArrayReal;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
  \ingroup Variable partielle
  \brief  Grandeur de type tableau de coordonnées 2D
*/
typedef ItemPartialVariableArrayRefT<Real2> PartialVariableItemArrayReal2;

/*!
  \ingroup Variable partielle
  \brief  Grandeur au noeud de type tableau de coordonnées
*/
typedef MeshPartialVariableArrayRefT<Node, Real2> PartialVariableNodeArrayReal2;

/*!
  \ingroup Variable partielle
  \brief  Grandeur au noeud de type tableau de coordonnées
*/
typedef MeshPartialVariableArrayRefT<Edge, Real2> PartialVariableEdgeArrayReal2;

/*!
  \ingroup Variable partielle
  \brief  Grandeur aux faces de type tableau de coordonnées
*/
typedef MeshPartialVariableArrayRefT<Face, Real2> PartialVariableFaceArrayReal2;

/*!
  \ingroup Variable partielle
  \brief  Grandeur au centre des mailles de type tableau de coordonnées
*/
typedef MeshPartialVariableArrayRefT<Cell, Real2> PartialVariableCellArrayReal2;

/*!
  \ingroup Variable partielle
  \brief  Grandeur particulaire de type tableau de coordonnées
*/
typedef MeshPartialVariableArrayRefT<Particle, Real2> PartialVariableParticleArrayReal2;

/*!
  \ingroup Variable partielle
  \brief Grandeur de DDL de type tableau de coordonnees
*/
typedef MeshPartialVariableArrayRefT<DoF, Real2> PartialVariableDoFArrayReal2;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
  \ingroup Variable partielle
  \brief  Grandeur de type tableau de coordonnées 3D
*/
typedef ItemPartialVariableArrayRefT<Real3> PartialVariableItemArrayReal3;

/*!
  \ingroup Variable partielle
  \brief  Grandeur au noeud de type tableau de coordonnées
*/
typedef MeshPartialVariableArrayRefT<Node, Real3> PartialVariableNodeArrayReal3;

/*!
  \ingroup Variable partielle
  \brief  Grandeur au noeud de type tableau de coordonnées
*/
typedef MeshPartialVariableArrayRefT<Edge, Real3> PartialVariableEdgeArrayReal3;

/*!
  \ingroup Variable partielle
  \brief  Grandeur aux faces de type tableau de coordonnées
*/
typedef MeshPartialVariableArrayRefT<Face, Real3> PartialVariableFaceArrayReal3;

/*!
  \ingroup Variable partielle
  \brief  Grandeur au centre des mailles de type tableau de coordonnées
*/
typedef MeshPartialVariableArrayRefT<Cell, Real3> PartialVariableCellArrayReal3;

/*!
  \ingroup Variable partielle
  \brief  Grandeur particulaire de type tableau de coordonnées
*/
typedef MeshPartialVariableArrayRefT<Particle, Real3> PartialVariableParticleArrayReal3;

/*!
  \ingroup Variable partielle
  \brief Grandeur de DDL de type tableau de coordonnees
*/
typedef MeshPartialVariableArrayRefT<DoF, Real3> PartialVariableDoFArrayReal3;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
  \ingroup Variable partielle
  \brief  Grandeur de type tableau de tenseur de réels
*/
typedef ItemPartialVariableArrayRefT<Real2x2> PartialVariableItemArrayReal2x2;

/*!
  \ingroup Variable partielle
  \brief  Grandeur au noeud de type tableau de tenseur de réels
*/
typedef MeshPartialVariableArrayRefT<Node, Real2x2> PartialVariableNodeArrayReal2x2;

/*!
  \ingroup Variable partielle
  \brief  Grandeur au noeud de type tableau de tenseur de réels
*/
typedef MeshPartialVariableArrayRefT<Edge, Real2x2> PartialVariableEdgeArrayReal2x2;

/*!
  \ingroup Variable partielle
  \brief  Grandeur aux faces de type tableau de tenseur de réels
*/
typedef MeshPartialVariableArrayRefT<Face, Real2x2> PartialVariableFaceArrayReal2x2;

/*!
  \ingroup Variable partielle
  \brief  Grandeur au centre des mailles de type tableau de tenseur de réels
*/
typedef MeshPartialVariableArrayRefT<Cell, Real2x2> PartialVariableCellArrayReal2x2;

/*!
  \ingroup Variable partielle
  \brief  Grandeur particulaire de type tableau de tenseur de réels
*/
typedef MeshPartialVariableArrayRefT<Particle, Real2x2> PartialVariableParticleArrayReal2x2;

/*!
  \ingroup Variable partielle
  \brief Grandeur de DDL de type tableau de tenseur de reels
*/
typedef MeshPartialVariableArrayRefT<DoF, Real2x2> PartialVariableDoFArrayReal2x2;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
  \ingroup Variable partielle
  \brief  Grandeur de type tableau de tenseur de réels
*/
typedef ItemPartialVariableArrayRefT<Real3x3> PartialVariableItemArrayReal3x3;

/*!
  \ingroup Variable partielle
  \brief  Grandeur au noeud de type tableau de tenseur de réels
*/
typedef MeshPartialVariableArrayRefT<Node, Real3x3> PartialVariableNodeArrayReal3x3;

/*!
  \ingroup Variable partielle
  \brief  Grandeur au noeud de type tableau de tenseur de réels
*/
typedef MeshPartialVariableArrayRefT<Edge, Real3x3> PartialVariableEdgeArrayReal3x3;

/*!
  \ingroup Variable partielle
  \brief  Grandeur aux faces de type tableau de tenseur de réels
*/
typedef MeshPartialVariableArrayRefT<Face, Real3x3> PartialVariableFaceArrayReal3x3;

/*!
  \ingroup Variable partielle
  \brief  Grandeur au centre des mailles de type tableau de tenseur de réels
*/
typedef MeshPartialVariableArrayRefT<Cell, Real3x3> PartialVariableCellArrayReal3x3;

/*!
  \ingroup Variable partielle
  \brief  Grandeur particulaire de type tableau de tenseur de réels
*/
typedef MeshPartialVariableArrayRefT<Particle, Real3x3> PartialVariableParticleArrayReal3x3;

/*!
  \ingroup Variable partielle
  \brief Grandeur de DDL de type tableau de tenseur de reels
*/
typedef MeshPartialVariableArrayRefT<DoF, Real3x3> PartialVariableDoFArrayReal3x3;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
  \ingroup Variable partielle
  \brief  Grandeur de type tableau d'entiers
*/
typedef ItemPartialVariableArrayRefT<Integer> PartialVariableItemArrayInteger;

/*!
  \ingroup Variable partielle
  \brief  Grandeur au noeud de type tableau d'entiers
*/
typedef MeshPartialVariableArrayRefT<Node, Integer> PartialVariableNodeArrayInteger;

/*!
  \ingroup Variable partielle
  \brief  Grandeur au noeud de type tableau d'entiers
*/
typedef MeshPartialVariableArrayRefT<Edge, Integer> PartialVariableEdgeArrayInteger;

/*!
  \ingroup Variable partielle
  \brief  Grandeur aux faces de type tableau d'entiers
*/
typedef MeshPartialVariableArrayRefT<Face, Integer> PartialVariableFaceArrayInteger;

/*!
  \ingroup Variable partielle
  \brief  Grandeur au centre des mailles de type tableau d'entiers
*/
typedef MeshPartialVariableArrayRefT<Cell, Integer> PartialVariableCellArrayInteger;

/*!
  \ingroup Variable partielle
  \brief  Grandeur particulaire de type tableau d'entiers
*/
typedef MeshPartialVariableArrayRefT<Particle, Integer> PartialVariableParticleArrayInteger;

/*!
  \ingroup Variable partielle
  \brief Grandeur de DDL de type tableau d'entiers
*/
typedef MeshPartialVariableArrayRefT<DoF, Integer> PartialVariableDoFArrayInteger;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
  \ingroup Variable partielle
  \brief  Grandeur de type tableau d'entiers 16 bits
*/
typedef ItemPartialVariableArrayRefT<Int16> PartialVariableItemArrayInt16;

/*!
  \ingroup Variable partielle
  \brief  Grandeur au noeud de type tableau d'entiers 16 bits
*/
typedef MeshPartialVariableArrayRefT<Node, Int16> PartialVariableNodeArrayInt16;

/*!
  \ingroup Variable partielle
  \brief  Grandeur au noeud de type tableau d'entiers 16 bits
*/
typedef MeshPartialVariableArrayRefT<Edge, Int16> PartialVariableEdgeArrayInt16;

/*!
  \ingroup Variable partielle
  \brief  Grandeur aux faces de type tableau d'entiers 16 bits
*/
typedef MeshPartialVariableArrayRefT<Face, Int16> PartialVariableFaceArrayInt16;

/*!
  \ingroup Variable partielle
  \brief  Grandeur au centre des mailles de type tableau d'entiers 16 bits
*/
typedef MeshPartialVariableArrayRefT<Cell, Int16> PartialVariableCellArrayInt16;

/*!
  \ingroup Variable partielle
  \brief  Grandeur particulaire de type tableau d'entiers 16 bits
*/
typedef MeshPartialVariableArrayRefT<Particle, Int16> PartialVariableParticleArrayInt16;

/*!
  \ingroup Variable partielle
  \brief Grandeur de DDL de type tableau d'entiers 16 bits
*/
typedef MeshPartialVariableArrayRefT<DoF, Int16> PartialVariableDoFArrayInt16;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
  \ingroup Variable partielle
  \brief  Grandeur de type tableau d'entiers 32 bits
*/
typedef ItemPartialVariableArrayRefT<Int32> PartialVariableItemArrayInt32;

/*!
  \ingroup Variable partielle
  \brief  Grandeur au noeud de type tableau d'entiers 32 bits
*/
typedef MeshPartialVariableArrayRefT<Node, Int32> PartialVariableNodeArrayInt32;

/*!
  \ingroup Variable partielle
  \brief  Grandeur au noeud de type tableau d'entiers 32 bits
*/
typedef MeshPartialVariableArrayRefT<Edge, Int32> PartialVariableEdgeArrayInt32;

/*!
  \ingroup Variable partielle
  \brief  Grandeur aux faces de type tableau d'entiers 32 bits
*/
typedef MeshPartialVariableArrayRefT<Face, Int32> PartialVariableFaceArrayInt32;

/*!
  \ingroup Variable partielle
  \brief  Grandeur au centre des mailles de type tableau d'entiers 32 bits
*/
typedef MeshPartialVariableArrayRefT<Cell, Int32> PartialVariableCellArrayInt32;

/*!
  \ingroup Variable partielle
  \brief  Grandeur particulaire de type tableau d'entiers 32 bits
*/
typedef MeshPartialVariableArrayRefT<Particle, Int32> PartialVariableParticleArrayInt32;

/*!
  \ingroup Variable partielle
  \brief Grandeur de DDL de type tableau d'entiers 32 bits
*/
typedef MeshPartialVariableArrayRefT<DoF, Int32> PartialVariableDoFArrayInt32;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
  \ingroup Variable
  \brief  Grandeur de type tableau d'entiers 64 bits
*/
typedef ItemPartialVariableArrayRefT<Int64> PartialVariableItemArrayInt64;

/*!
  \ingroup Variable partielle
  \brief  Grandeur au noeud de type tableau d'entiers 64 bits
*/
typedef MeshPartialVariableArrayRefT<Node, Int64> PartialVariableNodeArrayInt64;

/*!
  \ingroup Variable partielle
  \brief  Grandeur au noeud de type tableau d'entiers 64 bits
*/
typedef MeshPartialVariableArrayRefT<Edge, Int64> PartialVariableEdgeArrayInt64;

/*!
  \ingroup Variable partielle
  \brief  Grandeur aux faces de type tableau d'entiers 64 bits
*/
typedef MeshPartialVariableArrayRefT<Face, Int64> PartialVariableFaceArrayInt64;

/*!
  \ingroup Variable partielle
  \brief  Grandeur au centre des mailles de type tableau d'entiers 64 bits
*/
typedef MeshPartialVariableArrayRefT<Cell, Int64> PartialVariableCellArrayInt64;

/*!
  \ingroup Variable partielle
  \brief  Grandeur particulaire de type tableau d'entiers 64 bits
*/
typedef MeshPartialVariableArrayRefT<Particle, Int64> PartialVariableParticleArrayInt64;

/*!
  \ingroup Variable partielle
  \brief Grandeur de DDL de type tableau d'entiers 64 bits
*/
typedef MeshPartialVariableArrayRefT<DoF, Int64> PartialVariableDoFArrayInt64;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
  \ingroup Variable partielle
  \brief  Grandeur de type tableau d'entiers 64 bits
*/
typedef ItemPartialVariableArrayRefT<Byte> PartialVariableItemArrayByte;

/*!
  \ingroup Variable partielle
  \brief  Grandeur au noeud de type tableau d'octets
*/
typedef MeshPartialVariableArrayRefT<Node, Byte> PartialVariableNodeArrayByte;

/*!
  \ingroup Variable partielle
  \brief  Grandeur au noeud de type tableau d'octets
*/
typedef MeshPartialVariableArrayRefT<Edge, Byte> PartialVariableEdgeArrayByte;

/*!
  \ingroup Variable partielle
  \brief  Grandeur aux faces de type tableau d'octets
*/
typedef MeshPartialVariableArrayRefT<Face, Byte> PartialVariableFaceArrayByte;

/*!
  \ingroup Variable partielle
  \brief  Grandeur au centre des mailles de type tableau d'octets
*/
typedef MeshPartialVariableArrayRefT<Cell, Byte> PartialVariableCellArrayByte;

/*!
  \ingroup Variable partielle
  \brief  Grandeur particulaire de type tableau d'octets
*/
typedef MeshPartialVariableArrayRefT<Particle, Byte> PartialVariableParticleArrayByte;

/*!
  \ingroup Variable partielle
  \brief Grandeur de DDL de type tableau d'octets
*/
typedef MeshPartialVariableArrayRefT<DoF, Byte> PartialVariableDoFArrayByte;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
  \ingroup Variable partielle
  \brief  Grandeur au noeud de type tableau de booléens
*/
typedef MeshPartialVariableArrayRefT<Node, Byte> PartialVariableNodeArrayBool;

/*!
  \ingroup Variable partielle
  \brief  Grandeur au noeud de type tableau de booléens
*/
typedef MeshPartialVariableArrayRefT<Edge, Byte> PartialVariableEdgeArrayBool;

/*!
  \ingroup Variable partielle
  \brief  Grandeur aux faces de type tableau de booléens
*/
typedef MeshPartialVariableArrayRefT<Face, Byte> PartialVariableFaceArrayBool;

/*!
  \ingroup Variable partielle
  \brief  Grandeur au centre des mailles de type tableau de booléens
*/
typedef MeshPartialVariableArrayRefT<Cell, Byte> PartialVariableCellArrayBool;

/*!
  \ingroup Variable partielle
  \brief  Grandeur particulaire de type tableau de booléens
*/
typedef MeshPartialVariableArrayRefT<Particle, Byte> PartialVariableParticleArrayBool;

/*!
  \ingroup Variable partielle
  \brief Grandeur de DDL de type tableau de booleens
*/
typedef MeshPartialVariableArrayRefT<DoF, Byte> PartialVariableDoFArrayBool;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename ItemType, typename DataType> class SharedMeshVariableScalarRefT;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
  \ingroup Variable
  \brief Grandeur au noeud de type réel
*/
typedef SharedMeshVariableScalarRefT<Node, Real> SharedVariableNodeReal;

/*!
  \ingroup SharedVariable
  \brief Grandeur au noeud de type réel
*/
typedef SharedMeshVariableScalarRefT<Edge, Real> SharedVariableEdgeReal;

/*!
  \ingroup SharedVariable
  \brief Grandeur aux faces de type réel
*/
typedef SharedMeshVariableScalarRefT<Face, Real> SharedVariableFaceReal;

/*!
  \ingroup SharedVariable
  \brief Grandeur au centre des mailles de type réel
*/
typedef SharedMeshVariableScalarRefT<Cell, Real> SharedVariableCellReal;

/*!
  \ingroup SharedVariable
  \brief Grandeur particulaire de type réel
*/
typedef SharedMeshVariableScalarRefT<Particle, Real> SharedVariableParticleReal;

/*!
  \ingroup SharedVariable
  \brief Grandeur de DDL de type reel
*/
typedef SharedMeshVariableScalarRefT<DoF, Real> SharedVariableDoFReal;

/*!
  \ingroup SharedVariable
  \brief Grandeur au centre des mailles de type rel
*/
typedef SharedMeshVariableScalarRefT<Cell, Real> SharedVariableCellReal;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
  \ingroup SharedVariable
  \brief Grandeur au noeud de type coordonnées
*/
typedef SharedMeshVariableScalarRefT<Node, Real2> SharedVariableNodeReal2;

/*!
  \ingroup SharedVariable
  \brief Grandeur au noeud de type coordonnées
*/
typedef SharedMeshVariableScalarRefT<Edge, Real2> SharedVariableEdgeReal2;

/*!
  \ingroup SharedVariable
  \brief Grandeur aux faces de type coordonnées
*/
typedef SharedMeshVariableScalarRefT<Face, Real2> SharedVariableFaceReal2;

/*!
  \ingroup SharedVariable
  \brief Grandeur au centre des mailles de type coordonnées
*/
typedef SharedMeshVariableScalarRefT<Cell, Real2> SharedVariableCellReal2;

/*!
  \ingroup SharedVariable
  \brief Grandeur particulaire de type coordonnées
*/
typedef SharedMeshVariableScalarRefT<Particle, Real2> SharedVariableParticleReal2;

/*!
  \ingroup SharedVariable
  \brief Grandeur de DDL de type coordonnees
*/
typedef SharedMeshVariableScalarRefT<DoF, Real2> SharedVariableDoFReal2;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
  \ingroup SharedVariable
  \brief Grandeur de type coordonn?es 3D
*/
typedef SharedItemVariableScalarRefT<Real3> SharedVariableItemReal3;

/*!
  \ingroup SharedVariable
  \brief Grandeur au noeud de type coordonnées
*/
typedef SharedMeshVariableScalarRefT<Node, Real3> SharedVariableNodeReal3;

/*!
  \ingroup SharedVariable
  \brief Grandeur au noeud de type coordonnées
*/
typedef SharedMeshVariableScalarRefT<Edge, Real3> SharedVariableEdgeReal3;

/*!
  \ingroup SharedVariable
  \brief Grandeur aux faces de type coordonnées
*/
typedef SharedMeshVariableScalarRefT<Face, Real3> SharedVariableFaceReal3;

/*!
  \ingroup SharedVariable
  \brief Grandeur au centre des mailles de type coordonnées
*/
typedef SharedMeshVariableScalarRefT<Cell, Real3> SharedVariableCellReal3;

/*!
  \ingroup SharedVariable
  \brief Grandeur particulaire de type coordonnées
*/
typedef SharedMeshVariableScalarRefT<Particle, Real3> SharedVariableParticleReal3;

/*!
  \ingroup SharedVariable
  \brief Grandeur de DDL de type coordonnees
*/
typedef SharedMeshVariableScalarRefT<DoF, Real3> SharedVariableDoFReal3;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
  \ingroup SharedVariable
  \brief Grandeur de type tenseur de r?els
*/
typedef SharedItemVariableScalarRefT<Real2x2> SharedVariableItemReal2x2;

/*!
  \ingroup SharedVariable
  \brief Grandeur au noeud de type tenseur de réels
*/
typedef SharedMeshVariableScalarRefT<Node, Real2x2> SharedVariableNodeReal2x2;

/*!
  \ingroup SharedVariable
  \brief Grandeur au noeud de type tenseur de réels
*/
typedef SharedMeshVariableScalarRefT<Edge, Real2x2> SharedVariableEdgeReal2x2;

/*!
  \ingroup SharedVariable
  \brief Grandeur aux faces de type tenseur de réels
*/
typedef SharedMeshVariableScalarRefT<Face, Real2x2> SharedVariableFaceReal2x2;

/*!
  \ingroup SharedVariable
  \brief Grandeur au centre des mailles de type tenseur de réels
*/
typedef SharedMeshVariableScalarRefT<Cell, Real2x2> SharedVariableCellReal2x2;

/*!
  \ingroup SharedVariable
  \brief Grandeur particulaire de type tenseur de réels
*/
typedef SharedMeshVariableScalarRefT<Particle, Real2x2> SharedVariableParticleReal2x2;

/*!
  \ingroup SharedVariable
  \brief Grandeur de DDL de type tenseur de reels
*/
typedef SharedMeshVariableScalarRefT<DoF, Real2x2> SharedVariableDoFReal2x2;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
  \ingroup SharedVariable
  \brief Grandeur de type tenseur de r?els
*/
typedef SharedItemVariableScalarRefT<Real3x3> SharedVariableItemReal3x3;

/*!
  \ingroup SharedVariable
  \brief Grandeur au noeud de type tenseur de réels
*/
typedef SharedMeshVariableScalarRefT<Node, Real3x3> SharedVariableNodeReal3x3;

/*!
  \ingroup SharedVariable
  \brief Grandeur au noeud de type tenseur de réels
*/
typedef SharedMeshVariableScalarRefT<Edge, Real3x3> SharedVariableEdgeReal3x3;

/*!
  \ingroup SharedVariable
  \brief Grandeur aux faces de type tenseur de réels
*/
typedef SharedMeshVariableScalarRefT<Face, Real3x3> SharedVariableFaceReal3x3;

/*!
  \ingroup SharedVariable
  \brief Grandeur au centre des mailles de type tenseur de réels
*/
typedef SharedMeshVariableScalarRefT<Cell, Real3x3> SharedVariableCellReal3x3;

/*!
  \ingroup SharedVariable
  \brief Grandeur particulaire de type tenseur de réels
*/
typedef SharedMeshVariableScalarRefT<Particle, Real3x3> SharedVariableParticleReal3x3;

/*!
  \ingroup SharedVariable
  \brief Grandeur de DDL de type tenseur de reels
*/
typedef SharedMeshVariableScalarRefT<DoF, Real3x3> SharedVariableDoFReal3x3;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
  \ingroup SharedVariable
  \brief Grandeur de type entier
*/
typedef SharedItemVariableScalarRefT<Integer> SharedVariableItemInteger;

/*!
  \ingroup SharedVariable
  \brief Grandeur au noeud de type entier
*/
typedef SharedMeshVariableScalarRefT<Node, Integer> SharedVariableNodeInteger;

/*!
  \ingroup SharedVariable
  \brief Grandeur au noeud de type entier
*/
typedef SharedMeshVariableScalarRefT<Edge, Integer> SharedVariableEdgeInteger;

/*!
  \ingroup SharedVariable
  \brief Grandeur aux faces de type entier
*/
typedef SharedMeshVariableScalarRefT<Face, Integer> SharedVariableFaceInteger;

/*!
  \ingroup SharedVariable
  \brief Grandeur au centre des mailles de type entier
*/
typedef SharedMeshVariableScalarRefT<Cell, Integer> SharedVariableCellInteger;

/*!
  \ingroup SharedVariable
  \brief Grandeur particulaire de type entier
*/
typedef SharedMeshVariableScalarRefT<Particle, Integer> SharedVariableParticleInteger;

/*!
  \ingroup SharedVariable
  \brief Grandeur de DDL de type entier
*/
typedef SharedMeshVariableScalarRefT<DoF, Integer> SharedVariableDoFInteger;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
  \ingroup SharedVariable
  \brief Grandeur de type entier 32 bits
*/
typedef SharedItemVariableScalarRefT<Int32> SharedVariableItemInt32;

/*!
  \ingroup SharedVariable
  \brief Grandeur au noeud de type entier 32 bits
*/
typedef SharedMeshVariableScalarRefT<Node, Int32> SharedVariableNodeInt32;

/*!
  \ingroup SharedVariable
  \brief  Grandeur au noeud de type entier 32 bits
*/
typedef SharedMeshVariableScalarRefT<Edge, Int32> SharedVariableEdgeInt32;

/*!
  \ingroup SharedVariable
  \brief  Grandeur aux faces de type entier 32 bits
*/
typedef SharedMeshVariableScalarRefT<Face, Int32> SharedVariableFaceInt32;

/*!
  \ingroup SharedVariable
  \brief  Grandeur au centre des mailles de type entier 32 bits
*/
typedef SharedMeshVariableScalarRefT<Cell, Int32> SharedVariableCellInt32;

/*!
  \ingroup SharedVariable
  \brief  Grandeur particulaire de type entier 32 bits
*/
typedef SharedMeshVariableScalarRefT<Particle, Int32> SharedVariableParticleInt32;

/*!
  \ingroup SharedVariable
  \brief Grandeur de DDL de type entier 32 bits
*/
typedef SharedMeshVariableScalarRefT<DoF, Int32> SharedVariableDoFInt32;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
  \ingroup SharedVariable
  \brief Grandeur de type entier 16 bits
*/
typedef SharedItemVariableScalarRefT<Int16> SharedVariableItemInt16;

/*!
  \ingroup SharedVariable
  \brief Grandeur au noeud de type entier 16 bits
*/
typedef SharedMeshVariableScalarRefT<Node, Int16> SharedVariableNodeInt16;

/*!
  \ingroup SharedVariable
  \brief  Grandeur au noeud de type entier 16 bits
*/
typedef SharedMeshVariableScalarRefT<Edge, Int16> SharedVariableEdgeInt16;

/*!
  \ingroup SharedVariable
  \brief  Grandeur aux faces de type entier 16 bits
*/
typedef SharedMeshVariableScalarRefT<Face, Int16> SharedVariableFaceInt16;

/*!
  \ingroup SharedVariable
  \brief  Grandeur au centre des mailles de type entier 16 bits
*/
typedef SharedMeshVariableScalarRefT<Cell, Int16> SharedVariableCellInt16;

/*!
  \ingroup SharedVariable
  \brief  Grandeur particulaire de type entier 16 bits
*/
typedef SharedMeshVariableScalarRefT<Particle, Int16> SharedVariableParticleInt16;

/*!
  \ingroup SharedVariable
  \brief Grandeur de DDL de type entier 16 bits
*/
typedef SharedMeshVariableScalarRefT<DoF, Int16> SharedVariableDoFInt16;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
  \ingroup SharedVariable
  \brief  Grandeur de type entier 64 bits
*/
typedef SharedItemVariableScalarRefT<Int64> SharedVariableItemInt64;

/*!
  \ingroup SharedVariable
  \brief  Grandeur au noeud de type entier 64 bits
*/
typedef SharedMeshVariableScalarRefT<Node, Int64> SharedVariableNodeInt64;

/*!
  \ingroup SharedVariable
  \brief  Grandeur au noeud de type entier 64 bits
*/
typedef SharedMeshVariableScalarRefT<Edge, Int64> SharedVariableEdgeInt64;

/*!
  \ingroup SharedVariable
  \brief  Grandeur aux faces de type entier 64 bits
*/
typedef SharedMeshVariableScalarRefT<Face, Int64> SharedVariableFaceInt64;

/*!
  \ingroup SharedVariable
  \brief  Grandeur au centre des mailles de type entier 64 bits
*/
typedef SharedMeshVariableScalarRefT<Cell, Int64> SharedVariableCellInt64;

/*!
  \ingroup SharedVariable
  \brief  Grandeur particulaire de type entier 64 bits
*/
typedef SharedMeshVariableScalarRefT<Particle, Int64> SharedVariableParticleInt64;

/*!
  \ingroup SharedVariable
  \brief Grandeur de DDL de type entier 64 bits
*/
typedef SharedMeshVariableScalarRefT<DoF, Int64> SharedVariableDoFInt64;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
  \ingroup SharedVariable
  \brief  Grandeur au noeud de type entier naturel
  \deprecated Utiliser #VariableNodeInteger à la place
*/
typedef SharedMeshVariableScalarRefT<Node, Integer> SharedVariableNodeInteger;

/*!
  \ingroup SharedVariable
  \brief  Grandeur aux faces de type entier naturel
  \deprecated Utiliser #VariableFaceInteger à la place
*/
typedef SharedMeshVariableScalarRefT<Face, Integer> SharedVariableFaceInteger;

/*!
  \ingroup SharedVariable
  \brief  Grandeur au centre des mailles de type entier naturel
  \deprecated Utiliser #VariableCellInteger à la place
*/
typedef SharedMeshVariableScalarRefT<Cell, Integer> SharedVariableCellInteger;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
  \ingroup SharedVariable
  \brief  Grandeur de type entier 64 bits
*/
typedef ItemPartialVariableScalarRefT<Byte> PartialVariableItemByte;

/*!
  \ingroup SharedVariable
  \brief  Grandeur de type entier 64 bits
*/
typedef SharedItemVariableScalarRefT<Byte> SharedVariableItemByte;

/*!
  \ingroup SharedVariable
  \brief  Grandeur au noeud de type octet
*/
typedef SharedMeshVariableScalarRefT<Node, Byte> SharedVariableNodeByte;

/*!
  \ingroup SharedVariable
  \brief  Grandeur au noeud de type octet
*/
typedef SharedMeshVariableScalarRefT<Edge, Byte> SharedVariableEdgeByte;

/*!
  \ingroup SharedVariable
  \brief  Grandeur aux faces de type octet
*/
typedef SharedMeshVariableScalarRefT<Face, Byte> SharedVariableFaceByte;

/*!
  \ingroup SharedVariable
  \brief  Grandeur au centre des mailles de type octet
*/
typedef SharedMeshVariableScalarRefT<Cell, Byte> SharedVariableCellByte;

/*!
  \ingroup SharedVariable
  \brief  Grandeur particulaire de type octet
*/
typedef SharedMeshVariableScalarRefT<Particle, Byte> SharedVariableParticleByte;

/*!
  \ingroup SharedVariable
  \brief Grandeur de DDL de type octet
*/
typedef SharedMeshVariableScalarRefT<DoF, Byte> SharedVariableDoFByte;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
  \ingroup SharedVariable
  \brief  Grandeur au noeud de type booléen
*/
typedef SharedMeshVariableScalarRefT<Node, Byte> SharedVariableNodeBool;

/*!
  \ingroup SharedVariable
  \brief  Grandeur au noeud de type booléen
*/
typedef SharedMeshVariableScalarRefT<Edge, Byte> SharedVariableEdgeBool;

/*!
  \ingroup SharedVariable
  \brief  Grandeur aux faces de type booléen
*/
typedef SharedMeshVariableScalarRefT<Face, Byte> SharedVariableFaceBool;

/*!
  \ingroup SharedVariable
  \brief  Grandeur au centre des mailles de type booléen
*/
typedef SharedMeshVariableScalarRefT<Cell, Byte> SharedVariableCellBool;

/*!
  \ingroup SharedVariable
  \brief  Grandeur particulaire de type booléen
*/
typedef SharedMeshVariableScalarRefT<Particle, Byte> SharedVariableParticleBool;

/*!
  \ingroup SharedVariable
  \brief Grandeur de DDL de type booleen
*/
typedef SharedMeshVariableScalarRefT<DoF, Byte> SharedVariableDoFBool;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
