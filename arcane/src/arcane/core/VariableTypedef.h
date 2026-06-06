// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VariableTypedef.h                                           (C) 2000-2025 */
/*                                                                           */
/* Declarations of variable typedefs.                                        */
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
 * Declarations of variable types.
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
  \brief Scalar variable of byte type.
*/
typedef VariableRefScalarT<Byte> VariableScalarByte;

/*!
  \ingroup Variable
  \brief Scalar variable of real type
*/
typedef VariableRefScalarT<Real> VariableScalarReal;

/*!
  \ingroup Variable
  \brief Scalar variable of 16-bit integer type
*/
typedef VariableRefScalarT<Int16> VariableScalarInt16;

/*!
  \ingroup Variable
  \brief Scalar variable of 32-bit integer type
*/
typedef VariableRefScalarT<Int32> VariableScalarInt32;

/*!
  \ingroup Variable
  \brief Scalar variable of 64-bit integer type
*/
typedef VariableRefScalarT<Int64> VariableScalarInt64;

/*!
  \ingroup Variable
  \brief Scalar variable of integer type
*/
typedef VariableRefScalarT<Integer> VariableScalarInteger;

/*!
  \ingroup Variable
  \brief Scalar variable of dimension type
  \deprecated Use #VariableScalarInteger instead
*/
typedef VariableRefScalarT<Integer> VariableScalarInteger;

/*!
  \ingroup Variable
  \brief Scalar variable of character string type
*/
typedef VariableRefScalarT<String> VariableScalarString;

/*!
  \ingroup Variable
  \brief Scalar variable of coordinate type (x,y,z)
*/
typedef VariableRefScalarT<Real3> VariableScalarReal3;

/*!
  \ingroup Variable
  \brief Scalar variable of 3x3 tensor type
*/
typedef VariableRefScalarT<Real3x3> VariableScalarReal3x3;

/*!
  \ingroup Variable
  \brief Scalar variable of coordinate type (x,y)
*/
typedef VariableRefScalarT<Real2> VariableScalarReal2;

/*!
  \ingroup Variable
  \brief Scalar variable of 2x2 tensor type
*/
typedef VariableRefScalarT<Real2x2> VariableScalarReal2x2;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
  \ingroup Variable
  \brief Array variable of byte type
*/
typedef VariableRefArrayT<Byte> VariableArrayByte;

/*!
  \ingroup Variable
  \brief Array variable of real type
*/
typedef VariableRefArrayT<Real> VariableArrayReal;

/*!
  \ingroup Variable
  \brief Array variable of 16-bit integer type
*/
typedef VariableRefArrayT<Int16> VariableArrayInt16;

/*!
  \ingroup Variable
  \brief Array variable of 32-bit integer type
*/
typedef VariableRefArrayT<Int32> VariableArrayInt32;

/*!
  \ingroup Variable
  \brief Array variable of 64-bit integer type
*/
typedef VariableRefArrayT<Int64> VariableArrayInt64;

/*!
  \ingroup Variable
  \brief Array variable of integer type
*/
typedef VariableRefArrayT<Integer> VariableArrayInteger;

/*!
  \ingroup Variable
  \brief Array variable of index type
  \deprecated Use #VariableArrayInteger instead
*/
typedef VariableRefArrayT<Integer> VariableArrayInteger;

/*!
  \ingroup Variable
  \brief Array variable of character string type
*/
typedef VariableRefArrayT<String> VariableArrayString;

/*!
  \ingroup Variable
  \brief Array variable of coordinate type (x,y,z)
*/
typedef VariableRefArrayT<Real3> VariableArrayReal3;

/*!
  \ingroup Variable
  \brief Array variable of real tensor type
*/
typedef VariableRefArrayT<Real3x3> VariableArrayReal3x3;

/*!
  \ingroup Variable
  \brief Array variable of coordinate type (x,y)
*/
typedef VariableRefArrayT<Real2> VariableArrayReal2;

/*!
  \ingroup Variable
  \brief Array variable of 2x2 tensor type
*/
typedef VariableRefArrayT<Real2x2> VariableArrayReal2x2;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
  \ingroup Variable
  \brief Two-dimensional array variable of byte type
*/
typedef VariableRefArray2T<Byte> VariableArray2Byte;

/*!
  \ingroup Variable
  \brief Two-dimensional array variable of real type
*/
typedef VariableRefArray2T<Real> VariableArray2Real;

/*!
  \ingroup Variable
  \brief Two-dimensional array variable of integer type
*/
typedef VariableRefArray2T<Integer> VariableArray2Integer;

/*!
  \ingroup Variable
  \brief Two-dimensional array variable of 16-bit integer type
*/
typedef VariableRefArray2T<Int16> VariableArray2Int16;

/*!
  \ingroup Variable
  \brief Two-dimensional array variable of 32-bit integer type
*/
typedef VariableRefArray2T<Int32> VariableArray2Int32;

/*!
  \ingroup Variable
  \brief Two-dimensional array variable of 64-bit integer type
*/
typedef VariableRefArray2T<Int64> VariableArray2Int64;

/*!
  \ingroup Variable
  \brief Two-dimensional array variable of coordinate type (x,y,z)
*/
typedef VariableRefArray2T<Real3> VariableArray2Real3;

/*!
  \ingroup Variable
  \brief Two-dimensional array variable of 3x3 real tensor type
*/
typedef VariableRefArray2T<Real3x3> VariableArray2Real3x3;

/*!
  \ingroup Variable
  \brief Two-dimensional array variable of coordinate type (x,y)
*/
typedef VariableRefArray2T<Real2> VariableArray2Real2;

/*!
  \ingroup Variable
  \brief Two-dimensional array variable of 3x3 real tensor type
*/
typedef VariableRefArray2T<Real2x2> VariableArray2Real2x2;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
  \ingroup Variable
  \brief Real type quantity
*/
typedef ItemVariableScalarRefT<Real> VariableItemReal;

/*!
  \ingroup Variable
  \brief Real type quantity at node
*/
typedef MeshVariableScalarRefT<Node, Real> VariableNodeReal;

/*!
  \ingroup Variable
  \brief Real type quantity at node
*/
typedef MeshVariableScalarRefT<Edge, Real> VariableEdgeReal;

/*!
  \ingroup Variable
  \brief Real type quantity at face
*/
typedef MeshVariableScalarRefT<Face, Real> VariableFaceReal;

/*!
  \ingroup Variable
  \brief Real type quantity at cell center
*/
typedef MeshVariableScalarRefT<Cell, Real> VariableCellReal;

/*!
  \ingroup Variable
  \brief Real type particle quantity
*/
typedef MeshVariableScalarRefT<Particle, Real> VariableParticleReal;

/*!
  \ingroup Variable
  \brief Real type DoF quantity
*/
typedef MeshVariableScalarRefT<DoF, Real> VariableDoFReal;
/*!
  \ingroup Variable
  \brief Real type quantity at cell center
*/
typedef MeshVariableScalarRefT<Cell, Real> VariableCellReal;
/*!
  \ingroup Variable
  \brief 2D coordinate type quantity
*/
typedef ItemVariableScalarRefT<Real2> VariableItemReal2;

/*!
  \ingroup Variable
  \brief Coordinate type quantity at node
*/
typedef MeshVariableScalarRefT<Node, Real2> VariableNodeReal2;

/*!
  \ingroup Variable
  \brief Coordinate type quantity at node
*/
typedef MeshVariableScalarRefT<Edge, Real2> VariableEdgeReal2;

/*!
  \ingroup Variable
  \brief Coordinate type quantity at face
*/
typedef MeshVariableScalarRefT<Face, Real2> VariableFaceReal2;

/*!
  \ingroup Variable
  \brief Coordinate type quantity at cell center
*/
typedef MeshVariableScalarRefT<Cell, Real2> VariableCellReal2;

/*!
  \ingroup Variable
  \brief Coordinate type particle quantity
*/
typedef MeshVariableScalarRefT<Particle, Real2> VariableParticleReal2;

/*!
  \ingroup Variable
  \brief Coordinate type DoF quantity
*/
typedef MeshVariableScalarRefT<DoF, Real2> VariableDoFReal2;
/*!
  \ingroup Variable
  \brief 3D coordinate type quantity
*/
typedef ItemVariableScalarRefT<Real3> VariableItemReal3;

/*!
  \ingroup Variable
  \brief Coordinate type quantity at node
*/
typedef MeshVariableScalarRefT<Node, Real3> VariableNodeReal3;

/*!
  \ingroup Variable
  \brief Coordinate type quantity at node
*/
typedef MeshVariableScalarRefT<Edge, Real3> VariableEdgeReal3;

/*!
  \ingroup Variable
  \brief Coordinate type quantity at face
*/
typedef MeshVariableScalarRefT<Face, Real3> VariableFaceReal3;

/*!
  \ingroup Variable
  \brief Coordinate type quantity at cell center
*/
typedef MeshVariableScalarRefT<Cell, Real3> VariableCellReal3;

/*!
  \ingroup Variable
  \brief Coordinate type particle quantity
*/
typedef MeshVariableScalarRefT<Particle, Real3> VariableParticleReal3;

/*!
  \ingroup Variable
  \brief Coordinate type DoF quantity
*/
typedef MeshVariableScalarRefT<DoF, Real3> VariableDoFReal3;
/*!
  \ingroup Variable
  \brief Real tensor type quantity
*/
typedef ItemVariableScalarRefT<Real2x2> VariableItemReal2x2;

/*!
  \ingroup Variable
  \brief Real tensor type quantity at node
*/
typedef MeshVariableScalarRefT<Node, Real2x2> VariableNodeReal2x2;

/*!
  \ingroup Variable
  \brief Real tensor type quantity at node
*/
typedef MeshVariableScalarRefT<Edge, Real2x2> VariableEdgeReal2x2;

/*!
  \ingroup Variable
  \brief Real tensor type quantity at face
*/
typedef MeshVariableScalarRefT<Face, Real2x2> VariableFaceReal2x2;

/*!
  \ingroup Variable
  \brief Real tensor type quantity at cell center
*/
typedef MeshVariableScalarRefT<Cell, Real2x2> VariableCellReal2x2;

/*!
  \ingroup Variable
  \brief Real tensor type particle quantity
*/
typedef MeshVariableScalarRefT<Particle, Real2x2> VariableParticleReal2x2;

/*!
  \ingroup Variable
  \brief Real tensor type DoF quantity
*/
typedef MeshVariableScalarRefT<DoF, Real2x2> VariableDoFReal2x2;
/*!
  \ingroup Variable
\brief Quantity of real tensor type
*/
typedef ItemVariableScalarRefT<Real3x3> VariableItemReal3x3;

/*!
  \ingroup Variable
  \brief Quantity at the node of real tensor type
*/
typedef MeshVariableScalarRefT<Node, Real3x3> VariableNodeReal3x3;

/*!
  \ingroup Variable
  \brief Quantity at the node of real tensor type
*/
typedef MeshVariableScalarRefT<Edge, Real3x3> VariableEdgeReal3x3;

/*!
  \ingroup Variable
  \brief Quantity at the face of real tensor type
*/
typedef MeshVariableScalarRefT<Face, Real3x3> VariableFaceReal3x3;

/*!
  \ingroup Variable
  \brief Quantity at the cell center of real tensor type
*/
typedef MeshVariableScalarRefT<Cell, Real3x3> VariableCellReal3x3;

/*!
  \ingroup Variable
  \brief Particle quantity of real tensor type
*/
typedef MeshVariableScalarRefT<Particle, Real3x3> VariableParticleReal3x3;

/*!
  \ingroup Variable
  \brief Degree of Freedom quantity of real tensor type
*/
typedef MeshVariableScalarRefT<DoF, Real3x3> VariableDoFReal3x3;
/*!
  \ingroup Variable
  \brief Integer type quantity
*/
typedef ItemVariableScalarRefT<Integer> VariableItemInteger;

/*!
  \ingroup Variable
  \brief Quantity at the node of integer type
*/
typedef MeshVariableScalarRefT<Node, Integer> VariableNodeInteger;

/*!
  \ingroup Variable
  \brief Quantity at the node of integer type
*/
typedef MeshVariableScalarRefT<Edge, Integer> VariableEdgeInteger;

/*!
  \ingroup Variable
  \brief Quantity at the face of integer type
*/
typedef MeshVariableScalarRefT<Face, Integer> VariableFaceInteger;

/*!
  \ingroup Variable
  \brief Quantity at the cell center of integer type
*/
typedef MeshVariableScalarRefT<Cell, Integer> VariableCellInteger;

/*!
  \ingroup Variable
  \brief Particle quantity of integer type
*/
typedef MeshVariableScalarRefT<Particle, Integer> VariableParticleInteger;

/*!
  \ingroup Variable
  \brief Degree of Freedom quantity of integer type
*/
typedef MeshVariableScalarRefT<DoF, Integer> VariableDoFInteger;
/*!
  \ingroup Variable
  \brief 32-bit integer type quantity
*/
typedef ItemVariableScalarRefT<Int16> VariableItemInt16;
/*!
  \ingroup Variable
  \brief 32-bit integer type quantity
*/
typedef ItemVariableScalarRefT<Int32> VariableItemInt32;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
  \ingroup Variable
  \brief Quantity at the node of 16-bit integer type
*/
typedef MeshVariableScalarRefT<Node, Int16> VariableNodeInt16;

/*!
  \ingroup Variable
  \brief Quantity at the node of 16-bit integer type
*/
typedef MeshVariableScalarRefT<Edge, Int16> VariableEdgeInt16;

/*!
  \ingroup Variable
  \brief Quantity at the face of 16-bit integer type
*/
typedef MeshVariableScalarRefT<Face, Int16> VariableFaceInt16;

/*!
  \ingroup Variable
  \brief Quantity at the cell center of 16-bit integer type
*/
typedef MeshVariableScalarRefT<Cell, Int16> VariableCellInt16;

/*!
  \ingroup Variable
  \brief Particle quantity of 16-bit integer type
*/
typedef MeshVariableScalarRefT<Particle, Int16> VariableParticleInt16;

/*!
  \ingroup Variable
  \brief Degree of Freedom quantity of 16-bit integer type
*/
typedef MeshVariableScalarRefT<DoF, Int16> VariableDoFInt16;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
  \ingroup Variable
  \brief Quantity at the node of 32-bit integer type
*/
typedef MeshVariableScalarRefT<Node, Int32> VariableNodeInt32;

/*!
  \ingroup Variable
  \brief Quantity at the node of 32-bit integer type
*/
typedef MeshVariableScalarRefT<Edge, Int32> VariableEdgeInt32;

/*!
  \ingroup Variable
  \brief Quantity at the face of 32-bit integer type
*/
typedef MeshVariableScalarRefT<Face, Int32> VariableFaceInt32;

/*!
  \ingroup Variable
  \brief Quantity at the cell center of 32-bit integer type
*/
typedef MeshVariableScalarRefT<Cell, Int32> VariableCellInt32;

/*!
  \ingroup Variable
  \brief Particle quantity of 32-bit integer type
*/
typedef MeshVariableScalarRefT<Particle, Int32> VariableParticleInt32;

/*!
  \ingroup Variable
  \brief Degree of Freedom quantity of 32-bit integer type
*/
typedef MeshVariableScalarRefT<DoF, Int32> VariableDoFInt32;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
  \ingroup Variable
  \brief Quantity of 64-bit integer type
*/
typedef ItemVariableScalarRefT<Int64> VariableItemInt64;

/*!
  \ingroup Variable
  \brief Quantity at the node of 64-bit integer type
*/
typedef MeshVariableScalarRefT<Node, Int64> VariableNodeInt64;

/*!
  \ingroup Variable
  \brief Quantity at the node of 64-bit integer type
*/
typedef MeshVariableScalarRefT<Edge, Int64> VariableEdgeInt64;

/*!
  \ingroup Variable
  \brief Quantity at the face of 64-bit integer type
*/
typedef MeshVariableScalarRefT<Face, Int64> VariableFaceInt64;

/*!
  \ingroup Variable
  \brief Quantity at the cell center of 64-bit integer type
*/
typedef MeshVariableScalarRefT<Cell, Int64> VariableCellInt64;

/*!
  \ingroup Variable
  \brief Particle quantity of 64-bit integer type
*/
typedef MeshVariableScalarRefT<Particle, Int64> VariableParticleInt64;

/*!
  \ingroup Variable
  \brief Degree of Freedom quantity of 64-bit integer type
*/
typedef MeshVariableScalarRefT<DoF, Int64> VariableDoFInt64;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
  \ingroup Variable
  \brief Quantity of natural integer type
  \deprecated Use #VariableNodeInteger instead
*/
typedef MeshVariableScalarRefT<Node, Integer> VariableNodeInteger;

/*!
  \ingroup Variable
  \brief Quantity at the face of natural integer type
  \deprecated Use #VariableFaceInteger instead
*/
typedef MeshVariableScalarRefT<Face, Integer> VariableFaceInteger;

/*!
  \ingroup Variable
  \brief Quantity at the cell center of natural integer type
  \deprecated Use #VariableCellInteger instead
*/
typedef MeshVariableScalarRefT<Cell, Integer> VariableCellInteger;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
  \ingroup Variable
  \brief Quantity of 64-bit integer type
*/
typedef ItemVariableScalarRefT<Byte> VariableItemByte;

/*!
  \ingroup Variable
  \brief Quantity at the node of byte type
*/
typedef MeshVariableScalarRefT<Node, Byte> VariableNodeByte;

/*!
  \ingroup Variable
  \brief Quantity at the node of byte type
*/
typedef MeshVariableScalarRefT<Edge, Byte> VariableEdgeByte;

/*!
  \ingroup Variable
  \brief Quantity at the face of byte type
*/
typedef MeshVariableScalarRefT<Face, Byte> VariableFaceByte;

/*!
  \ingroup Variable
  \brief Quantity at the cell center of byte type
*/
typedef MeshVariableScalarRefT<Cell, Byte> VariableCellByte;

/*!
  \ingroup Variable
  \brief Particle quantity of byte type
*/
typedef MeshVariableScalarRefT<Particle, Byte> VariableParticleByte;

/*!
  \ingroup Variable
  \brief Degree of Freedom quantity of byte type
*/
typedef MeshVariableScalarRefT<DoF, Byte> VariableDoFByte;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
  \ingroup Variable
  \brief Quantity at the node of boolean type
*/
typedef MeshVariableScalarRefT<Node, Byte> VariableNodeBool;

/*!
  \ingroup Variable
  \brief Quantity at the node of boolean type
*/
typedef MeshVariableScalarRefT<Edge, Byte> VariableEdgeBool;

/*!
  \ingroup Variable
  \brief Quantity at the face of boolean type
*/
typedef MeshVariableScalarRefT<Face, Byte> VariableFaceBool;

/*!
  \ingroup Variable
  \brief Quantity at the cell center of boolean type
*/
typedef MeshVariableScalarRefT<Cell, Byte> VariableCellBool;

/*!
  \ingroup Variable
  \brief Particle quantity of boolean type
*/
typedef MeshVariableScalarRefT<Particle, Byte> VariableParticleBool;

/*!
  \ingroup Variable
  \brief Degree of Freedom quantity of boolean type
*/
typedef MeshVariableScalarRefT<DoF, Byte> VariableDoFBool;

/*!---------------------------------------------------------------------------*/
/*!---------------------------------------------------------------------------*/

/*!
  \ingroup Variable
  \brief Quantity at the node of real array type
*/
typedef MeshVariableArrayRefT<Node, Real> VariableNodeArrayReal;

/*!
  \ingroup Variable
  \brief Quantity at the node of real array type
*/
typedef MeshVariableArrayRefT<Edge, Real> VariableEdgeArrayReal;

/*!
  \ingroup Variable
  \brief Quantity at the face of real array type
*/
typedef MeshVariableArrayRefT<Face, Real> VariableFaceArrayReal;

/*!
  \ingroup Variable
  \brief Quantity at the cell center of real array type
*/
typedef MeshVariableArrayRefT<Cell, Real> VariableCellArrayReal;

/*!
  \ingroup Variable
  \brief Particle quantity of real array type
*/
typedef MeshVariableArrayRefT<Particle, Real> VariableParticleArrayReal;

/*!
  \ingroup Variable
  \brief Degree of Freedom quantity of real array type
*/
typedef MeshVariableArrayRefT<DoF, Real> VariableDoFArrayReal;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
  \ingroup Variable
  \brief Quantity at the node of coordinate array type
*/
typedef MeshVariableArrayRefT<Node, Real2> VariableNodeArrayReal2;

/*!
  \ingroup Variable
  \brief Quantity at the node of coordinate array type
*/
typedef MeshVariableArrayRefT<Edge, Real2> VariableEdgeArrayReal2;

/*!
  \ingroup Variable
  \brief Quantity at the face of coordinate array type
*/
typedef MeshVariableArrayRefT<Face, Real2> VariableFaceArrayReal2;

/*!
  \ingroup Variable
  \brief Quantity at the cell center of coordinate array type
*/
typedef MeshVariableArrayRefT<Cell, Real2> VariableCellArrayReal2;

/*!
  \ingroup Variable
  \brief Particle quantity of coordinate array type
*/
typedef MeshVariableArrayRefT<Particle, Real2> VariableParticleArrayReal2;

/*!
  \ingroup Variable
  \brief Degree of Freedom quantity of coordinate array type
*/
typedef MeshVariableArrayRefT<DoF, Real2> VariableDoFArrayReal2;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
  \ingroup Variable
  \brief Quantity at the node of coordinate array type
*/
typedef MeshVariableArrayRefT<Node, Real3> VariableNodeArrayReal3;

/*!
  \ingroup Variable
  \brief Quantity at the face of coordinate array type
*/
typedef MeshVariableArrayRefT<Face, Real3> VariableFaceArrayReal3;

/*!
  \ingroup Variable
  \brief Quantity at the cell center of coordinate array type
*/
typedef MeshVariableArrayRefT<Cell, Real3> VariableCellArrayReal3;

/*!
  \ingroup Variable
  \brief Particle quantity of coordinate array type
*/
typedef MeshVariableArrayRefT<Particle, Real3> VariableParticleArrayReal3;

/*!
  \ingroup Variable
  \brief Degree of Freedom quantity of real array type
*/
typedef MeshVariableArrayRefT<DoF, Real3> VariableDoFArrayReal3;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
  \ingroup Variable
  \brief Quantity at the node of real tensor array type
*/
typedef MeshVariableArrayRefT<Node, Real2x2> VariableNodeArrayReal2x2;

/*!
  \ingroup Variable
  \brief Quantity at faces of real tensor array type
*/
typedef MeshVariableArrayRefT<Face, Real2x2> VariableFaceArrayReal2x2;

/*!
  \ingroup Variable
  \brief Quantity at cell centers of real tensor array type
*/
typedef MeshVariableArrayRefT<Cell, Real2x2> VariableCellArrayReal2x2;

/*!
  \ingroup Variable
  \brief Particle quantity of real tensor array type
*/
typedef MeshVariableArrayRefT<Particle, Real2x2> VariableParticleArrayReal2x2;

/*!
  \ingroup Variable
  \brief DoF quantity of real tensor array type
*/
typedef MeshVariableArrayRefT<DoF, Real2x2> VariableDoFArrayReal2x2;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
  \ingroup Variable
  \brief Quantity at nodes of real tensor array type
*/
typedef MeshVariableArrayRefT<Node, Real3x3> VariableNodeArrayReal3x3;

/*!
  \ingroup Variable
  \brief Quantity at faces of real tensor array type
*/
typedef MeshVariableArrayRefT<Face, Real3x3> VariableFaceArrayReal3x3;

/*!
  \ingroup Variable
  \brief Quantity at cell centers of real tensor array type
*/
typedef MeshVariableArrayRefT<Cell, Real3x3> VariableCellArrayReal3x3;

/*!
  \ingroup Variable
  \brief Particle quantity of real tensor array type
*/
typedef MeshVariableArrayRefT<Particle, Real3x3> VariableParticleArrayReal3x3;

/*!
  \ingroup Variable
  \brief DoF quantity of real tensor array type
*/
typedef MeshVariableArrayRefT<DoF, Real3x3> VariableDoFArrayReal3x3;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
  \ingroup Variable
  \brief Quantity at nodes of integer array type
*/
typedef MeshVariableArrayRefT<Node, Integer> VariableNodeArrayInteger;

/*!
  \ingroup Variable
  \brief Quantity at faces of integer array type
*/
typedef MeshVariableArrayRefT<Face, Integer> VariableFaceArrayInteger;

/*!
  \ingroup Variable
  \brief Quantity at cell centers of integer array type
*/
typedef MeshVariableArrayRefT<Cell, Integer> VariableCellArrayInteger;

/*!
  \ingroup Variable
  \brief Particle quantity of integer array type
*/
typedef MeshVariableArrayRefT<Particle, Integer> VariableParticleArrayInteger;

/*!
  \ingroup Variable
  \brief DoF quantity of integer array type
*/
typedef MeshVariableArrayRefT<DoF, Integer> VariableDoFArrayInteger;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
  \ingroup Variable
  \brief Quantity at nodes of 16-bit integer array type
*/
typedef MeshVariableArrayRefT<Node, Int16> VariableNodeArrayInt16;

/*!
  \ingroup Variable
  \brief Quantity at faces of 16-bit integer array type
*/
typedef MeshVariableArrayRefT<Face, Int16> VariableFaceArrayInt16;

/*!
  \ingroup Variable
  \brief Quantity at cell centers of 16-bit integer array type
*/
typedef MeshVariableArrayRefT<Cell, Int16> VariableCellArrayInt16;

/*!
  \ingroup Variable
  \brief Particle quantity of 16-bit integer array type
*/
typedef MeshVariableArrayRefT<Particle, Int16> VariableParticleArrayInt16;

/*!
  \ingroup Variable
  \brief DoF quantity of 32-bit integer array type
*/
typedef MeshVariableArrayRefT<DoF, Int32> VariableDoFArrayInt32;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
  \ingroup Variable
  \brief Quantity at nodes of integer array type
*/
typedef MeshVariableArrayRefT<Node, Int32> VariableNodeArrayInt32;

/*!
  \ingroup Variable
  \brief Quantity at faces of integer array type
*/
typedef MeshVariableArrayRefT<Face, Int32> VariableFaceArrayInt32;

/*!
  \ingroup Variable
  \brief Quantity at cell centers of integer array type
*/
typedef MeshVariableArrayRefT<Cell, Int32> VariableCellArrayInt32;

/*!
  \ingroup Variable
  \brief Particle quantity of integer array type
*/
typedef MeshVariableArrayRefT<Particle, Int32> VariableParticleArrayInt32;

/*!
  \ingroup Variable
  \brief DoF quantity of 32-bit integer array type
*/
typedef MeshVariableArrayRefT<DoF, Int32> VariableDoFArrayInt32;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
  \ingroup Variable
  \brief Quantity at nodes of integer array type
*/
typedef MeshVariableArrayRefT<Node, Int64> VariableNodeArrayInt64;

/*!
  \ingroup Variable
  \brief Quantity at faces of integer array type
*/
typedef MeshVariableArrayRefT<Face, Int64> VariableFaceArrayInt64;

/*!
  \ingroup Variable
  \brief Quantity at cell centers of integer array type
*/
typedef MeshVariableArrayRefT<Cell, Int64> VariableCellArrayInt64;

/*!
  \ingroup Variable
  \brief Particle quantity of integer array type
*/
typedef MeshVariableArrayRefT<Particle, Int64> VariableParticleArrayInt64;

/*!
  \ingroup Variable
  \brief DoF quantity of 64-bit integer array type
*/
typedef MeshVariableArrayRefT<DoF, Int64> VariableDoFArrayInt64;

/*!
  \ingroup Variable
  \brief DoF quantity of 64-bit integer array type
*/
typedef MeshVariableArrayRefT<DoF, Int64> VariableDoFArrayInt64;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
  \ingroup Variable
  \brief Quantity at nodes of byte array type
*/
typedef MeshVariableArrayRefT<Node, Byte> VariableNodeArrayByte;

/*!
  \ingroup Variable
  \brief Quantity at faces of byte array type
*/
typedef MeshVariableArrayRefT<Face, Byte> VariableFaceArrayByte;

/*!
  \ingroup Variable
  \brief Quantity at cell centers of byte array type
*/
typedef MeshVariableArrayRefT<Cell, Byte> VariableCellArrayByte;

/*!
  \ingroup Variable
  \brief Particle quantity of byte array type
*/
typedef MeshVariableArrayRefT<Particle, Byte> VariableParticleArrayByte;

/*!
  \ingroup Variable
  \brief DoF quantity of byte array type
*/
typedef MeshVariableArrayRefT<DoF, Byte> VariableDoFArrayByte;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
  \ingroup Variable
  \brief Quantity at nodes of boolean array type
*/
typedef MeshVariableArrayRefT<Node, Byte> VariableNodeArrayBool;

/*!
  \ingroup Variable
  \brief Quantity at faces of boolean array type
*/
typedef MeshVariableArrayRefT<Face, Byte> VariableFaceArrayBool;

/*!
  \ingroup Variable
  \brief Quantity at cell centers of boolean array type
*/
typedef MeshVariableArrayRefT<Cell, Byte> VariableCellArrayBool;

/*!
  \ingroup Variable
  \brief Particle quantity of boolean array type
*/
typedef MeshVariableArrayRefT<Particle, Byte> VariableParticleArrayBool;

/*!
  \ingroup Variable
  \brief DoF quantity of boolean array type
*/
typedef MeshVariableArrayRefT<DoF, Byte> VariableDoFArrayBool;

/*!---------------------------------------------------------------------------*/
/*!---------------------------------------------------------------------------*/

/*!---------------------------------------------------------------------------*/
/*!---------------------------------------------------------------------------*/

/*!
  \ingroup Partial Variable
  \brief Quantity of real type
*/
typedef ItemPartialVariableScalarRefT<Real> PartialVariableItemReal;

/*!
  \ingroup Partial Variable
  \brief Quantity at node of real type
*/
typedef MeshPartialVariableScalarRefT<Node, Real> PartialVariableNodeReal;

/*!
  \ingroup Partial Variable
  \brief Quantity at node of real type
*/
typedef MeshPartialVariableScalarRefT<Edge, Real> PartialVariableEdgeReal;

/*!
  \ingroup Partial Variable
  \brief Quantity at faces of real type
*/
typedef MeshPartialVariableScalarRefT<Face, Real> PartialVariableFaceReal;

/*!
  \ingroup Partial Variable
  \brief Quantity at cell centers of real type
*/
typedef MeshPartialVariableScalarRefT<Cell, Real> PartialVariableCellReal;

/*!
  \ingroup Partial Variable
  \brief Particle quantity of real type
*/
typedef MeshPartialVariableScalarRefT<Particle, Real> PartialVariableParticleReal;

/*!
  \ingroup Partial Variable
  \brief DoF quantity of real type
*/
typedef MeshPartialVariableScalarRefT<DoF, Real> PartialVariableDoFReal;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
  \ingroup Partial Variable
  \brief Quantity of 2D coordinate type
*/
typedef ItemPartialVariableScalarRefT<Real2> PartialVariableItemReal2;

/*!
  \ingroup Partial Variable
  \brief Quantity at node of coordinate type
*/
typedef MeshPartialVariableScalarRefT<Node, Real2> PartialVariableNodeReal2;

/*!
  \ingroup Partial Variable
  \brief Quantity at node of coordinate type
*/
typedef MeshPartialVariableScalarRefT<Edge, Real2> PartialVariableEdgeReal2;

/*!
  \ingroup Partial Variable
  \brief Quantity at faces of coordinate type
*/
typedef MeshPartialVariableScalarRefT<Face, Real2> PartialVariableFaceReal2;

/*!
  \ingroup Partial Variable
  \brief Quantity at cell centers of coordinate type
*/
typedef MeshPartialVariableScalarRefT<Cell, Real2> PartialVariableCellReal2;

/*!
  \ingroup Partial Variable
  \brief Particle quantity of coordinate type
*/
typedef MeshPartialVariableScalarRefT<Particle, Real2> PartialVariableParticleReal2;

/*!
  \ingroup Partial Variable
  \brief DoF quantity of coordinate type
*/
typedef MeshPartialVariableScalarRefT<DoF, Real2> PartialVariableDoFReal2;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
  \ingroup Partial Variable
  \brief Quantity of 3D coordinate type
*/
typedef ItemPartialVariableScalarRefT<Real3> PartialVariableItemReal3;

/*!
  \ingroup Partial Variable
  \brief Quantity at node of coordinate type
*/
typedef MeshPartialVariableScalarRefT<Node, Real3> PartialVariableNodeReal3;

/*!
  \ingroup Partial Variable
  \brief Quantity at node of coordinate type
*/
typedef MeshPartialVariableScalarRefT<Edge, Real3> PartialVariableEdgeReal3;

/*!
  \ingroup Partial Variable
  \brief Quantity at faces of coordinate type
*/
typedef MeshPartialVariableScalarRefT<Face, Real3> PartialVariableFaceReal3;

/*!
  \ingroup Partial Variable
  \brief Quantity at cell centers of coordinate type
*/
typedef MeshPartialVariableScalarRefT<Cell, Real3> PartialVariableCellReal3;

/*!
  \ingroup Partial Variable
  \brief Particle quantity of coordinate type
*/
typedef MeshPartialVariableScalarRefT<Particle, Real3> PartialVariableParticleReal3;

/*!
  \ingroup Partial Variable
  \brief DoF quantity of coordinate type
*/
typedef MeshPartialVariableScalarRefT<DoF, Real3> PartialVariableDoFReal3;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
  \ingroup Partial Variable
  \brief Quantity of real tensor type
*/
typedef ItemPartialVariableScalarRefT<Real2x2> PartialVariableItemReal2x2;

/*!
  \ingroup Partial Variable
  \brief Quantity at node of real tensor type
*/
typedef MeshPartialVariableScalarRefT<Node, Real2x2> PartialVariableNodeReal2x2;

/*!
  \ingroup Partial Variable
  \brief Quantity at node of real tensor type
*/
typedef MeshPartialVariableScalarRefT<Edge, Real2x2> PartialVariableEdgeReal2x2;

/*!
  \ingroup Partial Variable
  \brief Quantity at faces of real tensor type
*/
typedef MeshPartialVariableScalarRefT<Face, Real2x2> PartialVariableFaceReal2x2;

/*!
  \ingroup Partial Variable
  \brief Quantity at cell centers of real tensor type
*/
typedef MeshPartialVariableScalarRefT<Cell, Real2x2> PartialVariableCellReal2x2;

/*!
  \ingroup Partial Variable
  \brief Particle quantity of real tensor type
*/
typedef MeshPartialVariableScalarRefT<Particle, Real2x2> PartialVariableParticleReal2x2;

/*!
  \ingroup Partial Variable
  \brief DoF quantity of real tensor type
*/
typedef MeshPartialVariableScalarRefT<DoF, Real2x2> PartialVariableDoFReal2x2;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
  \ingroup Partial Variable
  \brief Quantity of real tensor type
*/
typedef ItemPartialVariableScalarRefT<Real3x3> PartialVariableItemReal3x3;

/*!
  \ingroup Partial Variable
  \brief Quantity at the node of real tensor type
*/
typedef MeshPartialVariableScalarRefT<Node, Real3x3> PartialVariableNodeReal3x3;

/*!
  \ingroup Partial Variable
  \brief Quantity at the node of real tensor type
*/
typedef MeshPartialVariableScalarRefT<Edge, Real3x3> PartialVariableEdgeReal3x3;

/*!
  \ingroup Partial Variable
  \brief Quantity at the faces of real tensor type
*/
typedef MeshPartialVariableScalarRefT<Face, Real3x3> PartialVariableFaceReal3x3;

/*!
  \ingroup Partial Variable
  \brief Quantity at the cell center of real tensor type
*/
typedef MeshPartialVariableScalarRefT<Cell, Real3x3> PartialVariableCellReal3x3;

/*!
  \ingroup Partial Variable
  \brief Particle quantity of real tensor type
*/
typedef MeshPartialVariableScalarRefT<Particle, Real3x3> PartialVariableParticleReal3x3;

/*!
  \ingroup Partial Variable
  \brief DoF quantity of real tensor type
*/
typedef MeshPartialVariableScalarRefT<DoF, Real3x3> PartialVariableDoFReal3x3;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
  \ingroup Partial Variable
  \brief Integer type quantity
*/
typedef ItemPartialVariableScalarRefT<Integer> PartialVariableItemInteger;

/*!
  \ingroup Partial Variable
  \brief Integer type quantity at the node
*/
typedef MeshPartialVariableScalarRefT<Node, Integer> PartialVariableNodeInteger;

/*!
  \ingroup Partial Variable
  \brief Integer type quantity at the node
*/
typedef MeshPartialVariableScalarRefT<Edge, Integer> PartialVariableEdgeInteger;

/*!
  \ingroup Partial Variable
  \brief Integer type quantity at the faces
*/
typedef MeshPartialVariableScalarRefT<Face, Integer> PartialVariableFaceInteger;

/*!
  \ingroup Partial Variable
  \brief Integer type quantity at the cell center
*/
typedef MeshPartialVariableScalarRefT<Cell, Integer> PartialVariableCellInteger;

/*!
  \ingroup Partial Variable
  \brief Particle integer quantity
*/
typedef MeshPartialVariableScalarRefT<Particle, Integer> PartialVariableParticleInteger;

/*!
  \ingroup Partial Variable
  \brief DoF integer quantity
*/
typedef MeshPartialVariableScalarRefT<DoF, Integer> PartialVariableDoFInteger;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
  \ingroup Partial Variable
  \brief 32-bit integer type quantity
*/
typedef ItemPartialVariableScalarRefT<Int32> PartialVariableItemInt32;

/*!
  \ingroup Partial Variable
  \brief 32-bit integer type quantity at the node
*/
typedef MeshPartialVariableScalarRefT<Node, Int32> PartialVariableNodeInt32;

/*!
  \ingroup Partial Variable
  \brief 32-bit integer type quantity at the node
*/
typedef MeshPartialVariableScalarRefT<Edge, Int32> PartialVariableEdgeInt32;

/*!
  \ingroup Partial Variable
  \brief 32-bit integer type quantity at the faces
*/
typedef MeshPartialVariableScalarRefT<Face, Int32> PartialVariableFaceInt32;

/*!
  \ingroup Partial Variable
  \brief 32-bit integer type quantity at the cell center
*/
typedef MeshPartialVariableScalarRefT<Cell, Int32> PartialVariableCellInt32;

/*!
  \ingroup Partial Variable
  \brief 32-bit integer particle quantity
*/
typedef MeshPartialVariableScalarRefT<Particle, Int32> PartialVariableParticleInt32;

/*!
  \ingroup Partial Variable
  \brief 32-bit integer DoF quantity
*/
typedef MeshPartialVariableScalarRefT<DoF, Int32> PartialVariableDoFInt32;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
  \ingroup Partial Variable
  \brief 16-bit integer type quantity
*/
typedef ItemPartialVariableScalarRefT<Int16> PartialVariableItemInt16;

/*!
  \ingroup Partial Variable
  \brief 16-bit integer type quantity at the node
*/
typedef MeshPartialVariableScalarRefT<Node, Int16> PartialVariableNodeInt16;

/*!
  \ingroup Partial Variable
  \brief 16-bit integer type quantity at the node
*/
typedef MeshPartialVariableScalarRefT<Edge, Int16> PartialVariableEdgeInt16;

/*!
  \ingroup Partial Variable
  \brief 16-bit integer type quantity at the faces
*/
typedef MeshPartialVariableScalarRefT<Face, Int16> PartialVariableFaceInt16;

/*!
  \ingroup Partial Variable
  \brief 16-bit integer type quantity at the cell center
*/
typedef MeshPartialVariableScalarRefT<Cell, Int16> PartialVariableCellInt16;

/*!
  \ingroup Partial Variable
  \brief 16-bit integer particle quantity
*/
typedef MeshPartialVariableScalarRefT<Particle, Int16> PartialVariableParticleInt16;

/*!
  \ingroup Partial Variable
  \brief 16-bit integer DoF quantity
*/
typedef MeshPartialVariableScalarRefT<DoF, Int16> PartialVariableDoFInt16;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
  \ingroup Variable
  \brief 64-bit integer type quantity
*/
typedef ItemPartialVariableScalarRefT<Int64> PartialVariableItemInt64;

/*!
  \ingroup Partial Variable
  \brief 64-bit integer type quantity at the node
*/
typedef MeshPartialVariableScalarRefT<Node, Int64> PartialVariableNodeInt64;

/*!
  \ingroup Partial Variable
  \brief 64-bit integer type quantity at the node
*/
typedef MeshPartialVariableScalarRefT<Edge, Int64> PartialVariableEdgeInt64;

/*!
  \ingroup Partial Variable
  \brief 64-bit integer type quantity at the faces
*/
typedef MeshPartialVariableScalarRefT<Face, Int64> PartialVariableFaceInt64;

/*!
  \ingroup Partial Variable
  \brief 64-bit integer type quantity at the cell center
*/
typedef MeshPartialVariableScalarRefT<Cell, Int64> PartialVariableCellInt64;

/*!
  \ingroup Partial Variable
  \brief 64-bit integer particle quantity
*/
typedef MeshPartialVariableScalarRefT<Particle, Int64> PartialVariableParticleInt64;

/*!
  \ingroup Partial Variable
  \brief 64-bit integer DoF quantity
*/
typedef MeshPartialVariableScalarRefT<DoF, Int64> PartialVariableDoFInt64;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
  \ingroup Partial Variable
  \brief 64-bit integer type quantity
*/
typedef ItemPartialVariableScalarRefT<Byte> PartialVariableItemByte;

/*!
  \ingroup Partial Variable
  \brief Byte type quantity at the node
*/
typedef MeshPartialVariableScalarRefT<Node, Byte> PartialVariableNodeByte;

/*!
  \ingroup Partial Variable
  \brief Byte type quantity at the node
*/
typedef MeshPartialVariableScalarRefT<Edge, Byte> PartialVariableEdgeByte;

/*!
  \ingroup Partial Variable
  \brief Byte type quantity at the faces
*/
typedef MeshPartialVariableScalarRefT<Face, Byte> PartialVariableFaceByte;

/*!
  \ingroup Partial Variable
  \brief Byte type quantity at the cell center
*/
typedef MeshPartialVariableScalarRefT<Cell, Byte> PartialVariableCellByte;

/*!
  \ingroup Partial Variable
  \brief Byte type particle quantity
*/
typedef MeshPartialVariableScalarRefT<Particle, Byte> PartialVariableParticleByte;

/*!
  \ingroup Partial Variable
  \brief Byte type DoF quantity
*/
typedef MeshPartialVariableScalarRefT<DoF, Byte> PartialVariableDoFByte;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
  \ingroup Partial Variable
  \brief Boolean type quantity at the node
*/
typedef MeshPartialVariableScalarRefT<Node, Byte> PartialVariableNodeBool;

/*!
  \ingroup Partial Variable
  \brief Boolean type quantity at the node
*/
typedef MeshPartialVariableScalarRefT<Edge, Byte> PartialVariableEdgeBool;

/*!
  \ingroup Partial Variable
  \brief Boolean type quantity at the faces
*/
typedef MeshPartialVariableScalarRefT<Face, Byte> PartialVariableFaceBool;

/*!
  \ingroup Partial Variable
  \brief Boolean type quantity at the cell center
*/
typedef MeshPartialVariableScalarRefT<Cell, Byte> PartialVariableCellBool;

/*!
  \ingroup Partial Variable
  \brief Boolean type particle quantity
*/
typedef MeshPartialVariableScalarRefT<Particle, Byte> PartialVariableParticleBool;

/*!
  \ingroup Partial Variable
  \brief Boolean DoF quantity
*/
typedef MeshPartialVariableScalarRefT<DoF, Byte> PartialVariableDoFBool;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
  \ingroup Partial Variable
  \brief Real array type quantity
*/
typedef ItemPartialVariableArrayRefT<Real> PartialVariableItemArrayReal;

/*!
  \ingroup Partial Variable
  \brief Real array type quantity at the node
*/
typedef MeshPartialVariableArrayRefT<Node, Real> PartialVariableNodeArrayReal;

/*!
  \ingroup Partial Variable
  \brief Real array type quantity at the node
*/
typedef MeshPartialVariableArrayRefT<Edge, Real> PartialVariableEdgeArrayReal;

/*!
  \ingroup Partial Variable
  \brief Real array type quantity at the faces
*/
typedef MeshPartialVariableArrayRefT<Face, Real> PartialVariableFaceArrayReal;

/*!
  \ingroup Partial Variable
  \brief Real array type quantity at the cell center
*/
typedef MeshPartialVariableArrayRefT<Cell, Real> PartialVariableCellArrayReal;

/*!
  \ingroup Partial Variable
  \brief Real array type particle quantity
*/
typedef MeshPartialVariableArrayRefT<Particle, Real> PartialVariableParticleArrayReal;

/*!
  \ingroup Partial Variable
  \brief Real array DoF quantity
*/
typedef MeshPartialVariableArrayRefT<DoF, Real> PartialVariableDoFArrayReal;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
  \ingroup Partial Variable
  \brief 2D coordinate array type quantity
*/
typedef ItemPartialVariableArrayRefT<Real2> PartialVariableItemArrayReal2;

/*!
  \ingroup Partial Variable
  \brief 2D coordinate array type quantity at the node
*/
typedef MeshPartialVariableArrayRefT<Node, Real2> PartialVariableNodeArrayReal2;

/*!
  \ingroup Partial Variable
  \brief 2D coordinate array type quantity at the node
*/
typedef MeshPartialVariableArrayRefT<Edge, Real2> PartialVariableEdgeArrayReal2;

/*!
  \ingroup Partial Variable
  \brief 2D coordinate array type quantity at the faces
*/
typedef MeshPartialVariableArrayRefT<Face, Real2> PartialVariableFaceArrayReal2;

/*!
  \ingroup Partial Variable
  \brief 2D coordinate array type quantity at the cell center
*/
typedef MeshPartialVariableArrayRefT<Cell, Real2> PartialVariableCellArrayReal2;

/*!
  \ingroup Partial Variable
  \brief 2D coordinate array type particle quantity
*/
typedef MeshPartialVariableArrayRefT<Particle, Real2> PartialVariableParticleArrayReal2;

/*!
  \ingroup Partial Variable
  \brief 2D coordinate array DoF quantity
*/
typedef MeshPartialVariableArrayRefT<DoF, Real2> PartialVariableDoFArrayReal2;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
  \ingroup Partial Variable
  \brief 3D coordinate array type quantity
*/
typedef ItemPartialVariableArrayRefT<Real3> PartialVariableItemArrayReal3;

/*!
  \ingroup Partial Variable
  \brief 3D coordinate array type quantity at the node
*/
typedef MeshPartialVariableArrayRefT<Node, Real3> PartialVariableNodeArrayReal3;

/*!
  \ingroup Partial Variable
  \brief 3D coordinate array type quantity at the node
*/
typedef MeshPartialVariableArrayRefT<Edge, Real3> PartialVariableEdgeArrayReal3;

/*!
  \ingroup Partial Variable
  \brief 3D coordinate array type quantity at the faces
*/
typedef MeshPartialVariableArrayRefT<Face, Real3> PartialVariableFaceArrayReal3;

/*!
  \ingroup Partial Variable
  \brief 3D coordinate array type quantity at the cell center
*/
typedef MeshPartialVariableArrayRefT<Cell, Real3> PartialVariableCellArrayReal3;

/*!
  \ingroup Partial Variable
  \brief 3D coordinate array type particle quantity
*/
typedef MeshPartialVariableArrayRefT<Particle, Real3> PartialVariableParticleArrayReal3;

/*!
  \ingroup Partial Variable
  \brief DoF quantity of type array of coordinates
*/
typedef MeshPartialVariableArrayRefT<DoF, Real3> PartialVariableDoFArrayReal3;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
  \ingroup Partial Variable
  \brief Quantity of type array of real tensors
*/
typedef ItemPartialVariableArrayRefT<Real2x2> PartialVariableItemArrayReal2x2;

/*!
  \ingroup Partial Variable
  \brief Quantity at the node of type array of real tensors
*/
typedef MeshPartialVariableArrayRefT<Node, Real2x2> PartialVariableNodeArrayReal2x2;

/*!
  \ingroup Partial Variable
  \brief Quantity at the node of type array of real tensors
*/
typedef MeshPartialVariableArrayRefT<Edge, Real2x2> PartialVariableEdgeArrayReal2x2;

/*!
  \ingroup Partial Variable
  \brief Quantity on the faces of type array of real tensors
*/
typedef MeshPartialVariableArrayRefT<Face, Real2x2> PartialVariableFaceArrayReal2x2;

/*!
  \ingroup Partial Variable
  \brief Quantity at the cell center of type array of real tensors
*/
typedef MeshPartialVariableArrayRefT<Cell, Real2x2> PartialVariableCellArrayReal2x2;

/*!
  \ingroup Partial Variable
  \brief Particle quantity of type array of real tensors
*/
typedef MeshPartialVariableArrayRefT<Particle, Real2x2> PartialVariableParticleArrayReal2x2;

/*!
  \ingroup Partial Variable
  \brief DoF quantity of type array of real tensors
*/
typedef MeshPartialVariableArrayRefT<DoF, Real2x2> PartialVariableDoFArrayReal2x2;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
  \ingroup Partial Variable
  \brief Quantity of type array of real tensors
*/
typedef ItemPartialVariableArrayRefT<Real3x3> PartialVariableItemArrayReal3x3;

/*!
  \ingroup Partial Variable
  \brief Quantity at the node of type array of real tensors
*/
typedef MeshPartialVariableArrayRefT<Node, Real3x3> PartialVariableNodeArrayReal3x3;

/*!
  \ingroup Partial Variable
  \brief Quantity at the node of type array of real tensors
*/
typedef MeshPartialVariableArrayRefT<Edge, Real3x3> PartialVariableEdgeArrayReal3x3;

/*!
  \ingroup Partial Variable
  \brief Quantity on the faces of type array of real tensors
*/
typedef MeshPartialVariableArrayRefT<Face, Real3x3> PartialVariableFaceArrayReal3x3;

/*!
  \ingroup Partial Variable
  \brief Quantity at the cell center of type array of real tensors
*/
typedef MeshPartialVariableArrayRefT<Cell, Real3x3> PartialVariableCellArrayReal3x3;

/*!
  \ingroup Partial Variable
  \brief Particle quantity of type array of real tensors
*/
typedef MeshPartialVariableArrayRefT<Particle, Real3x3> PartialVariableParticleArrayReal3x3;

/*!
  \ingroup Partial Variable
  \brief DoF quantity of type array of real tensors
*/
typedef MeshPartialVariableArrayRefT<DoF, Real3x3> PartialVariableDoFArrayReal3x3;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
  \ingroup Partial Variable
  \brief Quantity of type array of integers
*/
typedef ItemPartialVariableArrayRefT<Integer> PartialVariableItemArrayInteger;

/*!
  \ingroup Partial Variable
  \brief Quantity at the node of type array of integers
*/
typedef MeshPartialVariableArrayRefT<Node, Integer> PartialVariableNodeArrayInteger;

/*!
  \ingroup Partial Variable
  \brief Quantity at the node of type array of integers
*/
typedef MeshPartialVariableArrayRefT<Edge, Integer> PartialVariableEdgeArrayInteger;

/*!
  \ingroup Partial Variable
  \brief Quantity on the faces of type array of integers
*/
typedef MeshPartialVariableArrayRefT<Face, Integer> PartialVariableFaceArrayInteger;

/*!
  \ingroup Partial Variable
  \brief Quantity at the cell center of type array of integers
*/
typedef MeshPartialVariableArrayRefT<Cell, Integer> PartialVariableCellArrayInteger;

/*!
  \ingroup Partial Variable
  \brief Particle quantity of type array of integers
*/
typedef MeshPartialVariableArrayRefT<Particle, Integer> PartialVariableParticleArrayInteger;

/*!
  \ingroup Partial Variable
  \brief DoF quantity of type array of integers
*/
typedef MeshPartialVariableArrayRefT<DoF, Integer> PartialVariableDoFArrayInteger;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
  \ingroup Partial Variable
  \brief Quantity of type 16-bit integer array
*/
typedef ItemPartialVariableArrayRefT<Int16> PartialVariableItemArrayInt16;

/*!
  \ingroup Partial Variable
  \brief Quantity at the node of type 16-bit integer array
*/
typedef MeshPartialVariableArrayRefT<Node, Int16> PartialVariableNodeArrayInt16;

/*!
  \ingroup Partial Variable
  \brief Quantity at the node of type 16-bit integer array
*/
typedef MeshPartialVariableArrayRefT<Edge, Int16> PartialVariableEdgeArrayInt16;

/*!
  \ingroup Partial Variable
  \brief Quantity on the faces of type 16-bit integer array
*/
typedef MeshPartialVariableArrayRefT<Face, Int16> PartialVariableFaceArrayInt16;

/*!
  \ingroup Partial Variable
  \brief Quantity at the cell center of type 16-bit integer array
*/
typedef MeshPartialVariableArrayRefT<Cell, Int16> PartialVariableCellArrayInt16;

/*!
  \ingroup Partial Variable
  \brief Particle quantity of type 16-bit integer array
*/
typedef MeshPartialVariableArrayRefT<Particle, Int16> PartialVariableParticleArrayInt16;

/*!
  \ingroup Partial Variable
  \brief DoF quantity of type 16-bit integer array
*/
typedef MeshPartialVariableArrayRefT<DoF, Int16> PartialVariableDoFArrayInt16;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
  \ingroup Partial Variable
  \brief Quantity of type 32-bit integer array
*/
typedef ItemPartialVariableArrayRefT<Int32> PartialVariableItemArrayInt32;

/*!
  \ingroup Partial Variable
  \brief Quantity at the node of type 32-bit integer array
*/
typedef MeshPartialVariableArrayRefT<Node, Int32> PartialVariableNodeArrayInt32;

/*!
  \ingroup Partial Variable
  \brief Quantity at the node of type 32-bit integer array
*/
typedef MeshPartialVariableArrayRefT<Edge, Int32> PartialVariableEdgeArrayInt32;

/*!
  \ingroup Partial Variable
  \brief Quantity on the faces of type 32-bit integer array
*/
typedef MeshPartialVariableArrayRefT<Face, Int32> PartialVariableFaceArrayInt32;

/*!
  \ingroup Partial Variable
  \brief Quantity at the cell center of type 32-bit integer array
*/
typedef MeshPartialVariableArrayRefT<Cell, Int32> PartialVariableCellArrayInt32;

/*!
  \ingroup Partial Variable
  \brief Particle quantity of type 32-bit integer array
*/
typedef MeshPartialVariableArrayRefT<Particle, Int32> PartialVariableParticleArrayInt32;

/*!
  \ingroup Partial Variable
  \brief DoF quantity of type 32-bit integer array
*/
typedef MeshPartialVariableArrayRefT<DoF, Int32> PartialVariableDoFArrayInt32;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
  \ingroup Variable
  \brief Quantity of type 64-bit integer array
*/
typedef ItemPartialVariableArrayRefT<Int64> PartialVariableItemArrayInt64;

/*!
  \ingroup Partial Variable
  \brief Quantity at the node of type 64-bit integer array
*/
typedef MeshPartialVariableArrayRefT<Node, Int64> PartialVariableNodeArrayInt64;

/*!
  \ingroup Partial Variable
  \brief Quantity at the node of type 64-bit integer array
*/
typedef MeshPartialVariableArrayRefT<Edge, Int64> PartialVariableEdgeArrayInt64;

/*!
  \ingroup Partial Variable
  \brief Quantity on the faces of type 64-bit integer array
*/
typedef MeshPartialVariableArrayRefT<Face, Int64> PartialVariableFaceArrayInt64;

/*!
  \ingroup Partial Variable
  \brief Quantity at the cell center of type 64-bit integer array
*/
typedef MeshPartialVariableArrayRefT<Cell, Int64> PartialVariableCellArrayInt64;

/*!
  \ingroup Partial Variable
  \brief Particle quantity of type 64-bit integer array
*/
typedef MeshPartialVariableArrayRefT<Particle, Int64> PartialVariableParticleArrayInt64;

/*!
  \ingroup Partial Variable
  \brief DoF quantity of type 64-bit integer array
*/
typedef MeshPartialVariableArrayRefT<DoF, Int64> PartialVariableDoFArrayInt64;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
  \ingroup Partial Variable
  \brief Quantity of type 64-bit integer array
*/
typedef ItemPartialVariableArrayRefT<Byte> PartialVariableItemArrayByte;

/*!
  \ingroup Partial Variable
  \brief Quantity at the node of type byte array
*/
typedef MeshPartialVariableArrayRefT<Node, Byte> PartialVariableNodeArrayByte;

/*!
  \ingroup Partial Variable
  \brief Quantity at the node of type byte array
*/
typedef MeshPartialVariableArrayRefT<Edge, Byte> PartialVariableEdgeArrayByte;

/*!
  \ingroup Partial Variable
  \brief Quantity on the faces of type byte array
*/
typedef MeshPartialVariableArrayRefT<Face, Byte> PartialVariableFaceArrayByte;

/*!
  \ingroup Partial Variable
  \brief Quantity at the cell center of type byte array
*/
typedef MeshPartialVariableArrayRefT<Cell, Byte> PartialVariableCellArrayByte;

/*!
  \ingroup Partial Variable
  \brief Particle quantity of type byte array
*/
typedef MeshPartialVariableArrayRefT<Particle, Byte> PartialVariableParticleArrayByte;

/*!
  \ingroup Partial Variable
  \brief DoF quantity of type byte array
*/
typedef MeshPartialVariableArrayRefT<DoF, Byte> PartialVariableDoFArrayByte;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
  \ingroup Partial Variable
  \brief Quantity at the node of type boolean array
*/
typedef MeshPartialVariableArrayRefT<Node, Byte> PartialVariableNodeArrayBool;

/*!
  \ingroup Partial Variable
  \brief Quantity at the node of type boolean array
*/
typedef MeshPartialVariableArrayRefT<Edge, Byte> PartialVariableEdgeArrayBool;

/*!
  \ingroup Partial Variable
  \brief Quantity on the faces of type boolean array
*/
typedef MeshPartialVariableArrayRefT<Face, Byte> PartialVariableFaceArrayBool;

/*!
  \ingroup Partial Variable
  \brief Quantity at the cell center of type boolean array
*/
typedef MeshPartialVariableArrayRefT<Cell, Byte> PartialVariableCellArrayBool;

/*!
  \ingroup Partial Variable
  \brief Particle quantity of type boolean array
*/
typedef MeshPartialVariableArrayRefT<Particle, Byte> PartialVariableParticleArrayBool;

/*!
  \ingroup Partial Variable
  \brief DoF quantity of type boolean array
*/
typedef MeshPartialVariableArrayRefT<DoF, Byte> PartialVariableDoFArrayBool;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename ItemType, typename DataType> class SharedMeshVariableScalarRefT;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
  \ingroup Variable
  \brief Quantity at the node of type real
*/
typedef SharedMeshVariableScalarRefT<Node, Real> SharedVariableNodeReal;

/*!
  \ingroup SharedVariable
  \brief Quantity at the node of type real
*/
typedef SharedMeshVariableScalarRefT<Edge, Real> SharedVariableEdgeReal;

/*!
  \ingroup SharedVariable
  \brief Quantity on the faces of type real
*/
typedef SharedMeshVariableScalarRefT<Face, Real> SharedVariableFaceReal;

/*!
  \ingroup SharedVariable
  \brief Quantity at the cell center of type real
*/
typedef SharedMeshVariableScalarRefT<Cell, Real> SharedVariableCellReal;

/*!
  \ingroup SharedVariable
  \brief Particle quantity of type real
*/
typedef SharedMeshVariableScalarRefT<Particle, Real> SharedVariableParticleReal;

/*!
  \ingroup SharedVariable
  \brief DoF quantity of type real
*/
typedef SharedMeshVariableScalarRefT<DoF, Real> SharedVariableDoFReal;

/*!
  \ingroup SharedVariable
  \brief Quantity at the cell center of type real
*/
typedef SharedMeshVariableScalarRefT<Cell, Real> SharedVariableCellReal;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
  \ingroup SharedVariable
  \brief Quantity at the node of type coordinates
*/
typedef SharedMeshVariableScalarRefT<Node, Real2> SharedVariableNodeReal2;

/*!
  \ingroup SharedVariable
  \brief Quantity at the node of type coordinates
*/
typedef SharedMeshVariableScalarRefT<Edge, Real2> SharedVariableEdgeReal2;

/*!
  \ingroup SharedVariable
  \brief Quantity on the faces of type coordinates
*/
typedef SharedMeshVariableScalarRefT<Face, Real2> SharedVariableFaceReal2;

/*!
  \ingroup SharedVariable
  \brief Quantity at the cell center of type coordinates
*/
typedef SharedMeshVariableScalarRefT<Cell, Real2> SharedVariableCellReal2;

/*!
  \ingroup SharedVariable
  \brief Particle quantity of type coordinates
*/
typedef SharedMeshVariableScalarRefT<Particle, Real2> SharedVariableParticleReal2;

/*!
  \ingroup SharedVariable
  \brief Quantity of DoF of coordinate type
*/
typedef SharedMeshVariableScalarRefT<DoF, Real2> SharedVariableDoFReal2;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
  \ingroup SharedVariable
  \brief Quantity of 3D coordinate type
*/
typedef SharedItemVariableScalarRefT<Real3> SharedVariableItemReal3;

/*!
  \ingroup SharedVariable
  \brief Quantity at the node of coordinate type
*/
typedef SharedMeshVariableScalarRefT<Node, Real3> SharedVariableNodeReal3;

/*!
  \ingroup SharedVariable
  \brief Quantity at the edge of coordinate type
*/
typedef SharedMeshVariableScalarRefT<Edge, Real3> SharedVariableEdgeReal3;

/*!
  \ingroup SharedVariable
  \brief Quantity on the faces of coordinate type
*/
typedef SharedMeshVariableScalarRefT<Face, Real3> SharedVariableFaceReal3;

/*!
  \ingroup SharedVariable
  \brief Quantity at the cell center of coordinate type
*/
typedef SharedMeshVariableScalarRefT<Cell, Real3> SharedVariableCellReal3;

/*!
  \ingroup SharedVariable
  \brief Particle-wise quantity of coordinate type
*/
typedef SharedMeshVariableScalarRefT<Particle, Real3> SharedVariableParticleReal3;

/*!
  \ingroup SharedVariable
  \brief Quantity of DoF of coordinate type
*/
typedef SharedMeshVariableScalarRefT<DoF, Real3> SharedVariableDoFReal3;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
  \ingroup SharedVariable
  \brief Quantity of real tensor type
*/
typedef SharedItemVariableScalarRefT<Real2x2> SharedVariableItemReal2x2;

/*!
  \ingroup SharedVariable
  \brief Quantity at the node of real tensor type
*/
typedef SharedMeshVariableScalarRefT<Node, Real2x2> SharedVariableNodeReal2x2;

/*!
  \ingroup SharedVariable
  \brief Quantity at the edge of real tensor type
*/
typedef SharedMeshVariableScalarRefT<Edge, Real2x2> SharedVariableEdgeReal2x2;

/*!
  \ingroup SharedVariable
  \brief Quantity on the faces of real tensor type
*/
typedef SharedMeshVariableScalarRefT<Face, Real2x2> SharedVariableFaceReal2x2;

/*!
  \ingroup SharedVariable
  \brief Quantity at the cell center of real tensor type
*/
typedef SharedMeshVariableScalarRefT<Cell, Real2x2> SharedVariableCellReal2x2;

/*!
  \ingroup SharedVariable
  \brief Particle-wise quantity of real tensor type
*/
typedef SharedMeshVariableScalarRefT<Particle, Real2x2> SharedVariableParticleReal2x2;

/*!
  \ingroup SharedVariable
  \brief Quantity of DoF of real tensor type
*/
typedef SharedMeshVariableScalarRefT<DoF, Real2x2> SharedVariableDoFReal2x2;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
  \ingroup SharedVariable
  \brief Quantity of real tensor type
*/
typedef SharedItemVariableScalarRefT<Real3x3> SharedVariableItemReal3x3;

/*!
  \ingroup SharedVariable
  \brief Quantity at the node of real tensor type
*/
typedef SharedMeshVariableScalarRefT<Node, Real3x3> SharedVariableNodeReal3x3;

/*!
  \ingroup SharedVariable
  \brief Quantity at the edge of real tensor type
*/
typedef SharedMeshVariableScalarRefT<Edge, Real3x3> SharedVariableEdgeReal3x3;

/*!
  \ingroup SharedVariable
  \brief Quantity on the faces of real tensor type
*/
typedef SharedMeshVariableScalarRefT<Face, Real3x3> SharedVariableFaceReal3x3;

/*!
  \ingroup SharedVariable
  \brief Quantity at the cell center of real tensor type
*/
typedef SharedMeshVariableScalarRefT<Cell, Real3x3> SharedVariableCellReal3x3;

/*!
  \ingroup SharedVariable
  \brief Particle-wise quantity of real tensor type
*/
typedef SharedMeshVariableScalarRefT<Particle, Real3x3> SharedVariableParticleReal3x3;

/*!
  \ingroup SharedVariable
  \brief Quantity of DoF of real tensor type
*/
typedef SharedMeshVariableScalarRefT<DoF, Real3x3> SharedVariableDoFReal3x3;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
  \ingroup SharedVariable
  \brief Quantity of integer type
*/
typedef SharedItemVariableScalarRefT<Integer> SharedVariableItemInteger;

/*!
  \ingroup SharedVariable
  \brief Quantity at the node of integer type
*/
typedef SharedMeshVariableScalarRefT<Node, Integer> SharedVariableNodeInteger;

/*!
  \ingroup SharedVariable
  \brief Quantity at the edge of integer type
*/
typedef SharedMeshVariableScalarRefT<Edge, Integer> SharedVariableEdgeInteger;

/*!
  \ingroup SharedVariable
  \brief Quantity on the faces of integer type
*/
typedef SharedMeshVariableScalarRefT<Face, Integer> SharedVariableFaceInteger;

/*!
  \ingroup SharedVariable
  \brief Quantity at the cell center of integer type
*/
typedef SharedMeshVariableScalarRefT<Cell, Integer> SharedVariableCellInteger;

/*!
  \ingroup SharedVariable
  \brief Particle-wise quantity of integer type
*/
typedef SharedMeshVariableScalarRefT<Particle, Integer> SharedVariableParticleInteger;

/*!
  \ingroup SharedVariable
  \brief Quantity of DoF of integer type
*/
typedef SharedMeshVariableScalarRefT<DoF, Integer> SharedVariableDoFInteger;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
  \ingroup SharedVariable
  \brief Quantity of 32-bit integer type
*/
typedef SharedItemVariableScalarRefT<Int32> SharedVariableItemInt32;

/*!
  \ingroup SharedVariable
  \brief Quantity at the node of 32-bit integer type
*/
typedef SharedMeshVariableScalarRefT<Node, Int32> SharedVariableNodeInt32;

/*!
  \ingroup SharedVariable
  \brief Quantity at the edge of 32-bit integer type
*/
typedef SharedMeshVariableScalarRefT<Edge, Int32> SharedVariableEdgeInt32;

/*!
  \ingroup SharedVariable
  \brief Quantity on the faces of 32-bit integer type
*/
typedef SharedMeshVariableScalarRefT<Face, Int32> SharedVariableFaceInt32;

/*!
  \ingroup SharedVariable
  \brief Quantity at the cell center of 32-bit integer type
*/
typedef SharedMeshVariableScalarRefT<Cell, Int32> SharedVariableCellInt32;

/*!
  \ingroup SharedVariable
  \brief Particle-wise quantity of 32-bit integer type
*/
typedef SharedMeshVariableScalarRefT<Particle, Int32> SharedVariableParticleInt32;

/*!
  \ingroup SharedVariable
  \brief Quantity of DoF of 32-bit integer type
*/
typedef SharedMeshVariableScalarRefT<DoF, Int32> SharedVariableDoFInt32;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
  \ingroup SharedVariable
  \brief Quantity of 16-bit integer type
*/
typedef SharedItemVariableScalarRefT<Int16> SharedVariableItemInt16;

/*!
  \ingroup SharedVariable
  \brief Quantity at the node of 16-bit integer type
*/
typedef SharedMeshVariableScalarRefT<Node, Int16> SharedVariableNodeInt16;

/*!
  \ingroup SharedVariable
  \brief Quantity at the edge of 16-bit integer type
*/
typedef SharedMeshVariableScalarRefT<Edge, Int16> SharedVariableEdgeInt16;

/*!
  \ingroup SharedVariable
  \brief Quantity on the faces of 16-bit integer type
*/
typedef SharedMeshVariableScalarRefT<Face, Int16> SharedVariableFaceInt16;

/*!
  \ingroup SharedVariable
  \brief Quantity at the cell center of 16-bit integer type
*/
typedef SharedMeshVariableScalarRefT<Cell, Int16> SharedVariableCellInt16;

/*!
  \ingroup SharedVariable
  \brief Particle-wise quantity of 16-bit integer type
*/
typedef SharedMeshVariableScalarRefT<Particle, Int16> SharedVariableParticleInt16;

/*!
  \ingroup SharedVariable
  \brief Quantity of DoF of 16-bit integer type
*/
typedef SharedMeshVariableScalarRefT<DoF, Int16> SharedVariableDoFInt16;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
  \ingroup SharedVariable
  \brief Quantity of 64-bit integer type
*/
typedef SharedItemVariableScalarRefT<Int64> SharedVariableItemInt64;

/*!
  \ingroup SharedVariable
  \brief Quantity at the node of 64-bit integer type
*/
typedef SharedMeshVariableScalarRefT<Node, Int64> SharedVariableNodeInt64;

/*!
  \ingroup SharedVariable
  \brief Quantity at the edge of 64-bit integer type
*/
typedef SharedMeshVariableScalarRefT<Edge, Int64> SharedVariableEdgeInt64;

/*!
  \ingroup SharedVariable
  \brief Quantity on the faces of 64-bit integer type
*/
typedef SharedMeshVariableScalarRefT<Face, Int64> SharedVariableFaceInt64;

/*!
  \ingroup SharedVariable
  \brief Quantity at the cell center of 64-bit integer type
*/
typedef SharedMeshVariableScalarRefT<Cell, Int64> SharedVariableCellInt64;

/*!
  \ingroup SharedVariable
  \brief Particle-wise quantity of 64-bit integer type
*/
typedef SharedMeshVariableScalarRefT<Particle, Int64> SharedVariableParticleInt64;

/*!
  \ingroup SharedVariable
  \brief Quantity of DoF of 64-bit integer type
*/
typedef SharedMeshVariableScalarRefT<DoF, Int64> SharedVariableDoFInt64;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
  \ingroup SharedVariable
  \brief Quantity at the node of natural integer type
  \deprecated Use #VariableNodeInteger instead
*/
typedef SharedMeshVariableScalarRefT<Node, Integer> SharedVariableNodeInteger;

/*!
  \ingroup SharedVariable
  \brief Quantity on the faces of natural integer type
  \deprecated Use #VariableFaceInteger instead
*/
typedef SharedMeshVariableScalarRefT<Face, Integer> SharedVariableFaceInteger;

/*!
  \ingroup SharedVariable
  \brief Quantity at the cell center of natural integer type
  \deprecated Use #VariableCellInteger instead
*/
typedef SharedMeshVariableScalarRefT<Cell, Integer> SharedVariableCellInteger;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
  \ingroup SharedVariable
  \brief Quantity of 64-bit integer type
*/
typedef ItemPartialVariableScalarRefT<Byte> PartialVariableItemByte;

/*!
  \ingroup SharedVariable
  \brief Quantity of 64-bit integer type
*/
typedef SharedItemVariableScalarRefT<Byte> SharedVariableItemByte;

/*!
  \ingroup SharedVariable
  \brief Quantity at the node of byte type
*/
typedef SharedMeshVariableScalarRefT<Node, Byte> SharedVariableNodeByte;

/*!
  \ingroup SharedVariable
  \brief Quantity at the edge of byte type
*/
typedef SharedMeshVariableScalarRefT<Edge, Byte> SharedVariableEdgeByte;

/*!
  \ingroup SharedVariable
  \brief Quantity on the faces of byte type
*/
typedef SharedMeshVariableScalarRefT<Face, Byte> SharedVariableFaceByte;

/*!
  \ingroup SharedVariable
  \brief Quantity at the cell center of byte type
*/
typedef SharedMeshVariableScalarRefT<Cell, Byte> SharedVariableCellByte;

/*!
  \ingroup SharedVariable
  \brief Particle-wise quantity of byte type
*/
typedef SharedMeshVariableScalarRefT<Particle, Byte> SharedVariableParticleByte;

/*!
  \ingroup SharedVariable
  \brief Quantity of DoF of byte type
*/
typedef SharedMeshVariableScalarRefT<DoF, Byte> SharedVariableDoFByte;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
  \ingroup SharedVariable
  \brief Quantity at the node of boolean type
*/
typedef SharedMeshVariableScalarRefT<Node, Byte> SharedVariableNodeBool;

/*!
  \ingroup SharedVariable
  \brief Quantity at the edge of boolean type
*/
typedef SharedMeshVariableScalarRefT<Edge, Byte> SharedVariableEdgeBool;

/*!
  \ingroup SharedVariable
  \brief Quantity on the faces of boolean type
*/
typedef SharedMeshVariableScalarRefT<Face, Byte> SharedVariableFaceBool;

/*!
  \ingroup SharedVariable
  \brief Quantity at the cell center of boolean type
*/
typedef SharedMeshVariableScalarRefT<Cell, Byte> SharedVariableCellBool;

/*!
  \ingroup SharedVariable
  \brief Particle-wise quantity of boolean type
*/
typedef SharedMeshVariableScalarRefT<Particle, Byte> SharedVariableParticleBool;

/*!
  \ingroup SharedVariable
  \brief Quantity of DoF of boolean type
*/
typedef SharedMeshVariableScalarRefT<DoF, Byte> SharedVariableDoFBool;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
