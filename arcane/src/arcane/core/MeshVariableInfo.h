// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshVariableInfo.h                                          (C) 2000-2025 */
/*                                                                           */
/* Generic information for mesh variables.                                   */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_MESHVARIABLEINFO_H
#define ARCANE_CORE_MESHVARIABLEINFO_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/MeshVariableRef.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Generic information about the types of a mesh variable.
 *
 * This class must be specialized for each variable type:
 * - \a MeshItem entity type: Cell, Node or Face
 * - \a ValueType variable type: Real, Int64, Int32, Real3, Real3x3 
 * - \a Dimension variable dimension: 0 for scalars, 1 for arrays.
 */
template <class MeshItem, class ValueType, int Dimension>
class MeshVariableInfoT
{
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <>
class MeshVariableInfoT<Node, Real, 0>
{
 public:

  //! Type of the variable reference
  typedef MeshVariableScalarRefT<Node, Real> RefType;
  //! Type of the private part of the variable
  typedef VariableArrayT<Real> PrivateType;
};
template <>
class MeshVariableInfoT<Node, Real, 1>
{
 public:

  //! Type of the variable reference
  typedef MeshVariableArrayRefT<Node, Real> RefType;
  //! Type of the private part of the variable
  typedef Array2VariableT<Real> PrivateType;
};
template <>
class MeshVariableInfoT<Node, Real2, 0>
{
 public:

  //! Type of the variable reference
  typedef MeshVariableScalarRefT<Node, Real2> RefType;
  //! Type of the private part of the variable
  typedef VariableArrayT<Real2> PrivateType;
};
template <>
class MeshVariableInfoT<Node, Real2, 1>
{
 public:

  //! Type of the variable reference
  typedef MeshVariableArrayRefT<Node, Real2> RefType;
  //! Type of the private part of the variable
  typedef Array2VariableT<Real2> PrivateType;
};
template <>
class MeshVariableInfoT<Node, Real2x2, 0>
{
 public:

  //! Type of the variable reference
  typedef MeshVariableScalarRefT<Node, Real2x2> RefType;
  //! Type of the private part of the variable
  typedef VariableArrayT<Real2x2> PrivateType;
};
template <>
class MeshVariableInfoT<Node, Real2x2, 1>
{
 public:

  //! Type of the variable reference
  typedef MeshVariableArrayRefT<Node, Real2x2> RefType;
  //! Type of the private part of the variable
  typedef Array2VariableT<Real2x2> PrivateType;
};
template <>
class MeshVariableInfoT<Node, Real3, 0>
{
 public:

  //! Type of the variable reference
  typedef MeshVariableScalarRefT<Node, Real3> RefType;
  //! Type of the private part of the variable
  typedef VariableArrayT<Real3> PrivateType;
};
template <>
class MeshVariableInfoT<Node, Real3, 1>
{
 public:

  //! Type of the variable reference
  typedef MeshVariableArrayRefT<Node, Real3> RefType;
  //! Type of the private part of the variable
  typedef Array2VariableT<Real3> PrivateType;
};
template <>
class MeshVariableInfoT<Node, Real3x3, 0>
{
 public:

  //! Type of the variable reference
  typedef MeshVariableScalarRefT<Node, Real3x3> RefType;
  //! Type of the private part of the variable
  typedef VariableArrayT<Real3x3> PrivateType;
};
template <>
class MeshVariableInfoT<Node, Real3x3, 1>
{
 public:

  //! Type of the variable reference
  typedef MeshVariableArrayRefT<Node, Real3x3> RefType;
  //! Type of the private part of the variable
  typedef Array2VariableT<Real3x3> PrivateType;
};
template <>
class MeshVariableInfoT<Node, Int32, 0>
{
 public:

  //! Type of the variable reference
  typedef MeshVariableScalarRefT<Node, Int32> RefType;
  //! Type of the private part of the variable
  typedef VariableArrayT<Int32> PrivateType;
};
template <>
class MeshVariableInfoT<Node, Int32, 1>
{
 public:

  //! Type of the variable reference
  typedef MeshVariableArrayRefT<Node, Int32> RefType;
  //! Type of the private part of the variable
  typedef Array2VariableT<Int32> PrivateType;
};
template <>
class MeshVariableInfoT<Node, Int64, 0>
{
 public:

  //! Type of the variable reference
  typedef MeshVariableScalarRefT<Node, Int64> RefType;
  //! Type of the private part of the variable
  typedef VariableArrayT<Int64> PrivateType;
};
template <>
class MeshVariableInfoT<Node, Int64, 1>
{
 public:

  //! Type of the variable reference
  typedef MeshVariableArrayRefT<Node, Int64> RefType;
  //! Type of the private part of the variable
  typedef Array2VariableT<Int64> PrivateType;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <>
class MeshVariableInfoT<Edge, Real, 0>
{
 public:

  //! Type of the variable reference
  typedef MeshVariableScalarRefT<Edge, Real> RefType;
  //! Type of the private part of the variable
  typedef VariableArrayT<Real> PrivateType;
};
template <>
class MeshVariableInfoT<Edge, Real, 1>
{
 public:

  //! Type of the variable reference
  typedef MeshVariableArrayRefT<Edge, Real> RefType;
  //! Type of the private part of the variable
  typedef Array2VariableT<Real> PrivateType;
};
template <>
class MeshVariableInfoT<Edge, Real3, 0>
{
 public:

  //! Type of the variable reference
  typedef MeshVariableScalarRefT<Edge, Real3> RefType;
  //! Type of the private part of the variable
  typedef VariableArrayT<Real3> PrivateType;
};
template <>
class MeshVariableInfoT<Edge, Real3, 1>
{
 public:

  //! Type of the variable reference
  typedef MeshVariableArrayRefT<Edge, Real3> RefType;
  //! Type of the private part of the variable
  typedef Array2VariableT<Real3> PrivateType;
};
template <>
class MeshVariableInfoT<Edge, Real2, 0>
{
 public:

  //! Type of the variable reference
  typedef MeshVariableScalarRefT<Edge, Real2> RefType;
  //! Type of the private part of the variable
  typedef VariableArrayT<Real2> PrivateType;
};
template <>
class MeshVariableInfoT<Edge, Real2, 1>
{
 public:

  //! Type of the variable reference
  typedef MeshVariableArrayRefT<Edge, Real2> RefType;
  //! Type of the private part of the variable
  typedef Array2VariableT<Real2> PrivateType;
};
template <>
class MeshVariableInfoT<Edge, Real2x2, 0>
{
 public:

  //! Type of the variable reference
  typedef MeshVariableScalarRefT<Edge, Real2x2> RefType;
  //! Type of the private part of the variable
  typedef VariableArrayT<Real2x2> PrivateType;
};
template <>
class MeshVariableInfoT<Edge, Real2x2, 1>
{
 public:

  //! Type of the variable reference
  typedef MeshVariableArrayRefT<Edge, Real2x2> RefType;
  //! Type of the private part of the variable
  typedef Array2VariableT<Real2x2> PrivateType;
};
template <>
class MeshVariableInfoT<Edge, Real3x3, 0>
{
 public:

  //! Type of the variable reference
  typedef MeshVariableScalarRefT<Edge, Real3x3> RefType;
  //! Type of the private part of the variable
  typedef VariableArrayT<Real3x3> PrivateType;
};
template <>
class MeshVariableInfoT<Edge, Real3x3, 1>
{
 public:

  //! Type of the variable reference
  typedef MeshVariableArrayRefT<Edge, Real3x3> RefType;
  //! Type of the private part of the variable
  typedef Array2VariableT<Real3x3> PrivateType;
};
template <>
class MeshVariableInfoT<Edge, Int32, 0>
{
 public:

  //! Type of the variable reference
  typedef MeshVariableScalarRefT<Edge, Int32> RefType;
  //! Type of the private part of the variable
  typedef VariableArrayT<Int32> PrivateType;
};
template <>
class MeshVariableInfoT<Edge, Int32, 1>
{
 public:

  //! Type of the variable reference
  typedef MeshVariableArrayRefT<Edge, Int32> RefType;
  //! Type of the private part of the variable
  typedef Array2VariableT<Int32> PrivateType;
};
template <>
class MeshVariableInfoT<Edge, Int64, 0>
{
 public:

  //! Type of the variable reference
  typedef MeshVariableScalarRefT<Edge, Int64> RefType;
  //! Type of the private part of the variable
  typedef VariableArrayT<Int64> PrivateType;
};
template <>
class MeshVariableInfoT<Edge, Int64, 1>
{
 public:

  //! Type of the variable reference
  typedef MeshVariableArrayRefT<Edge, Int64> RefType;
  //! Type of the private part of the variable
  typedef Array2VariableT<Int64> PrivateType;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <>
class MeshVariableInfoT<Face, Real, 0>
{
 public:

  //! Type of the variable reference
  typedef MeshVariableScalarRefT<Face, Real> RefType;
  //! Type of the private part of the variable
  typedef VariableArrayT<Real> PrivateType;
};
template <>
class MeshVariableInfoT<Face, Real, 1>
{
 public:

  //! Type of the variable reference
  typedef MeshVariableArrayRefT<Face, Real> RefType;
  //! Type of the private part of the variable
  typedef Array2VariableT<Real> PrivateType;
};
template <>
class MeshVariableInfoT<Face, Real2, 0>
{
 public:

  //! Type of the variable reference
  typedef MeshVariableScalarRefT<Face, Real2> RefType;
  //! Type of the private part of the variable
  typedef VariableArrayT<Real2> PrivateType;
};
template <>
class MeshVariableInfoT<Face, Real2, 1>
{
 public:

  //! Type of the variable reference
  typedef MeshVariableArrayRefT<Face, Real2> RefType;
  //! Type of the private part of the variable
  typedef Array2VariableT<Real2> PrivateType;
};
template <>
class MeshVariableInfoT<Face, Real3, 0>
{
 public:

  //! Type of the variable reference
  typedef MeshVariableScalarRefT<Face, Real3> RefType;
  //! Type of the private part of the variable
  typedef VariableArrayT<Real3> PrivateType;
};
template <>
class MeshVariableInfoT<Face, Real3, 1>
{
 public:

  //! Type of the variable reference
  typedef MeshVariableArrayRefT<Face, Real3> RefType;
  //! Type of the variable's private part
  typedef Array2VariableT<Real3> PrivateType;
};
template <>
class MeshVariableInfoT<Face, Real2x2, 0>
{
 public:

  //! Type of the variable reference
  typedef MeshVariableScalarRefT<Face, Real2x2> RefType;
  //! Type of the variable's private part
  typedef VariableArrayT<Real2x2> PrivateType;
};
template <>
class MeshVariableInfoT<Face, Real2x2, 1>
{
 public:

  //! Type of the variable reference
  typedef MeshVariableArrayRefT<Face, Real2x2> RefType;
  //! Type of the variable's private part
  typedef Array2VariableT<Real2x2> PrivateType;
};
template <>
class MeshVariableInfoT<Face, Real3x3, 0>
{
 public:

  //! Type of the variable reference
  typedef MeshVariableScalarRefT<Face, Real3x3> RefType;
  //! Type of the variable's private part
  typedef VariableArrayT<Real3x3> PrivateType;
};
template <>
class MeshVariableInfoT<Face, Real3x3, 1>
{
 public:

  //! Type of the variable reference
  typedef MeshVariableArrayRefT<Face, Real3x3> RefType;
  //! Type of the variable's private part
  typedef Array2VariableT<Real3x3> PrivateType;
};
template <>
class MeshVariableInfoT<Face, Int32, 0>
{
 public:

  //! Type of the variable reference
  typedef MeshVariableScalarRefT<Face, Int32> RefType;
  //! Type of the variable's private part
  typedef VariableArrayT<Int32> PrivateType;
};
template <>
class MeshVariableInfoT<Face, Int32, 1>
{
 public:

  //! Type of the variable reference
  typedef MeshVariableArrayRefT<Face, Int32> RefType;
  //! Type of the variable's private part
  typedef Array2VariableT<Int32> PrivateType;
};
template <>
class MeshVariableInfoT<Face, Int64, 0>
{
 public:

  //! Type of the variable reference
  typedef MeshVariableScalarRefT<Face, Int64> RefType;
  //! Type of the variable's private part
  typedef VariableArrayT<Int64> PrivateType;
};
template <>
class MeshVariableInfoT<Face, Int64, 1>
{
 public:

  //! Type of the variable reference
  typedef MeshVariableArrayRefT<Face, Int64> RefType;
  //! Type of the variable's private part
  typedef Array2VariableT<Int64> PrivateType;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <>
class MeshVariableInfoT<Cell, Real, 0>
{
 public:

  //! Type of the variable reference
  typedef MeshVariableScalarRefT<Cell, Real> RefType;
  //! Type of the variable's private part
  typedef VariableArrayT<Real> PrivateType;
};
template <>
class MeshVariableInfoT<Cell, Real, 1>
{
 public:

  //! Type of the variable reference
  typedef MeshVariableArrayRefT<Cell, Real> RefType;
  //! Type of the variable's private part
  typedef Array2VariableT<Real> PrivateType;
};
template <>
class MeshVariableInfoT<Cell, Real3, 0>
{
 public:

  //! Type of the variable reference
  typedef MeshVariableScalarRefT<Cell, Real3> RefType;
  //! Type of the variable's private part
  typedef VariableArrayT<Real3> PrivateType;
};
template <>
class MeshVariableInfoT<Cell, Real3, 1>
{
 public:

  //! Type of the variable reference
  typedef MeshVariableArrayRefT<Cell, Real3> RefType;
  //! Type of the variable's private part
  typedef Array2VariableT<Real3> PrivateType;
};
template <>
class MeshVariableInfoT<Cell, Real2, 0>
{
 public:

  //! Type of the variable reference
  typedef MeshVariableScalarRefT<Cell, Real2> RefType;
  //! Type of the variable's private part
  typedef VariableArrayT<Real2> PrivateType;
};
template <>
class MeshVariableInfoT<Cell, Real2, 1>
{
 public:

  //! Type of the variable reference
  typedef MeshVariableArrayRefT<Cell, Real2> RefType;
  //! Type of the variable's private part
  typedef Array2VariableT<Real2> PrivateType;
};
template <>
class MeshVariableInfoT<Cell, Real2x2, 0>
{
 public:

  //! Type of the variable reference
  typedef MeshVariableScalarRefT<Cell, Real2x2> RefType;
  //! Type of the variable's private part
  typedef VariableArrayT<Real2x2> PrivateType;
};
template <>
class MeshVariableInfoT<Cell, Real2x2, 1>
{
 public:

  //! Type of the variable reference
  typedef MeshVariableArrayRefT<Cell, Real2x2> RefType;
  //! Type of the variable's private part
  typedef Array2VariableT<Real2x2> PrivateType;
};
template <>
class MeshVariableInfoT<Cell, Real3x3, 0>
{
 public:

  //! Type of the variable reference
  typedef MeshVariableScalarRefT<Cell, Real3x3> RefType;
  //! Type of the variable's private part
  typedef VariableArrayT<Real3x3> PrivateType;
};
template <>
class MeshVariableInfoT<Cell, Real3x3, 1>
{
 public:

  //! Type of the variable reference
  typedef MeshVariableArrayRefT<Cell, Real3x3> RefType;
  //! Type of the variable's private part
  typedef Array2VariableT<Real3x3> PrivateType;
};
template <>
class MeshVariableInfoT<Cell, Int32, 0>
{
 public:

  //! Type of the variable reference
  typedef MeshVariableScalarRefT<Cell, Int32> RefType;
  //! Type of the variable's private part
  typedef VariableArrayT<Int32> PrivateType;
};
template <>
class MeshVariableInfoT<Cell, Int32, 1>
{
 public:

  //! Type of the variable reference
  typedef MeshVariableArrayRefT<Cell, Int32> RefType;
  //! Type of the variable's private part
  typedef Array2VariableT<Int32> PrivateType;
};
template <>
class MeshVariableInfoT<Cell, Int64, 0>
{
 public:

  //! Type of the variable reference
  typedef MeshVariableScalarRefT<Cell, Int64> RefType;
  //! Type of the variable's private part
  typedef VariableArrayT<Int64> PrivateType;
};
template <>
class MeshVariableInfoT<Cell, Int64, 1>
{
 public:

  //! Type of the variable reference
  typedef MeshVariableArrayRefT<Cell, Int64> RefType;
  //! Type of the variable's private part
  typedef Array2VariableT<Int64> PrivateType;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <>
class MeshVariableInfoT<DoF, Real, 0>
{
 public:

  //! Type of the variable reference
  typedef MeshVariableScalarRefT<DoF, Real> RefType;
  //! Type of the variable's private part
  typedef VariableArrayT<Real> PrivateType;
};
template <>
class MeshVariableInfoT<DoF, Real, 1>
{
 public:

  //! Type of the variable reference
  typedef MeshVariableArrayRefT<DoF, Real> RefType;
  //! Type of the variable's private part
  typedef Array2VariableT<Real> PrivateType;
};
template <>
class MeshVariableInfoT<DoF, Real2, 0>
{
 public:

  //! Type of the variable reference
  typedef MeshVariableScalarRefT<DoF, Real2> RefType;
  //! Type of the variable's private part
  typedef VariableArrayT<Real2> PrivateType;
};
template <>
class MeshVariableInfoT<DoF, Real2, 1>
{
 public:

  //! Type of the variable reference
  typedef MeshVariableArrayRefT<DoF, Real2> RefType;
  //! Type of the variable's private part
  typedef Array2VariableT<Real2> PrivateType;
};
template <>
class MeshVariableInfoT<DoF, Real2x2, 0>
{
 public:

  //! Type of the variable reference
  typedef MeshVariableScalarRefT<DoF, Real2x2> RefType;
  //! Type of the variable's private part
  typedef VariableArrayT<Real2x2> PrivateType;
};
template <>
class MeshVariableInfoT<DoF, Real2x2, 1>
{
 public:

  //! Type of the variable reference
  typedef MeshVariableArrayRefT<DoF, Real2x2> RefType;
  //! Type of the variable's private part
  typedef Array2VariableT<Real2x2> PrivateType;
};
template <>
class MeshVariableInfoT<DoF, Real3, 0>
{
 public:

  //! Type of the variable reference
  typedef MeshVariableScalarRefT<DoF, Real3> RefType;
  //! Type of the variable's private part
  typedef VariableArrayT<Real3> PrivateType;
};
template <>
class MeshVariableInfoT<DoF, Real3, 1>
{
 public:

  //! Type of the variable reference
  typedef MeshVariableArrayRefT<DoF, Real3> RefType;
  //! Type of the variable's private part
  typedef Array2VariableT<Real3> PrivateType;
};
template <>
class MeshVariableInfoT<DoF, Real3x3, 0>
{
 public:

  //! Type of the variable reference
  typedef MeshVariableScalarRefT<DoF, Real3x3> RefType;
  //! Type of the variable's private part
  typedef VariableArrayT<Real3x3> PrivateType;
};
template <>
class MeshVariableInfoT<DoF, Real3x3, 1>
{
 public:

  //! Type of the variable reference
  typedef MeshVariableArrayRefT<DoF, Real3x3> RefType;
  //! Type of the variable's private part
  typedef Array2VariableT<Real3x3> PrivateType;
};
template <>
class MeshVariableInfoT<DoF, Int32, 0>
{
 public:

  //! Type of the variable reference
  typedef MeshVariableScalarRefT<DoF, Int32> RefType;
  //! Type of the variable's private part
  typedef VariableArrayT<Int32> PrivateType;
};
template <>
class MeshVariableInfoT<DoF, Int32, 1>
{
 public:

  //! Type of the variable reference
  typedef MeshVariableArrayRefT<DoF, Int32> RefType;
  //! Type of the variable's private part
  typedef Array2VariableT<Int32> PrivateType;
};
template <>
class MeshVariableInfoT<DoF, Int64, 0>
{
 public:

  //! Type of the variable reference
  typedef MeshVariableScalarRefT<DoF, Int64> RefType;
  //! Type of the variable's private part
  typedef VariableArrayT<Int64> PrivateType;
};
template <>
class MeshVariableInfoT<DoF, Int64, 1>
{
 public:

  //! Type of the variable reference
  typedef MeshVariableArrayRefT<DoF, Int64> RefType;
  //! Type of the variable's private part
  typedef Array2VariableT<Int64> PrivateType;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
