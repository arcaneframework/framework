// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshVariableTpl.cc                                          (C) 2000-2016 */
/*                                                                           */
/* Instanciation des classes templates des variables du maillage.            */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/MeshVariableScalarRefT.H"
#include "arcane/MeshPartialVariableScalarRefT.H"
#include "arcane/VariableFactoryRegisterer.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template class ARCANE_TEMPLATE_EXPORT ItemVariableScalarRefT<Byte>;
template class ARCANE_TEMPLATE_EXPORT ItemVariableScalarRefT<Int16>;
template class ARCANE_TEMPLATE_EXPORT ItemVariableScalarRefT<Int32>;
template class ARCANE_TEMPLATE_EXPORT ItemVariableScalarRefT<Int64>;
template class ARCANE_TEMPLATE_EXPORT ItemVariableScalarRefT<Real>;
template class ARCANE_TEMPLATE_EXPORT ItemVariableScalarRefT<Real2>;
template class ARCANE_TEMPLATE_EXPORT ItemVariableScalarRefT<Real3>;
template class ARCANE_TEMPLATE_EXPORT ItemVariableScalarRefT<Real2x2>;
template class ARCANE_TEMPLATE_EXPORT ItemVariableScalarRefT<Real3x3>;

template class ARCANE_TEMPLATE_EXPORT ItemPartialVariableScalarRefT<Byte>;
template class ARCANE_TEMPLATE_EXPORT ItemPartialVariableScalarRefT<Int16>;
template class ARCANE_TEMPLATE_EXPORT ItemPartialVariableScalarRefT<Int32>;
template class ARCANE_TEMPLATE_EXPORT ItemPartialVariableScalarRefT<Int64>;
template class ARCANE_TEMPLATE_EXPORT ItemPartialVariableScalarRefT<Real>;
template class ARCANE_TEMPLATE_EXPORT ItemPartialVariableScalarRefT<Real2>;
template class ARCANE_TEMPLATE_EXPORT ItemPartialVariableScalarRefT<Real3>;
template class ARCANE_TEMPLATE_EXPORT ItemPartialVariableScalarRefT<Real2x2>;
template class ARCANE_TEMPLATE_EXPORT ItemPartialVariableScalarRefT<Real3x3>;

#define ARCANE_INSTANTIATE_MESHVARIABLE_SCALAR(datatype) \
template class ARCANE_TEMPLATE_EXPORT MeshVariableScalarRefT<Node,datatype>; \
template class ARCANE_TEMPLATE_EXPORT MeshVariableScalarRefT<Edge,datatype>; \
template class ARCANE_TEMPLATE_EXPORT MeshVariableScalarRefT<Face,datatype>; \
template class ARCANE_TEMPLATE_EXPORT MeshVariableScalarRefT<Cell,datatype>; \
template class ARCANE_TEMPLATE_EXPORT MeshVariableScalarRefT<Particle,datatype>; \
template class ARCANE_TEMPLATE_EXPORT MeshVariableScalarRefT<DualNode,datatype>; \
template class ARCANE_TEMPLATE_EXPORT MeshVariableScalarRefT<Link,datatype>; \
template class ARCANE_TEMPLATE_EXPORT MeshVariableScalarRefT<DoF,datatype>; \
template class ARCANE_TEMPLATE_EXPORT MeshPartialVariableScalarRefT<Node,datatype>; \
template class ARCANE_TEMPLATE_EXPORT MeshPartialVariableScalarRefT<Edge,datatype>; \
template class ARCANE_TEMPLATE_EXPORT MeshPartialVariableScalarRefT<Face,datatype>; \
template class ARCANE_TEMPLATE_EXPORT MeshPartialVariableScalarRefT<Cell,datatype>; \
template class ARCANE_TEMPLATE_EXPORT MeshPartialVariableScalarRefT<Particle,datatype>; \
template class ARCANE_TEMPLATE_EXPORT MeshPartialVariableScalarRefT<DualNode,datatype>; \
template class ARCANE_TEMPLATE_EXPORT MeshPartialVariableScalarRefT<Link,datatype>; \
template class ARCANE_TEMPLATE_EXPORT MeshPartialVariableScalarRefT<DoF,datatype>

/*---------------------------------------------------------------------------*/

ARCANE_INSTANTIATE_MESHVARIABLE_SCALAR(Byte);
ARCANE_INSTANTIATE_MESHVARIABLE_SCALAR(Int16);
ARCANE_INSTANTIATE_MESHVARIABLE_SCALAR(Int32);
ARCANE_INSTANTIATE_MESHVARIABLE_SCALAR(Int64);
ARCANE_INSTANTIATE_MESHVARIABLE_SCALAR(Real);
ARCANE_INSTANTIATE_MESHVARIABLE_SCALAR(Real2);
ARCANE_INSTANTIATE_MESHVARIABLE_SCALAR(Real2x2);
ARCANE_INSTANTIATE_MESHVARIABLE_SCALAR(Real3);
ARCANE_INSTANTIATE_MESHVARIABLE_SCALAR(Real3x3);

/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
