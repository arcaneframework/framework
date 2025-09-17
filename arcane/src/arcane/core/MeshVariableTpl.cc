// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshVariableTpl.cc                                          (C) 2000-2025 */
/*                                                                           */
/* Instanciation des classes templates des variables du maillage.            */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/MeshVariableScalarRef.inst.h"
#include "arcane/core/MeshPartialVariableScalarRef.inst.h"
#include "arcane/core/VariableFactoryRegisterer.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#define ARCANE_INSTANTIATE_MESHVARIABLE_SCALAR(datatype) \
template class ARCANE_TEMPLATE_EXPORT ItemVariableScalarRefT<datatype>;\
template class ARCANE_TEMPLATE_EXPORT ItemPartialVariableScalarRefT<datatype>;\
template class ARCANE_TEMPLATE_EXPORT MeshVariableScalarRefT<Node,datatype>; \
template class ARCANE_TEMPLATE_EXPORT MeshVariableScalarRefT<Edge,datatype>; \
template class ARCANE_TEMPLATE_EXPORT MeshVariableScalarRefT<Face,datatype>; \
template class ARCANE_TEMPLATE_EXPORT MeshVariableScalarRefT<Cell,datatype>; \
template class ARCANE_TEMPLATE_EXPORT MeshVariableScalarRefT<Particle,datatype>; \
template class ARCANE_TEMPLATE_EXPORT MeshVariableScalarRefT<DoF,datatype>; \
template class ARCANE_TEMPLATE_EXPORT MeshPartialVariableScalarRefT<Node,datatype>; \
template class ARCANE_TEMPLATE_EXPORT MeshPartialVariableScalarRefT<Edge,datatype>; \
template class ARCANE_TEMPLATE_EXPORT MeshPartialVariableScalarRefT<Face,datatype>; \
template class ARCANE_TEMPLATE_EXPORT MeshPartialVariableScalarRefT<Cell,datatype>; \
template class ARCANE_TEMPLATE_EXPORT MeshPartialVariableScalarRefT<Particle,datatype>; \
template class ARCANE_TEMPLATE_EXPORT MeshPartialVariableScalarRefT<DoF,datatype>

/*---------------------------------------------------------------------------*/

ARCANE_INSTANTIATE_MESHVARIABLE_SCALAR(Byte);
ARCANE_INSTANTIATE_MESHVARIABLE_SCALAR(Int8);
ARCANE_INSTANTIATE_MESHVARIABLE_SCALAR(Int16);
ARCANE_INSTANTIATE_MESHVARIABLE_SCALAR(Int32);
ARCANE_INSTANTIATE_MESHVARIABLE_SCALAR(Int64);
ARCANE_INSTANTIATE_MESHVARIABLE_SCALAR(BFloat16);
ARCANE_INSTANTIATE_MESHVARIABLE_SCALAR(Float16);
ARCANE_INSTANTIATE_MESHVARIABLE_SCALAR(Float32);
ARCANE_INSTANTIATE_MESHVARIABLE_SCALAR(Real);
ARCANE_INSTANTIATE_MESHVARIABLE_SCALAR(Real2);
ARCANE_INSTANTIATE_MESHVARIABLE_SCALAR(Real2x2);
ARCANE_INSTANTIATE_MESHVARIABLE_SCALAR(Real3);
ARCANE_INSTANTIATE_MESHVARIABLE_SCALAR(Real3x3);

/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
