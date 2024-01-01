// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshVariableTplArray.cc                                     (C) 2000-2024 */
/*                                                                           */
/* Instanciation des classes templates des variables tableaux du maillage.   */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/MeshVariableArrayRefT.H"

#include "arcane/utils/BFloat16.h"
#include "arcane/utils/Float16.h"

#include "arcane/core/MeshPartialVariableArrayRefT.H"
#include "arcane/core/VariableFactoryRegisterer.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template class ARCANE_TEMPLATE_EXPORT ItemVariableArrayRefT<Byte>;
template class ARCANE_TEMPLATE_EXPORT ItemVariableArrayRefT<Int16>;
template class ARCANE_TEMPLATE_EXPORT ItemVariableArrayRefT<Int32>;
template class ARCANE_TEMPLATE_EXPORT ItemVariableArrayRefT<Int64>;
template class ARCANE_TEMPLATE_EXPORT ItemVariableArrayRefT<Real>;
template class ARCANE_TEMPLATE_EXPORT ItemVariableArrayRefT<Real2>;
template class ARCANE_TEMPLATE_EXPORT ItemVariableArrayRefT<Real3>;
template class ARCANE_TEMPLATE_EXPORT ItemVariableArrayRefT<Real2x2>;
template class ARCANE_TEMPLATE_EXPORT ItemVariableArrayRefT<Real3x3>;

template class ARCANE_TEMPLATE_EXPORT ItemPartialVariableArrayRefT<Byte>;
template class ARCANE_TEMPLATE_EXPORT ItemPartialVariableArrayRefT<Int16>;
template class ARCANE_TEMPLATE_EXPORT ItemPartialVariableArrayRefT<Int32>;
template class ARCANE_TEMPLATE_EXPORT ItemPartialVariableArrayRefT<Int64>;
template class ARCANE_TEMPLATE_EXPORT ItemPartialVariableArrayRefT<Real>;
template class ARCANE_TEMPLATE_EXPORT ItemPartialVariableArrayRefT<Real2>;
template class ARCANE_TEMPLATE_EXPORT ItemPartialVariableArrayRefT<Real3>;
template class ARCANE_TEMPLATE_EXPORT ItemPartialVariableArrayRefT<Real2x2>;
template class ARCANE_TEMPLATE_EXPORT ItemPartialVariableArrayRefT<Real3x3>;

#define ARCANE_INSTANTIATE_MESHVARIABLE_ARRAY(datatype) \
template class ARCANE_TEMPLATE_EXPORT MeshVariableArrayRefT<Node,datatype>; \
template class ARCANE_TEMPLATE_EXPORT MeshVariableArrayRefT<Edge,datatype>; \
template class ARCANE_TEMPLATE_EXPORT MeshVariableArrayRefT<Face,datatype>; \
template class ARCANE_TEMPLATE_EXPORT MeshVariableArrayRefT<Cell,datatype>; \
template class ARCANE_TEMPLATE_EXPORT MeshVariableArrayRefT<Particle,datatype>; \
template class ARCANE_TEMPLATE_EXPORT MeshVariableArrayRefT<DoF,datatype>; \
template class ARCANE_TEMPLATE_EXPORT MeshPartialVariableArrayRefT<Node,datatype>; \
template class ARCANE_TEMPLATE_EXPORT MeshPartialVariableArrayRefT<Edge,datatype>; \
template class ARCANE_TEMPLATE_EXPORT MeshPartialVariableArrayRefT<Face,datatype>; \
template class ARCANE_TEMPLATE_EXPORT MeshPartialVariableArrayRefT<Cell,datatype>; \
template class ARCANE_TEMPLATE_EXPORT MeshPartialVariableArrayRefT<Particle,datatype>; \
template class ARCANE_TEMPLATE_EXPORT MeshPartialVariableArrayRefT<DoF,datatype>

/*---------------------------------------------------------------------------*/

ARCANE_INSTANTIATE_MESHVARIABLE_ARRAY(Byte);
ARCANE_INSTANTIATE_MESHVARIABLE_ARRAY(Int8);
ARCANE_INSTANTIATE_MESHVARIABLE_ARRAY(Int16);
ARCANE_INSTANTIATE_MESHVARIABLE_ARRAY(Int32);
ARCANE_INSTANTIATE_MESHVARIABLE_ARRAY(Int64);
ARCANE_INSTANTIATE_MESHVARIABLE_ARRAY(BFloat16);
ARCANE_INSTANTIATE_MESHVARIABLE_ARRAY(Float16);
ARCANE_INSTANTIATE_MESHVARIABLE_ARRAY(Float32);
ARCANE_INSTANTIATE_MESHVARIABLE_ARRAY(Real);
ARCANE_INSTANTIATE_MESHVARIABLE_ARRAY(Real2);
ARCANE_INSTANTIATE_MESHVARIABLE_ARRAY(Real2x2);
ARCANE_INSTANTIATE_MESHVARIABLE_ARRAY(Real3);
ARCANE_INSTANTIATE_MESHVARIABLE_ARRAY(Real3x3);

/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
