// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshVariableTplArray.cc                                     (C) 2000-2025 */
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

#define ARCANE_INSTANTIATE_MESHVARIABLE_ARRAY(datatype) \
template class ARCANE_TEMPLATE_EXPORT ItemVariableArrayRefT<datatype>;\
template class ARCANE_TEMPLATE_EXPORT ItemPartialVariableArrayRefT<datatype>;\
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
