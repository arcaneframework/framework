// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshVariableScalarRefTpl2.cc                                (C) 2000-2025 */
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

ARCANE_INSTANTIATE_MESHVARIABLE_SCALAR(Int8);
ARCANE_INSTANTIATE_MESHVARIABLE_SCALAR(Int16);
ARCANE_INSTANTIATE_MESHVARIABLE_SCALAR(Int32);
ARCANE_INSTANTIATE_MESHVARIABLE_SCALAR(Int64);
ARCANE_INSTANTIATE_MESHVARIABLE_SCALAR(Byte);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
