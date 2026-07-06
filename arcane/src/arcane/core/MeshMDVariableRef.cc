// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshMDVariableRef.cc                                        (C) 2000-2026 */
/*                                                                           */
/* Class managing a multi-dimensional variable on a mesh entity.             */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/MeshMDVariableRef.h"

#include "arcane/utils/MDDim.h"

#include "arcane/core/Item.h"
#include "arcane/core/MeshMatrixMDVariableRef.h"
#include "arcane/core/MeshVectorMDVariableRef.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Explicit instantiation to check compilation

template class MeshMDVariableRefT<Cell, Real, MDDim2>;
template class MeshMDVariableRefT<Cell, Real, MDDim3>;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
