// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MaterialsCoreGlobal.cc                                      (C) 2000-2026 */
/*                                                                           */
/* General declarations of Arcane materials.                                 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/materials/IMeshMaterialVariable.h"
#include "arcane/core/materials/IMeshBlock.h"
#include "arcane/core/materials/IMeshComponent.h"
#include "arcane/core/materials/IMeshMaterial.h"
#include "arcane/core/materials/IMeshEnvironment.h"
#include "arcane/core/materials/MatVarIndex.h"
#include "arcane/core/materials/MatItem.h"
#include "arcane/core/materials/IEnumeratorTracer.h"
#include "arcane/core/materials/IMeshMaterialVariableFactoryMng.h"
#include "arcane/core/materials/IMeshMaterialVariableFactory.h"

// Not used directly but necessary for symbol export.
#include "arcane/core/materials/internal/IMeshComponentInternal.h"
#include "arcane/core/materials/internal/IMeshMaterialMngInternal.h"
#include "arcane/core/materials/ConstituentItemIndexedSelectionView.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IEnumeratorTracer* IEnumeratorTracer::m_singleton = nullptr;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void IEnumeratorTracer::
_setSingleton(IEnumeratorTracer* tracer)
{
  delete m_singleton;
  m_singleton = tracer;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
