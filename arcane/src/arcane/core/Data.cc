// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Data.cc                                                     (C) 2000-2023 */
/*                                                                           */
/* Types liés aux 'IData'.                                                   */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/IData.h"

#include "arcane/utils/String.h"
#include "arcane/utils/NotSupportedException.h"

#include "arcane/IDataFactory.h"
#include "arcane/IDataStorageFactory.h"
#include "arcane/IDataVisitor.h"
#include "arcane/ISerializedData.h"
#include "arcane/core/internal/IDataInternal.h"

#include "arccore/base/ReferenceCounterImpl.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore
{
ARCCORE_DEFINE_REFERENCE_COUNTED_CLASS(Arcane::IData);
ARCCORE_DEFINE_REFERENCE_COUNTED_CLASS(Arcane::ISerializedData);
} // namespace Arccore

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void IDataVisitor::
applyDataVisitor(IMultiArray2Data*)
{
  ARCANE_THROW(NotSupportedException, "using applyDataVisitor with IMultiArray2Data is no longer supported");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void IData::
visitMultiArray2(IMultiArray2DataVisitor*)
{
  ARCANE_THROW(NotSupportedException, "Visiting IMultiArray2Data is no longer supported");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
