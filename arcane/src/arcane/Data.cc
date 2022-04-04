// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Data.cc                                                     (C) 2000-2021 */
/*                                                                           */
/* Types liés aux 'IData'.                                                   */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/IDataFactory.h"
#include "arcane/IDataStorageFactory.h"
#include "arcane/IData.h"
#include "arcane/ISerializedData.h"
#include "arcane/core/internal/IDataInternal.h"

#include "arccore/base/ReferenceCounterImpl.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore
{
ARCCORE_DEFINE_REFERENCE_COUNTED_CLASS(Arcane::IData);
ARCCORE_DEFINE_REFERENCE_COUNTED_CLASS(Arcane::ISerializedData);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
