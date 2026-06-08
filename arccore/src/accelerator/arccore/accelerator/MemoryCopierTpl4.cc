// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MemoryCopierTpl4.cc                                         (C) 2000-2026 */
/*                                                                           */
/* Memory copy functions on accelerator.                                     */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/accelerator/internal/AcceleratorMemoryCopier.h"

namespace Arcane::Accelerator::Impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AcceleratorSpecificMemoryCopyList::
addExplicitTemplate4()
{
  using namespace Arcane::impl;

  //! Adds specific implementations for the first common sizes
  addCopier<SpecificType<Int64, ExtentValue<6>>>(); // 48
  addCopier<SpecificType<Int64, ExtentValue<7>>>(); // 56
  addCopier<SpecificType<Int64, ExtentValue<8>>>(); // 64
  addCopier<SpecificType<Int64, ExtentValue<9>>>(); // 72
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
