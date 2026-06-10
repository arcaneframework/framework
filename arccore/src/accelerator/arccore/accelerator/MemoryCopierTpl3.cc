// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MemoryCopierTpl3.cc                                         (C) 2000-2026 */
/*                                                                           */
/* Accelerator memory copy functions.                                        */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/accelerator/internal/AcceleratorMemoryCopier.h"

namespace Arcane::Accelerator::Impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AcceleratorSpecificMemoryCopyList::
addExplicitTemplate3()
{
  using namespace Arcane::impl;

  //! Adds specific implementations for the first common sizes
  addCopier<SpecificType<Int64, ExtentValue<2>>>(); // 16
  addCopier<SpecificType<Int64, ExtentValue<3>>>(); // 24
  addCopier<SpecificType<Int64, ExtentValue<4>>>(); // 32
  addCopier<SpecificType<Int64, ExtentValue<5>>>(); // 40
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator::Impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
