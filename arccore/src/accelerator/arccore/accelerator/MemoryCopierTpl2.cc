// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MemoryCopierTpl2.cc                                         (C) 2000-2026 */
/*                                                                           */
/* Memory copy functions on accelerator.                                     */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/accelerator/internal/AcceleratorMemoryCopier.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator::Impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AcceleratorSpecificMemoryCopyList::
addExplicitTemplate2()
{
  using namespace Arcane::impl;

  //! Adds specific implementations for the first common sizes
  addCopier<SpecificType<Int16, ExtentValue<3>>>(); // 6
  addCopier<SpecificType<std::byte, ExtentValue<7>>>(); // 7
  addCopier<SpecificType<Int64, ExtentValue<1>>>(); // 8
  addCopier<SpecificType<std::byte, ExtentValue<9>>>(); // 9
  addCopier<SpecificType<Int16, ExtentValue<5>>>(); // 10
  addCopier<SpecificType<Int32, ExtentValue<3>>>(); // 12
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator::Impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
