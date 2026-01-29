// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MemoryCopierTpl1.cc                                         (C) 2000-2026 */
/*                                                                           */
/* Fonctions de copie mémoire sur accélérateur.                              */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/accelerator/internal/AcceleratorMemoryCopier.h"

namespace Arcane::Accelerator::Impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AcceleratorSpecificMemoryCopyList::
addExplicitTemplate1()
{
  using namespace Arcane::impl;

  //! Ajoute des implémentations spécifiques pour les premières tailles courantes
  addCopier<SpecificType<std::byte, ExtentValue<1>>>(); // 1
  addCopier<SpecificType<Int16, ExtentValue<1>>>(); // 2
  addCopier<SpecificType<std::byte, ExtentValue<3>>>(); // 3
  addCopier<SpecificType<Int32, ExtentValue<1>>>(); // 4
  addCopier<SpecificType<std::byte, ExtentValue<5>>>(); // 5
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator::Impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
