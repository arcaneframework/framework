// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MemoryCopierTpl4.cc                                         (C) 2000-2025 */
/*                                                                           */
/* Fonctions de copie mémoire sur accélérateur.                              */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/accelerator/internal/AcceleratorMemoryCopier.h"

namespace Arcane::Accelerator::impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void AcceleratorSpecificMemoryCopyList::
addExplicitTemplate4()
{
  using namespace Arcane::impl;

  //! Ajoute des implémentations spécifiques pour les premières tailles courantes
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
