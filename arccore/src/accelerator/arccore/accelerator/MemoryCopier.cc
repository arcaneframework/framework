// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MemoryCopier.cc                                             (C) 2000-2025 */
/*                                                                           */
/* Fonctions diverses de copie mémoire.                                      */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/accelerator/internal/AcceleratorMemoryCopier.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator::impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{
AcceleratorSpecificMemoryCopyList m_singleton_instance;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AcceleratorSpecificMemoryCopyList::
AcceleratorSpecificMemoryCopyList()
{
  using namespace Arcane::impl;
  GlobalMemoryCopyList::setAcceleratorInstance(this);

  // Pour raccourcir les temps de compilation, les instantiations
  // explicites sont faites dans plusieurs fichiers.
  addExplicitTemplate1();
  addExplicitTemplate2();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
