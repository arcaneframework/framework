// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AcceleratorInitializer.cc                                 (C) 2000-2026 */
/*                                                                           */
/* Initialiseur pour un runtime-accélérator.                                 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/accelerator/AcceleratorInitializer.h"

#include "arccore/accelerator/internal/Initializer.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AcceleratorInitializer::
AcceleratorInitializer()
: m_initializer(std::make_unique<Initializer>(false, 1))
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AcceleratorInitializer::
AcceleratorInitializer(bool use_accelerator, Int32 nb_thread)
: m_initializer(std::make_unique<Initializer>(use_accelerator, nb_thread))
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AcceleratorInitializer::
~AcceleratorInitializer()
{
  // Le destructeur doit être ici car la classe 'm_initializer' est opaque
  // et n'est pas connu dans le fichier d'en-tête
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Politique d'exécution initialisée par défaut
eExecutionPolicy AcceleratorInitializer::
executionPolicy() const
{
  return m_initializer->executionPolicy();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ITraceMng* AcceleratorInitializer::
traceMng() const
{
  return m_initializer->traceMng();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
