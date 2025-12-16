// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Initializer.h                                               (C) 2000-2025 */
/*                                                                           */
/* Classe pour initialiser le runtime accélérateur.                          */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_ACCELERATOR_INTERNAL_INITIALIZER_H
#define ARCCORE_ACCELERATOR_INTERNAL_INITIALIZER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/accelerator/AcceleratorGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe pour initialiser le runtime accélérateur.
 */
class ARCCORE_ACCELERATOR_EXPORT Initializer
{
 public:

  Initializer(bool use_accelerator, Int32 max_allowed_thread);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
