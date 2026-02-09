// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Initializer.h                                               (C) 2000-2026 */
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
 * \internal
 * \brief Classe interne pour initialiser le runtime accélérateur.
 */
class ARCCORE_ACCELERATOR_EXPORT Initializer
{
 public:

  Initializer(bool use_accelerator, Int32 max_allowed_thread);
  ~Initializer() noexcept(false);

 public:

  Initializer(const Initializer&) = delete;
  Initializer(Initializer&&) = delete;
  Initializer& operator=(const Initializer&) = delete;
  Initializer& operator=(Initializer&&) = delete;

 public:

  eExecutionPolicy executionPolicy() const { return m_policy; }

 private:

  eExecutionPolicy m_policy = eExecutionPolicy::Sequential;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
