// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* RegisterRuntimeInfo.h                                       (C) 2000-2024 */
/*                                                                           */
/* Informations pour initialiser le runtime accélérateur.                    */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ACCELERATOR_CORE_INTERNAL_REGISTERRUNTIMEINFO_H
#define ARCANE_ACCELERATOR_CORE_INTERNAL_REGISTERRUNTIMEINFO_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/accelerator/core/internal/AcceleratorCoreGlobalInternal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Informations pour initialiser le runtime accélérateur.
 */
class RegisterRuntimeInfo
{
 public:

  void setVerbose(bool v) { m_is_verbose = v; }
  bool isVerbose() const { return m_is_verbose; }

 private:

  bool m_is_verbose = false;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
