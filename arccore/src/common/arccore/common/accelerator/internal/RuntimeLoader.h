// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* RuntimeLoader.h                                             (C) 2000-2025 */
/*                                                                           */
/* Gestion du chargement du runtime accélérateur.                            */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_COMMON_ACCELERATOR_INTERNAL_RUNTIMELOADER_H
#define ARCCORE_COMMON_ACCELERATOR_INTERNAL_RUNTIMELOADER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/common/accelerator/CommonAcceleratorGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator::Impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Gestion du chargement du runtime accélérateur.
 */
class ARCCORE_COMMON_EXPORT RuntimeLoader
{
 public:

  static int loadRuntime(AcceleratorRuntimeInitialisationInfo& init_info,
                         const String& default_runtime_name,
                         const String& library_path,
                         bool& has_accelerator);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator::Impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
