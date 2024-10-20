// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ProfileRegion.h                                             (C) 2000-2024 */
/*                                                                           */
/* Région pour le profiling.                                                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ACCELERATOR_CORE_INTERNAL_PROFILEREGION_H
#define ARCANE_ACCELERATOR_CORE_INTERNAL_PROFILEREGION_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/accelerator/core/AcceleratorCoreGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Région pour le profiling.
 *
 * Cette classe permet d'associer des informations de profiling à tous les
 * noyaux de calcul exécutés entre le constructeur et le destructeur d'une
 * instance de cette classe.
 */
class ARCANE_ACCELERATOR_CORE_EXPORT ProfileRegion
{
 public:

  ProfileRegion(const RunQueue& queue, const String& name);
  ~ProfileRegion();

 private:

  impl::IRunnerRuntime* m_runtime = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
