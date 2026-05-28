// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* StandaloneAcceleratorMng.h                                  (C) 2000-2025 */
/*                                                                           */
/* Standalone implementation (without IApplication) of 'IAcceleratorMng.h'.  */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_LAUNCHER_STANDALONEACCELERATORMNG_H
#define ARCANE_LAUNCHER_STANDALONEACCELERATORMNG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/launcher/LauncherGlobal.h"

#include "arcane/utils/Ref.h"
#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Standalone implementation of 'IAcceleratorMng.h'.
 *
 * Instances of this class are created by
 * ArcaneLauncher::createStandaloneAcceleratorMng().
 *
 * This class uses a reference semantics.
 *
 * This instance allows using the %Arcane functionalities managing the
 * accelerators without being forced to create a classic %Arcane application.
 */
class ARCANE_LAUNCHER_EXPORT StandaloneAcceleratorMng
{
  friend class ArcaneLauncher;
  class Impl;

 public:

  StandaloneAcceleratorMng();

 public:

  //! Associated trace manager.
  ITraceMng* traceMng() const;

  //! Associated accelerator manager.
  IAcceleratorMng* acceleratorMng() const;

 private:

  Ref<Impl> m_p;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
