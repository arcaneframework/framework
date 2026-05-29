// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* StandaloneSubDomain.h                                       (C) 2000-2023 */
/*                                                                           */
/* Standalone implementation of a sub-domain.                                */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_LAUNCHER_STANDALONESUBDOMAIN_H
#define ARCANE_LAUNCHER_STANDALONESUBDOMAIN_H
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
 * \brief Standalone implementation of a sub-domain.
 *
 * The instance of this class must be created by
 * ArcaneLauncher::createStandaloneSubDomain().
 *
 * Only one instance is allowed.
 *
 * This class uses a reference semantics.
 */
class ARCANE_LAUNCHER_EXPORT StandaloneSubDomain
{
  friend class ArcaneLauncher;
  class Impl;

 public:

  //! Uninitialized constructor.
  StandaloneSubDomain();

 public:

  //! Associated trace manager.
  ITraceMng* traceMng();

  //! Sub-domain.
  ISubDomain* subDomain();

 private:

  Ref<Impl> m_p;

 private:

  void _checkIsInitialized();

 private:

  // For ArcaneLauncher.
  void _initUniqueInstance(const String& case_file_name);
  bool _isValid();
  static void _notifyRemoveStandaloneSubDomain();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
