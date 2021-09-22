// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* StandaloneAcceleratorMng.h                                  (C) 2000-2021 */
/*                                                                           */
/* Implémentation autonome (sans IApplication) de 'IAcceleratorMng.h'.       */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_LAUNCHER_STANDALONEACCELERATORMNG_H
#define ARCANE_LAUNCHER_STANDALONEACCELERATORMNG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/launcher/LauncherGlobal.h"

#include "arcane/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Implémentation autonome de 'IAcceleratorMng.h'.
 *
 * Les instances de cette classe sont créées par
 * ArcaneLauncher::createStandaloneAcceleratorMng(). Elles ne sont pas copiables.
 *
 * Cette instance permet d'utiliser les fonctionnalités de %Arcane gérant les
 * accélérateurs sans être obligé de créér une application %Arcane classique.
 */
class ARCANE_LAUNCHER_EXPORT StandaloneAcceleratorMng
{
  friend class ArcaneLauncher;
  class Impl;

 protected:

  StandaloneAcceleratorMng();
  StandaloneAcceleratorMng(const StandaloneAcceleratorMng&) = delete;
  StandaloneAcceleratorMng& operator=(const StandaloneAcceleratorMng&) = delete;

 public:

  ~StandaloneAcceleratorMng();

 public:

  //! Gestionnaire de trace associé.
  ITraceMng* traceMng();

  //! Gestionnaire des accélérateurs associé.
  IAcceleratorMng* acceleratorMng();

 private:

  Impl* m_p = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
