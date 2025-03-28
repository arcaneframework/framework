// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* StandaloneAcceleratorMng.h                                  (C) 2000-2025 */
/*                                                                           */
/* Implémentation autonome (sans IApplication) de 'IAcceleratorMng.h'.       */
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
 * \brief Implémentation autonome de 'IAcceleratorMng.h'.
 *
 * Les instances de cette classe sont créées par
 * ArcaneLauncher::createStandaloneAcceleratorMng().
 *
 * Cette classe utilise une sémantique par référence.
 *
 * Cette instance permet d'utiliser les fonctionnalités de %Arcane gérant les
 * accélérateurs sans être obligé de créér une application %Arcane classique.
 */
class ARCANE_LAUNCHER_EXPORT StandaloneAcceleratorMng
{
  friend class ArcaneLauncher;
  class Impl;

 public:

  StandaloneAcceleratorMng();

 public:

  //! Gestionnaire de trace associé.
  ITraceMng* traceMng() const;

  //! Gestionnaire des accélérateurs associé.
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
