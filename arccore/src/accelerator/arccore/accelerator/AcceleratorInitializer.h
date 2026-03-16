// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AcceleratorInitializer.h                                    (C) 2000-2026 */
/*                                                                           */
/* Initialiseur pour un runtime-accélérator.                                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_ACCELERATOR_ACCELERATORINITIALIZER_H
#define ARCCORE_ACCELERATOR_ACCELERATORINITIALIZER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/accelerator/AcceleratorGlobal.h"

#include <memory>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator
{
class Initializer;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe pour initialiser un runtime pour l'API accélérateur.
 *
 * \warning API expérimentatle en cours de définition.
 *
 * Une seule instance de cette classe peut exister à un moment donné.
 */
class ARCCORE_ACCELERATOR_EXPORT AcceleratorInitializer
{
 public:

  //! Initialise un runtime séquentiel
  AcceleratorInitializer();

  /*!
   * \brief Initialise un runtime.
   *
   * Si \a use_accelerator est vrai, on initialise le runtime accélérateur
   * utilisé pour compiler Arcane. Dans ce cas executionPolicy() retournera
   * ce runtime.
   *
   * Si \a nb_thread est supérieur à 1, alors on initialise aussi le
   * runtime multi-thread.
   */
  explicit AcceleratorInitializer(bool use_accelerator, Int32 nb_thread = 1);

  ~AcceleratorInitializer();

 public:

  AcceleratorInitializer(const AcceleratorInitializer&) = delete;
  AcceleratorInitializer(AcceleratorInitializer&&) = delete;
  AcceleratorInitializer& operator=(const AcceleratorInitializer&) = delete;
  AcceleratorInitializer& operator=(AcceleratorInitializer&&) = delete;

 public:

  //! Politique d'exécution initialisée par défaut
  eExecutionPolicy executionPolicy() const;

  //! Gestionnaire de trace associé
  ITraceMng* traceMng() const;

 private:

  std::unique_ptr<Initializer> m_initializer;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
