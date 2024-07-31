// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* RunnerInternal.h                                            (C) 2000-2024 */
/*                                                                           */
/* API interne à Arcane de 'Runner'.                                         */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ACCELERATOR_CORE_INTERNAL_RUNNERINTERNAL_H
#define ARCANE_ACCELERATOR_CORE_INTERNAL_RUNNERINTERNAL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/accelerator/core/AcceleratorCoreGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_ACCELERATOR_CORE_EXPORT RunnerInternal
{
  friend ::Arcane::Accelerator::Runner;
  friend ::Arcane::Accelerator::impl::RunnerImpl;

 private:

  explicit RunnerInternal(impl::RunnerImpl* p)
  : m_runner_impl(p)
  {}

 public:

  //! Stoppe toutes les activités de profiling.
  static void stopAllProfiling();

  // Les méthodes suivantes qui gèrent le profiling agissent sur
  // le runtime  (CUDA, ROCM, ...) associé au runner. Par exemple si on
  // a deux runners associés à CUDA, si on appelle startProfiling() pour l'un
  // alors isProfilingActive() sera vrai pour le second runner.

  //! Indique si le profiling est actif pour le runtime associé
  bool isProfilingActive();
  //! Démarre le profiling pour le runtime associé
  void startProfiling();
  //! Stoppe le profiling pour le runtime associé
  void stopProfiling();

  /*!
   * \brief Affiche les informations de profiling.
   *
   * S'il est actif, le profiling est temporairement arrêté et redémaré.
   */
  void printProfilingInfos(std::ostream& o);

 private:

  impl::RunnerImpl* m_runner_impl = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
