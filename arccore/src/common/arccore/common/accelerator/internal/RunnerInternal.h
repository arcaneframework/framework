// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* RunnerInternal.h                                            (C) 2000-2025 */
/*                                                                           */
/* API interne à Arcane de 'Runner'.                                         */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_COMMON_ACCELERATOR_INTERNAL_RUNNERINTERNAL_H
#define ARCCORE_COMMON_ACCELERATOR_INTERNAL_RUNNERINTERNAL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/common/accelerator/CommonAcceleratorGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCCORE_COMMON_EXPORT RunnerInternal
{
  friend ::Arcane::Accelerator::Runner;
  friend ::Arcane::Accelerator::Impl::RunnerImpl;

 private:

  explicit RunnerInternal(Impl::RunnerImpl* p)
  : m_runner_impl(p)
  {}

 public:

  //! Stoppe toutes les activités de profiling.
  static void stopAllProfiling();

  /*!
   * \brief Finalise l'exécution.
   *
   * Cela sert à afficher certaines statistiques et libérer les ressources.
   */
  static void finalize(ITraceMng* tm);

 public:

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

  Impl::RunnerImpl* m_runner_impl = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Accelerator::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
