// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ViewBuildInfo.h                                             (C) 2000-2024 */
/*                                                                           */
/* Informations pour construire une vue pour les données sur accélérateur.   */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ACCELERATOR_CORE_VIEWBUILDINFO_H
#define ARCANE_ACCELERATOR_CORE_VIEWBUILDINFO_H
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
 * \brief Informations pour construire une vue pour les données sur accélérateur.
 *
 * Les instances de cette classes sont temporaires et ne doivent pas être
 * conservées au dela de la durée de vie de la RunCommand ou RunQueue utilisées
 * pour leur création.
 */
class ARCANE_ACCELERATOR_CORE_EXPORT ViewBuildInfo
{
  friend class NumArrayViewBase;
  friend class VariableViewBase;

 public:

  //! Créé instance associée a la file \a queue.
  explicit(false) ViewBuildInfo(const RunQueue& queue);
  //! Créé instance associée a la file \a queue.
  explicit(false) ViewBuildInfo(const RunQueue* queue);
  //! Créé instance associée a la commande \a command.
  explicit(false) ViewBuildInfo(RunCommand& command);

 private:

  impl::RunQueueImpl* _internalQueue() const { return m_queue_impl; }

 private:

  impl::RunQueueImpl* m_queue_impl = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
