// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ViewBuildInfo.h                                             (C) 2000-2025 */
/*                                                                           */
/* Informations pour construire une vue pour les données sur accélérateur.   */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_COMMON_ACCELERATOR_VIEWBUILDINFO_H
#define ARCCORE_COMMON_ACCELERATOR_VIEWBUILDINFO_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/common/accelerator/CommonAcceleratorGlobal.h"

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
class ARCCORE_COMMON_EXPORT ViewBuildInfo
{
  friend class NumArrayViewBase;
  friend class VariableViewBase;

 public:

  // NOTE: les constructeurs suivant doivent être implicites

  //! Créé instance associée a la file \a queue.
  ViewBuildInfo(const RunQueue& queue);
  //! Créé instance associée a la file \a queue.
  ViewBuildInfo(const RunQueue* queue);
  //! Créé instance associée a la commande \a command.
  ViewBuildInfo(RunCommand& command);

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
