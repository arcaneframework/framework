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
 public:

  //! Créé instance associée a la file \a queue.
  ViewBuildInfo(const RunQueue& queue)
  : m_queue(queue)
  {}
  //! Créé instance associée a la file \a queue.
  ViewBuildInfo(const RunQueue* queue)
  : m_queue(*queue)
  {}
  //! Créé instance associée a la commande \a command.
  ViewBuildInfo(RunCommand& command);

 public:

  const RunQueue& queue() const { return m_queue; }

 private:

  const RunQueue& m_queue;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
