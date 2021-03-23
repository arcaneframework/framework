// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMessagePassingProfilingService.h                           (C) 2000-2016 */
/*                                                                           */
/* Interface d'un service de profiling dédié au "message passing"            */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IMESSAGEPASSINGPROFILINGSERVICE_H
#define ARCANE_IMESSAGEPASSINGPROFILINGSERVICE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcaneGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Interface d'un service de profiling dedie au "message passing"
 *
 * \note En cours de developpement
 */
class ARCANE_UTILS_EXPORT IMessagePassingProfilingService
{
 public:

  virtual ~IMessagePassingProfilingService() {}

 public:

  //! Démarre un profiling
  virtual void startProfiling() =0;

  //! Stoppe le profiling
  virtual void stopProfiling() =0;

  //! Affiche les informations issues du profiling
  virtual void printInfos(std::ostream& output) =0;

  //! Donne le nom du service qui implemente l'interface
  virtual String implName() =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
