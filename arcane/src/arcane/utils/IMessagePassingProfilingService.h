// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMessagePassingProfilingService.h                           (C) 2000-2016 */
/*                                                                           */
/* Interface of a profiling service dedicated to "message passing"           */
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
 * \brief Interface of a profiling service dedicated to "message passing"
 *
 * \note In development
 */
class ARCANE_UTILS_EXPORT IMessagePassingProfilingService
{
 public:

  virtual ~IMessagePassingProfilingService() {}

 public:

  //! Starts profiling
  virtual void startProfiling() =0;

  //! Stops profiling
  virtual void stopProfiling() =0;

  //! Displays information from the profiling
  virtual void printInfos(std::ostream& output) =0;

  //! Gives the name of the service that implements the interface
  virtual String implName() =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
