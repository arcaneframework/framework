// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IDirectExecution.h                                          (C) 2000-2025 */
/*                                                                           */
/* Interface of a direct execution service.                                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IDIRECTEXECUTION_H
#define ARCANE_CORE_IDIRECTEXECUTION_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IMesh;
class IParallelMng;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Interface of a direct execution service.
 *
 * A direct execution service is a service that executes a single operation
 * instead of a time loop, generally to perform internal tests within Arcane.
 *
 * Once the operation is finished, the code stops.
 *
 * This service can be associated with an application, and in this
 * case it does not have a subdomain or a mesh, and the parallelism manager must be positioned
 * before execution.
 */
class ARCANE_CORE_EXPORT IDirectExecution
{
 public:

  virtual ~IDirectExecution() {} //!< Frees resources.

 public:

  virtual void build() = 0;

 public:

  //! Executes the service operation
  virtual void execute() = 0;

  //! True if the service is active
  virtual bool isActive() const = 0;

  /*!
   * \internal.
   * \brief Positions the associated parallelism manager.
   * This method must be called before execute()
   */
  virtual void setParallelMng(IParallelMng* pm) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
