// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IDirectExecutionContext.h                                   (C) 2000-2021 */
/*                                                                           */
/* Implementation of the execution management class.                         */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_LAUNCHER_IDIRECTEXECUTIONCONTEXT_H
#define ARCANE_LAUNCHER_IDIRECTEXECUTIONCONTEXT_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/launcher/LauncherGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class ISubDomain;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Implementation of the execution management class.
 */
class ARCANE_LAUNCHER_EXPORT IDirectExecutionContext
{
 public:
  virtual ~IDirectExecutionContext() = default;
 public:
  /*!
   * \brief Create a sequential sub-domain without a dataset
   */
  virtual ISubDomain* createSequentialSubDomain() =0;

  /*!
   * \brief Create a sequential sub-domain with the dataset file
   * named \a case_file_name.
   */
  virtual ISubDomain* createSequentialSubDomain(const String& case_file_name) =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
