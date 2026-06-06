// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMeshStats.h                                                (C) 2000-2025 */
/*                                                                           */
/* Interface of a class providing mesh information.                          */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IMESHSTATS_H
#define ARCANE_CORE_IMESHSTATS_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Interface of a class providing mesh information.
 */
class ARCANE_CORE_EXPORT IMeshStats
{
 public:

  //! Releases resources
  virtual ~IMeshStats() = default;

 public:

  //! Creation of a default instance
  static IMeshStats* create(ITraceMng* trace, IMesh* mesh, IParallelMng* pm);

 public:

  //! Prints mesh information
  virtual void dumpStats() = 0;

  //! Prints mesh graph information
  virtual void dumpGraphStats() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
