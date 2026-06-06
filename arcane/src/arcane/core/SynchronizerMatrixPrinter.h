// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SynchronizerMatrixPrinter.h                                 (C) 2011-2025 */
/*                                                                           */
/* Prints the synchronization matrix.                                        */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_SYNCHRONIZATIONMATRIXPRINTER_H
#define ARCANE_CORE_SYNCHRONIZATIONMATRIXPRINTER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/Parallel.h"
#include "arcane/core/IVariableSynchronizer.h"

#include <iostream>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Parallel operations on ghost entities.
 */
class ARCANE_CORE_EXPORT SynchronizerMatrixPrinter
{
 public:

  explicit SynchronizerMatrixPrinter(IVariableSynchronizer* synchronizer);
  virtual ~SynchronizerMatrixPrinter() = default; //!< Frees resources.

 public:

  void print(std::ostream& o) const;
  friend std::ostream&
  operator<<(std::ostream& o, const SynchronizerMatrixPrinter& s)
  {
    s.print(o);
    return o;
  }

 private:

  IVariableSynchronizer* m_synchronizer = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
