// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SynchronizerMatrixPrinter.h                              (C) 2011-2011 */
/*                                                                           */
/* Affiche la matrix de synchronization.                                     */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_PARALLEL_SYNCHRONIZATIONMATRIXPRINTER_H
#define ARCANE_PARALLEL_SYNCHRONIZATIONMATRIXPRINTER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/Parallel.h"
#include "arcane/ArcaneTypes.h"
#include "arcane/IVariableSynchronizer.h"
#include <iostream>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Opérations parallèle sur les entités fantômes.
 */
class ARCANE_CORE_EXPORT SynchronizerMatrixPrinter
{
public:

  SynchronizerMatrixPrinter(IVariableSynchronizer* synchronizer);
  virtual ~SynchronizerMatrixPrinter() {} //!< Libère les ressources.

public:
  void print(std::ostream & o) const;

private:
  IVariableSynchronizer * m_synchronizer;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

std::ostream & 
operator<<(std::ostream & o, const SynchronizerMatrixPrinter & s)
{
  s.print(o);
  return o;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

