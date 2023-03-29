// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ICaseMngInternal.h                                          (C) 2000-2023 */
/*                                                                           */
/* Partie interne à Arcane de ICaseMng.                                      */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_INTERNAL_ICASEMNGINTERNAL_H
#define ARCANE_CORE_INTERNAL_ICASEMNGINTERNAL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Partie interne de ICaseMng.
 */
class ARCANE_CORE_EXPORT ICaseMngInternal
{
 public:

  virtual ~ICaseMngInternal() = default;

 public:

  /*!
   * \brief Lit une option du jeu de données.
   */
  virtual void internalReadOneOption(ICaseOptions* opt, bool is_phase1) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
