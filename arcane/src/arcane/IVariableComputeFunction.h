// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IVariableComputeFunction.h                                  (C) 2000-2018 */
/*                                                                           */
/* Interface de la classe fonctor de recalcul d'une variable.                */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IVARIABLECOMPUTEFUNCTION_H
#define ARCANE_IVARIABLECOMPUTEFUNCTION_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcaneGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 \brief Interface de la classe fonctor de recalcul d'une variable.
 */
class IVariableComputeFunction
{
 public:

  virtual ~IVariableComputeFunction(){} //!< Libère les ressources

 public:

  //! Exécute la fonction de calcul
  virtual void execute() =0;

  //! Informations de trace de la définition de la fonction de calcul
  virtual const Arccore::TraceInfo& traceInfo() const =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

