// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IVariableComputeFunction.h                                  (C) 2000-2025 */
/*                                                                           */
/* Interface de la classe functor de re-calcul d'une variable.               */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IVARIABLECOMPUTEFUNCTION_H
#define ARCANE_CORE_IVARIABLECOMPUTEFUNCTION_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcaneGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 \brief Interface de la classe functor de re-calcul d'une variable.
 */
class IVariableComputeFunction
{
 public:

  virtual ~IVariableComputeFunction() = default; //!< Libère les ressources

 public:

  //! Exécute la fonction de calcul
  virtual void execute() = 0;

  //! Informations de trace de la définition de la fonction de calcul
  virtual const TraceInfo& traceInfo() const = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

