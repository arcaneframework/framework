// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMeshMaterialVariableComputeFunction.h                      (C) 2000-2018 */
/*                                                                           */
/* Interface de la classe fonctor de recalcul d'une variable.                */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MATERIALS_IMESHMATERIALVARIABLECOMPUTEFUNCTION_H
#define ARCANE_MATERIALS_IMESHMATERIALVARIABLECOMPUTEFUNCTION_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcaneGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MATERIALS_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 \brief Interface de la classe fonctor de recalcul d'une variable.
 */
class ARCANE_MATERIALS_EXPORT IMeshMaterialVariableComputeFunction
{
 public:

  virtual ~IMeshMaterialVariableComputeFunction(){} //!< Libère les ressources

 public:

  //! Exécute la fonction de calcul
  virtual void execute(IMeshMaterial* mat) =0;

  //! Informations de trace de la définition de la fonction de calcul
  virtual const TraceInfo& traceInfo() const =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MATERIALS_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

