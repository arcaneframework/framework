// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMeshMaterialVariableComputeFunction.h                      (C) 2000-2022 */
/*                                                                           */
/* Interface de la classe fonctor de recalcul d'une variable.                */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_MATERIALS_IMESHMATERIALVARIABLECOMPUTEFUNCTION_H
#define ARCANE_CORE_MATERIALS_IMESHMATERIALVARIABLECOMPUTEFUNCTION_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/materials/MaterialsCoreGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 \brief Interface de la classe fonctor de recalcul d'une variable.
 */
class ARCANE_CORE_EXPORT IMeshMaterialVariableComputeFunction
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

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

