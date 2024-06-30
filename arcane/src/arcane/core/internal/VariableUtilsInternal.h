// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VariableUtilsInternal.h                                     (C) 2000-2024 */
/*                                                                           */
/* Fonctions utilitaires diverses sur les variables internes à Arcane.       */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_INTERNAL_VARIABLEUTILSINTERNAL_H
#define ARCANE_CORE_INTERNAL_VARIABLEUTILSINTERNAL_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"
#include "arcane/core/materials/MaterialsCoreGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_CORE_EXPORT VariableUtilsInternal
{
 public:

  /*!
   * \brief Remplit \a values avec les valeurs de la variable.
   *
   * Seule les variables 1D de type \a DT_Real dont convertibles.
   * \retval false si tout s'est bien passé.
   * \retval true si rien n'a été effectué.
   */
  static bool fillFloat64Array(IVariable* v, ArrayView<double> values);

  //! Retourne l'API internal de IData associé à la variable \a v
  static IDataInternal* getDataInternal(IVariable* v);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::VariableUtils

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
