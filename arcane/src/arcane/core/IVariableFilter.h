// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IVariableFilter.h                                           (C) 2000-2006 */
/*                                                                           */
/* Fonctor d'un filtre applicable sur des variables.                         */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IVARIABLEFILTER_H
#define ARCANE_IVARIABLEFILTER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcaneGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup Variable
 * \brief Fonctor d'un filtre applicable sur des variables.
 *
 * Ce fonctor s'utilise lorsqu'on itère sur des variables. La méthode
 * applyFilter() est appelée pour chaque variable et indique si la
 * variable testée doit ou non être filtrée.
 */
class IVariableFilter
{
 public:

  virtual ~IVariableFilter() {} //!< Libère les ressources

  /*!
   * \brief Applique le filtre sur la variable \a var
   * \retval true si la variable remplit les conditions du filtre
   * \retval false sinon.
   */
  virtual bool applyFilter(IVariable& var) =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

