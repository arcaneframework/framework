// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VariableStatusChangedEventArgs.h                            (C) 2000-2025 */
/*                                                                           */
/* Arguments des évènements générés par IVariableMng.                        */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_VARIABLESTATUSCHANGEDEVENTARGS_H
#define ARCANE_CORE_VARIABLESTATUSCHANGEDEVENTARGS_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/UtilsTypes.h"
#include "arcane/utils/Array.h"

#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Arguments de l'évènement notifiant un changement de l'état
 * d'une variable.
 */
class ARCANE_CORE_EXPORT VariableStatusChangedEventArgs
{
 public:

  enum class Status
  {
    //! Variable ajoutée
    Added,
    //! Variable supprimée
    Removed
  };

 public:

  VariableStatusChangedEventArgs(IVariable* var, Status s)
  : m_variable(var)
  , m_status(s)
  {}
  VariableStatusChangedEventArgs(const VariableStatusChangedEventArgs& rhs) = default;
  VariableStatusChangedEventArgs& operator=(const VariableStatusChangedEventArgs& rhs) = default;

 public:

  //! Variable dont l'état change
  IVariable* variable() const { return m_variable; }
  //! Etat
  Status status() const { return m_status; }

 private:

  IVariable* m_variable = nullptr;
  Status m_status = Status::Added;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
