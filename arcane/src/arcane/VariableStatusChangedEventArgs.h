// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VariableStatusChangedEventArgs.h                            (C) 2000-2017 */
/*                                                                           */
/* Arguments des évènements générés par IVariableMng.                        */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_VARIABLESTATUSCHANGEDEVENTARGS_H
#define ARCANE_VARIABLESTATUSCHANGEDEVENTARGS_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/UtilsTypes.h"
#include "arcane/utils/Array.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IVariable;

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
  VariableStatusChangedEventArgs(IVariable* var,Status s)
  : m_variable(var), m_status(s){}
  VariableStatusChangedEventArgs(const VariableStatusChangedEventArgs& rhs) = default;
  ~VariableStatusChangedEventArgs(){}
  VariableStatusChangedEventArgs& operator=(const VariableStatusChangedEventArgs& rhs) = default;
 public:
  //! Variable dont l'état change
  IVariable* variable() const { return m_variable; }
  //! Etat
  Status status() const { return m_status; }
 private:
  IVariable* m_variable;
  Status m_status;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
