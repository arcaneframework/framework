// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VariableSynchronizerEventArgs.h                             (C) 2000-2017 */
/*                                                                           */
/* Arguments des évènements générés par IVariableSynchronizer.               */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_VARIABLESYNCHRONIZEREVENTARGS_H
#define ARCANE_VARIABLESYNCHRONIZEREVENTARGS_H
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
class IVariableSynchronizer;
class VariableCollection;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Arguments de l'évènement notifiant une synchronisation de
 * variables.
 */
class ARCANE_CORE_EXPORT VariableSynchronizerEventArgs
{
 public:

	//! Enum pour savoir si on est au debut ou a la fin de la synchronisation
  enum class State {
    BeginSynchronize,
    EndSynchronize
  };

  VariableSynchronizerEventArgs(VariableCollection vars,IVariableSynchronizer* vs,
                                Real elapsed_time, State state = State::EndSynchronize);
  VariableSynchronizerEventArgs(IVariable* var,IVariableSynchronizer* vs,
                                Real elapsed_time, State state = State::EndSynchronize);
  // Ctor sans temps => debut de synchronisation
  VariableSynchronizerEventArgs(VariableCollection vars,IVariableSynchronizer* vs);
  VariableSynchronizerEventArgs(IVariable* var,IVariableSynchronizer* vs);

  VariableSynchronizerEventArgs(const VariableSynchronizerEventArgs& rhs) = default;
  ~VariableSynchronizerEventArgs();
  VariableSynchronizerEventArgs& operator=(const VariableSynchronizerEventArgs& rhs) = default;

 public:
  //! Liste des variables synchronisées.
  ConstArrayView<IVariable*> variables() const;
  //! Synchroniseur utilisé.
  IVariableSynchronizer* synchronizer() const { return m_var_syncer; }
  //! Temps passé dans la synchronisation.
  Real elapsedTime() const { return m_elapsed_time; }
  //! Indicateur du moment de l'evenement
  State state() const { return m_state; }

 private:
  UniqueArray<IVariable*> m_variables;
  IVariable* m_unique_variable;
  IVariableSynchronizer* m_var_syncer;
  Real m_elapsed_time;
  State m_state;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
