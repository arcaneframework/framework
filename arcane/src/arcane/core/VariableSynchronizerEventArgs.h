// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VariableSynchronizerEventArgs.h                             (C) 2000-2023 */
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

#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

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
  enum class State
  {
    BeginSynchronize,
    EndSynchronize
  };

  VariableSynchronizerEventArgs(VariableCollection vars, IVariableSynchronizer* vs,
                                Real elapsed_time, State state = State::EndSynchronize);
  VariableSynchronizerEventArgs(IVariable* var, IVariableSynchronizer* vs,
                                Real elapsed_time, State state = State::EndSynchronize);
  // Constructor sans temps => debut de synchronisation
  VariableSynchronizerEventArgs(VariableCollection vars, IVariableSynchronizer* vs);
  VariableSynchronizerEventArgs(IVariable* var, IVariableSynchronizer* vs);

  VariableSynchronizerEventArgs(IVariableSynchronizer* vs)
  : m_var_syncer(vs)
  {}

 public:

  VariableSynchronizerEventArgs(const VariableSynchronizerEventArgs& rhs) = default;
  ~VariableSynchronizerEventArgs();
  VariableSynchronizerEventArgs& operator=(const VariableSynchronizerEventArgs& rhs) = default;

 public:

  //! Synchroniseur utilisé.
  IVariableSynchronizer* synchronizer() const { return m_var_syncer; }

  //! Liste des variables synchronisées.
  ConstArrayView<IVariable*> variables() const;
  void setVariables(const VariableCollection& vars);
  void setVariable(IVariable* var);

  //! Temps passé dans la synchronisation.
  Real elapsedTime() const { return m_elapsed_time; }
  void setElapsedTime(Real v) { m_elapsed_time = v; }

  //! Indicateur du moment de l'evenement
  State state() const { return m_state; }
  void setState(State v) { m_state = v; }

 private:

  IVariableSynchronizer* m_var_syncer = nullptr;
  UniqueArray<IVariable*> m_variables;
  Real m_elapsed_time = 0.0;
  State m_state = State::BeginSynchronize;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
