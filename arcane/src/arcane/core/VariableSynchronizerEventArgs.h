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
 *
 * Les instances de cette classe peuvent être utilisées plusieurs fois.
 * Il faut appeler initialize() pour initialiser ou réinitialiser l'instance
 * avec les valeurs par défaut.
 */
class ARCANE_CORE_EXPORT VariableSynchronizerEventArgs
{
 public:

  //! Enum pour savoir si on est au début ou à la fin de la synchronisation
  enum class State
  {
    BeginSynchronize,
    EndSynchronize
  };

  //! Comparaison des valeurs des entités fantômes avant/après une synchronisation
  enum class CompareStatus
  {
    //! Pas de comparaison ou résultat inconnue
    Unknown,
    //! Même valeurs avant et après la synchronisation
    Same,
    //! Valeurs différentes avant et après la synchronisation
    Different
  };

 public:

  ARCANE_DEPRECATED_REASON("Y2023: Use VariableSynchronizerEventArgs(IVariableSynchronizer* vs) and call initialize() instead")
  VariableSynchronizerEventArgs(VariableCollection vars, IVariableSynchronizer* vs,
                                Real elapsed_time, State state = State::EndSynchronize);
  ARCANE_DEPRECATED_REASON("Y2023: Use VariableSynchronizerEventArgs(IVariableSynchronizer* vs) and call initialize() instead")
  VariableSynchronizerEventArgs(IVariable* var, IVariableSynchronizer* vs,
                                Real elapsed_time, State state = State::EndSynchronize);

  ARCANE_DEPRECATED_REASON("Y2023: Use VariableSynchronizerEventArgs(IVariableSynchronizer* vs) and call initialize() instead")
  VariableSynchronizerEventArgs(VariableCollection vars, IVariableSynchronizer* vs);

  ARCANE_DEPRECATED_REASON("Y2023: Use VariableSynchronizerEventArgs(IVariableSynchronizer* vs) and call initialize() instead")
  VariableSynchronizerEventArgs(IVariable* var, IVariableSynchronizer* vs);

  VariableSynchronizerEventArgs(IVariableSynchronizer* vs)
  : m_var_syncer(vs)
  {}

 public:

  void initialize(const VariableCollection& vars);
  void initialize(IVariable* var);

  //! Synchroniseur associé.
  IVariableSynchronizer* synchronizer() const { return m_var_syncer; }

  //! Liste des variables synchronisées.
  ConstArrayView<IVariable*> variables() const;

  /*!
   * \brief Liste de l'état de comparaison.
   *
   * La valeur du i-ème élément de compareStatus() indique l'état
   * de comparaison pour la i-ème variable de variables().
   *
   * Cette liste n'est valide que pour les évènements de fin de synchronisation
   * (state()==State::EndSynchronize).
   */
  ConstArrayView<CompareStatus> compareStatusList() const { return m_compare_status_list; }

  //! Positionne l'état de comparaison de la i-ème variable.
  void setCompareStatus(Int32 i, CompareStatus v) { m_compare_status_list[i] = v; }

  //! Temps passé dans la synchronisation.
  Real elapsedTime() const { return m_elapsed_time; }
  void setElapsedTime(Real v) { m_elapsed_time = v; }

  //! Indicateur du moment de l'évènement
  State state() const { return m_state; }
  void setState(State v) { m_state = v; }

 private:

  IVariableSynchronizer* m_var_syncer = nullptr;
  UniqueArray<IVariable*> m_variables;
  UniqueArray<CompareStatus> m_compare_status_list;
  Real m_elapsed_time = 0.0;
  State m_state = State::BeginSynchronize;

 private:

  void _reset();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
