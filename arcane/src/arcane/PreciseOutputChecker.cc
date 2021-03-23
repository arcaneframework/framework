// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* PreciseOutputChecker.cc                                     (C) 2000-2020 */
/*                                                                           */
/* Sorties basées sur un temps (physique ou CPU) ou un nombre d'itération.   */
/*---------------------------------------------------------------------------*/

#include "arcane/PreciseOutputChecker.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void PreciseOutputChecker::
initializeOutputPhysicalTime(Real output_period)
{
  m_output_period_physical_time = output_period;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void PreciseOutputChecker::
initializeOutputPhysicalTime(ICaseFunction* output_table)
{
  m_table_values_physical_time = output_table;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void PreciseOutputChecker::
initializeOutputIteration(Integer output_period)
{
  m_output_period_iteration = output_period;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void PreciseOutputChecker::
initializeOutputIteration(ICaseFunction* output_table)
{
  m_table_values_iteration = output_table;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool PreciseOutputChecker::
checkIfOutput(Real old_time, Real current_time, Integer current_iteration)
{
  bool output_requested = false;

  //Mise à jour de la période de sortie dans le cas d'une table de marche
  if (m_table_values_physical_time != nullptr) {
    m_table_values_physical_time->value(current_time, m_output_period_physical_time);
  }
  //Contrôle sur le temps exact
  if (!output_requested) {
    output_requested = _checkTime(old_time, current_time, m_output_period_physical_time);
  }
  //Contrôle sur l'encadrement du temps
  if (!output_requested) {
    output_requested = _checkTimeInterval(old_time, current_time, m_output_period_physical_time);
  }

  //Contrôle sur les itérations
  if (!output_requested) {
    //Mise a jour de la fréquence de sortie pour les tables de marche
    if (m_table_values_iteration != nullptr) {
      m_table_values_iteration->value(current_iteration, m_output_period_iteration);
    }
    if (m_output_period_iteration > 0) {
      //Contrôle exact
      if (current_iteration % m_output_period_iteration == 0) {
        output_requested = true;
      }
    }
  }
  return output_requested;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool PreciseOutputChecker::
_checkTimeInterval(Real old_time, Real current_time, Real output_period)
{
  bool output_requested = false;
  if (output_period > 0) {
    Integer number_of_previous_outputs = (int)floor(old_time / output_period);
    Integer number_of_current_outputs = (int)floor(current_time / output_period);

    if (number_of_previous_outputs != number_of_current_outputs) {
      // Il faut vérifier qu'à cause des erreurs de troncature, on ne se
      // retrouve pas sur un cas où la sortie aurrait déjà été réalisée sur
      // une sortie exacte à l'itération précédente.
      Integer number_of_previous_outputs_ceil = (int)ceil(old_time / output_period);
      Real old_time_reconstruct = number_of_previous_outputs_ceil * output_period;
      if ((old_time_reconstruct - old_time) > 1e-7 * old_time) {
        output_requested = true;
      }
    }
  }
  return output_requested;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool PreciseOutputChecker::
_checkOldTime(Real old_time, Real output_period, Integer current_number_of_outputs)
{
  bool output_requested = true;
  Integer prev_number_of_outputs_round = (int)floor(old_time / output_period);
  if (prev_number_of_outputs_round == current_number_of_outputs) {
    // Puis on reconstruit le temps de sortie :
    Real prev_output_time_round = current_number_of_outputs * output_period;
    // Et on contrôle si le pas de temps précédent conduit à un output
    bool old_time_output_requested = _compareTime(old_time, prev_output_time_round);
    // Si oui, on annule l'output
    if (old_time_output_requested) {
      output_requested = false;
    }
  }

  // On vérifie la borne sup pour les recouvrements
  if (output_requested) {
    Integer prev_number_of_outputs_ceil = (int)ceil(old_time / output_period);
    if (prev_number_of_outputs_ceil == current_number_of_outputs) {
      // Puis on reconstruit le temps de sortie :
      Real prev_output_time_ceil = prev_number_of_outputs_ceil * output_period;
      // Et on contrôle si le pas de temps précédent conduit à un output
      bool old_time_output_requested = _compareTime(old_time, prev_output_time_ceil);
      // Si oui, on annule l'output
      if (old_time_output_requested) {
        output_requested = false;
      }
    }
  }
  return output_requested;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool PreciseOutputChecker::
_checkTime(Real old_time, Real current_time, Real output_period)
{
  bool output_requested = false;

  if (output_period > 0) {
    // Reconstruction de l'intervalle de temps dans lequel se trouve le code.
    // Pour connaitre le nombre d'output déjà effectués en période fixe,
    // on prend la conversion entière inf et sup.
    Integer number_of_outputs_round = (int)floor(current_time / output_period);
    Integer number_of_outputs_ceil = (int)ceil(current_time / output_period);
    // Puis on reconstruit le temps de sortie:
    Real next_output_time_round = number_of_outputs_round * output_period;
    Real next_output_time_ceil = number_of_outputs_ceil * output_period;

    output_requested = _compareTime(current_time, next_output_time_round);

    // Les deux blocs suivants constituent un FIX :
    // il faut vérifier que le temps précédent ne conduit pas à un temps de sortie
    // pour annuler la demande de sortie si tel est le cas.
    // Dans le cas contraire, les erreurs d'arrondis sur la comparaison exacte
    // peuvent conduire à considérer plusieurs temps
    // comme faisant parti du temps de sortie demandé (ex : si t=1e-7 et dt=1e-15,
    // alors la comparaison relative de comparTime donnera plusieurs temps
    // valides, conduisant à ...). La comparaison est faite sur les bornes sup et
    // inf pour lever l'incertitude dans le cas ou dt=temps de sortie.
    if (output_requested) {
      output_requested = _checkOldTime(old_time, output_period, number_of_outputs_round);
    }

    if (!output_requested) {
      output_requested = _compareTime(current_time, next_output_time_ceil);

      if (output_requested) {
        output_requested = _checkOldTime(old_time, output_period, number_of_outputs_ceil);
      }
    }
  }

  return output_requested;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool PreciseOutputChecker::
_compareTime(Real current_time, Real compar_time)
{
  bool output_requested = false;
  if (compar_time > 0.0) {
    // Si l'écart relatif entre les temps est < à 1e-7 (soit la précision
    // liée à la division), alors on considère
    // que le temps demandé correspond à celui de l'output.
    if (((math::abs(current_time - compar_time))) < 1.0e-7 * compar_time) {
      output_requested = true;
    }
  }
  return output_requested;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
