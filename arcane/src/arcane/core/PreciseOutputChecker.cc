// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* PreciseOutputChecker.cc                                     (C) 2000-2020 */
/*                                                                           */
/* Outputs based on time (physical or CPU) or a number of iterations.        */
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

  // Update of the output period in the case of a marching table
  if (m_table_values_physical_time != nullptr) {
    m_table_values_physical_time->value(current_time, m_output_period_physical_time);
  }
  // Check on exact time
  if (!output_requested) {
    output_requested = _checkTime(old_time, current_time, m_output_period_physical_time);
  }
  // Check on time interval
  if (!output_requested) {
    output_requested = _checkTimeInterval(old_time, current_time, m_output_period_physical_time);
  }

  // Check on iterations
  if (!output_requested) {
    // Update of the output frequency for marching tables
    if (m_table_values_iteration != nullptr) {
      m_table_values_iteration->value(current_iteration, m_output_period_iteration);
    }
    if (m_output_period_iteration > 0) {
      // Exact check
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
      // We must check that due to truncation errors, we do not end up in a
      // case where the output was already performed at an exact output in
      // the previous iteration.
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
    // Then we reconstruct the output time:
    Real prev_output_time_round = current_number_of_outputs * output_period;
    // And we check if the previous time step leads to an output
    bool old_time_output_requested = _compareTime(old_time, prev_output_time_round);
    // If yes, we cancel the output
    if (old_time_output_requested) {
      output_requested = false;
    }
  }

  // We check the upper bound for overlaps
  if (output_requested) {
    Integer prev_number_of_outputs_ceil = (int)ceil(old_time / output_period);
    if (prev_number_of_outputs_ceil == current_number_of_outputs) {
      // Then we reconstruct the output time:
      Real prev_output_time_ceil = prev_number_of_outputs_ceil * output_period;
      // And we check if the previous time step leads to an output
      bool old_time_output_requested = _compareTime(old_time, prev_output_time_ceil);
      // If yes, we cancel the output
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
    // Reconstruction of the time interval in which the code is located.
    // To know the number of outputs already performed in a fixed period,
    // we take the floor and ceiling conversion.
    Integer number_of_outputs_round = (int)floor(current_time / output_period);
    Integer number_of_outputs_ceil = (int)ceil(current_time / output_period);
    // Then we reconstruct the output time:
    Real next_output_time_round = number_of_outputs_round * output_period;
    Real next_output_time_ceil = number_of_outputs_ceil * output_period;

    output_requested = _compareTime(current_time, next_output_time_round);

    // The following two blocks constitute a FIX:
    // we must check that the previous time does not lead to an output time
    // to cancel the output request if so.
    // Otherwise, rounding errors in the exact comparison
    // can lead to considering several times
    // as part of the requested output time (e.g., if t=1e-7 and dt=1e-15,
    // then the relative comparison of comparTime will give several valid times,
    // leading to...). The comparison is made on the upper and
    // lower bounds to remove uncertainty in the case where dt=output time.
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
    // If the relative difference between the times is < 1e-7 (i.e., the precision
    // related to the division), then we consider
    // that the requested time corresponds to the output time.
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
