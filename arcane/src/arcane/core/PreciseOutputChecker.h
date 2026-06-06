// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* PreciseOutputChecker.h                                      (C) 2000-2025 */
/*                                                                           */
/* Outputs based on time (physical or CPU) or a number of iterations.        */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_PRECISEOUTPUTCHECKER_H
#define ARCANE_CORE_PRECISEOUTPUTCHECKER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/VariableTypes.h"
#include "arcane/core/ICaseFunction.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * Service for controlling the execution of file outputs.
 * This service is a singleton that must be initialized when the application starts.
 */
class PreciseOutputChecker
{
 public:

  /*!
   * Initialization based on the user data set. 
   **/
  void initializeOutputPhysicalTime(double output_period);
  void initializeOutputPhysicalTime(ICaseFunction* output_period);

  void initializeOutputIteration(Integer output_period);
  void initializeOutputIteration(ICaseFunction* output_period);

  //! Indicates whether or not an output should be made
  bool checkIfOutput(double old_time, double current_time, Integer current_iteration);

 private:

  //! Internal method for comparing times
  bool _compareTime(Real current_time, Real compar_time);

  //! Method to control a fixed output
  bool _checkTime(Real old_time, Real current_time, Real output_period);

  //! Method to check if a previous time does not trigger an output
  bool _checkOldTime(Real old_time, Real output_period, Integer curr_number_of_outputs);

  //! Method to control an output by interval
  bool _checkTimeInterval(Real output_period, Real current_time, Real period);

  double m_output_period_physical_time = 0.0;
  ICaseFunction* m_table_values_physical_time = nullptr;
  Integer m_output_period_iteration = -1;
  ICaseFunction* m_table_values_iteration = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
