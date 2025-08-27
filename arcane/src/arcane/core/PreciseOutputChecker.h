// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* PreciseOutputChecker.h                                      (C) 2000-2020 */
/*                                                                           */
/* Sorties basées sur un temps (physique ou CPU) ou un nombre d'itération.   */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_PRECISEOUTPUTCHECKER_H
#define ARCANE_PRECISEOUTPUTCHECKER_H
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
 * Service de contrôle sur la réalisation des sorties fichier.
 * Ce dervice est un singleton devant être initialisé au lancement de l'applicaiton.
 */
class PreciseOutputChecker
{
 public:
  /*!
   * Initialisation en fonction du jeu de données utilisateur. 
   **/
  void initializeOutputPhysicalTime(double output_period);
  void initializeOutputPhysicalTime(ICaseFunction* output_period);

  void initializeOutputIteration(Integer output_period);
  void initializeOutputIteration(ICaseFunction* output_period);

  //! Indique s'il faut ou non faire une sortie
  bool checkIfOutput(double old_time,double current_time,Integer current_iteration);

 private:

  //! Méthode interne de comparaison des temps
  bool _compareTime(Real current_time, Real compar_time);

  //! Méthode pour contrôler un output fixé
  bool _checkTime(Real old_time, Real current_time, Real output_period);

  //! Méthode pour vérifier si un temps précédent ne provoque pas d'output
  bool _checkOldTime(Real old_time, Real output_period, Integer curr_number_of_outputs);

  //! Méthode pour contrôler un output par encadrement
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
