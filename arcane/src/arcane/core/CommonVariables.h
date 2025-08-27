// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CommonVariables.h                                           (C) 2000-2025 */
/*                                                                           */
/* Variables communes décrivant un cas.                                      */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_COMMONVARIABLES_H
#define ARCANE_CORE_COMMONVARIABLES_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/String.h"

#include "arcane/core/VariableTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ModuleMaster;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup Variable
 * \brief Variables communes d'un cas.
 */
class ARCANE_CORE_EXPORT CommonVariables
{
 public:

  friend class ModuleMaster;

 public:

  //! Construit les références des variables communes pour le module \a c
  CommonVariables(IModule* c);
  //! Construit les références des variables communes pour le gestionnaire \a variable_mng
  CommonVariables(IVariableMng* variable_mng);
  // TODO: make deprecated
  //! Construit les références des variables communes pour le sous-domaine \a sd
  CommonVariables(ISubDomain* sd);
  virtual ~CommonVariables() {} //!< Libère les ressources.

 public:
	
  //! Numéro de l'itération courante
  Int32 globalIteration() const;
  //! Temps courant
  Real globalTime() const;
  //! Temps courant précédent.
  Real globalOldTime() const;
  //! Temps final de la simulation
  Real globalFinalTime() const;
  //! Delta T courant.
  Real globalDeltaT() const;
  //! Temps CPU utilisé (en seconde)
  Real globalCPUTime() const;
  //! Temps CPU utilisé précédent (en seconde)
  Real globalOldCPUTime() const;
  //! Temps horloge (elapsed) utilisé (en seconde)
  Real globalElapsedTime() const;
  //! Temps horloge (elapsed) utilisé précédent (en seconde)
  Real globalOldElapsedTime() const;

 private:

 public:
	
  VariableScalarInt32 m_global_iteration; //!< Itération courante
  VariableScalarReal m_global_time; //!< Temps actuel
  VariableScalarReal m_global_deltat; //!< Delta T global
  VariableScalarReal m_global_old_time; //!< Temps précédent le temps actuel
  VariableScalarReal m_global_old_deltat; //!< Delta T au temps précédent le temps global
  VariableScalarReal m_global_final_time; //!< Temps final du cas
  VariableScalarReal m_global_old_cpu_time; //!< Temps précédent CPU utilisé (en seconde)
  VariableScalarReal m_global_cpu_time; //!< Temps CPU utilisé (en seconde)
  VariableScalarReal m_global_old_elapsed_time; //!< Temps précédent horloge utilisé (en seconde)
  VariableScalarReal m_global_elapsed_time; //!< Temps horloge utilisé (en seconde)
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

