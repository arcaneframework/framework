// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IVariableUtilities.h                                        (C) 2000-2025 */
/*                                                                           */
/* Interface proposant des fonctions utilitaires sur les variables.          */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IVARIABLEUTILITIES_H
#define ARCANE_CORE_IVARIABLEUTILITIES_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface proposant des fonctions utilitaires sur les variables.
 */
class ARCANE_CORE_EXPORT IVariableUtilities
{
 public:

  virtual ~IVariableUtilities() = default; //!< Libère les ressources.

 public:

  //! Gestionnaire de variables associé
  virtual IVariableMng* variableMng() const = 0;

  /*!
   * \brief Affiche les informations de dépendance sur une variable.
   *
   * Affiche sur le flot \a ostr les informations sur les variables
   * dont dépend \a var. Si \a is_recursive vaut \a true, cette
   * méthode est aussi appelé pour ces variables.
   */
  virtual void dumpDependencies(IVariable* var, std::ostream& ostr, bool is_recursive) = 0;

  /*!
   * \brief Affiche les informations de dépendance de toutes les variables.
   *
   * Affiche sur le flot \a ostr les informations de toutes les
   * variables utilisées.
   */
  virtual void dumpAllVariableDependencies(std::ostream& ostr, bool is_recursive) = 0;

  /*!
   * \brief Filtre les variables communes entre plusieurs rangs.
   *
   * Cette méthode permet de filtrer les variables de \a input_variables
   * qui sont présentes sur tous les rangs de \a pm. Elle retourne
   * la liste triée par ordre alphabétique des variables communes à tous
   * les rangs.
   *
   * Si \a dump_no_common est vrai, affiche (via ITraceMng::info()) la liste
   * des variables qui ne sont pas communes sur tous les rangs.
   */
  virtual VariableCollection filterCommonVariables(IParallelMng* pm,
                                                   VariableCollection input_variables,
                                                   bool dump_not_common) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

