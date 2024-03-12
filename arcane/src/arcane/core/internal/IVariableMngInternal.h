// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IVariableMngInternal.h                                      (C) 2000-2024 */
/*                                                                           */
/* Partie interne à Arcane de IVariableMng.                                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_INTERNAL_IVARIABLEMNG_H
#define ARCANE_CORE_INTERNAL_IVARIABLEMNG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface du gestionnaire de variables.
 *
 * Ce gestionnaire contient la liste des variables déclarées dans le
 * sous-domaine associé \a subDomain(). Il maintient la liste des variables
 * et permet de les lire ou de les écrire.
 */
class ARCANE_CORE_EXPORT IVariableMngInternal
{
 public:

  virtual ~IVariableMngInternal() = default; //!< Libère les ressources.

 public:

  /*!
   * \brief Construit les membres de l'instance.
   *
   * L'instance n'est pas utilisable tant que cette méthode n'a pas été
   * appelée. Cette méthode doit être appelée avant initialize().
   * \warning Cette méthode ne doit être appelée qu'une seule fois.
   */
  virtual void build() = 0;

  /*!
   * \brief Initialise l'instance.
   * L'instance n'est pas utilisable tant que cette méthode n'a pas été
   * appelée.
   * \warning Cette méthode ne doit être appelée qu'une seule fois.
   */
  virtual void initialize() = 0;

  //! Supprime et détruit les variables gérées par ce gestionnaire
  virtual void removeAllVariables() = 0;

  //! Détache les variables associées au maillage \a mesh.
  virtual void detachMeshVariables(IMesh* mesh) = 0;

 public:

  /*!
   * \brief Ajoute une référence à une variable.
   *
   * Ajoute la référence \a var au gestionnaire.
   *
   * \pre var != 0
   * \pre var ne doit pas déjà être référencée.
   * \return l'implémentation associée à \a var.
   */
  virtual void addVariableRef(VariableRef* var) = 0;

  /*!
   * \brief Supprime une référence à une variable.
   *
   * Supprime la référence \a var du gestionnaire.
   *
   * Si \a var n'est pas référencée par le gestionnaire, rien n'est effectué.
   * \pre var != 0
   */
  virtual void removeVariableRef(VariableRef* var) = 0;

  /*!
   * \brief Ajoute une variable.
   *
   * Ajoute la variable \a var.
   *
   * La validité de la variable n'est pas effectuée (void checkVariable()).
   *
   * \pre var != 0
   * \pre var ne doit pas déjà être référencée.
   * \return l'implémentation associée à \a var.
   */
  virtual void addVariable(IVariable* var) = 0;

  /*!
   * \brief Supprime une variable.
   *
   * Supprime la variable \a var.
   *
   * Après appel à cette méthode, la variable ne doit plus être utilisée.
   *
   * \pre var != 0
   * \pre var doit avoir une seule référence.
   */
  virtual void removeVariable(IVariable* var) = 0;

  /*!
   * \brief Initialise les variables.
   *
   * Parcours la liste des variables et les initialisent.
   * Seules les variables d'un module utilisé sont initialisées.
   *
   * \param is_continue \a true vrai si on est en reprise.
   */
  virtual void initializeVariables(bool is_continue) = 0;

 public:

  //! Fonction interne temporaire pour récupérer le sous-domaine.
  virtual ISubDomain* internalSubDomain() const = 0;

  //! Gestionnaire pour les accélérateurs
  virtual IAcceleratorMng* acceleratorMng() const = 0;

  //! Positionne le gestionnaire des accélérateurs
  virtual void setAcceleratorMng(Ref<IAcceleratorMng> v) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
