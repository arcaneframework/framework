// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ParameterListWithCaseOption.h                               (C) 2000-2025 */
/*                                                                           */
/* Liste de paramètres avec support pour les options du jeu de données.      */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_INTERNAL_PARAMETERLISTWITHCASEOPTION_H
#define ARCANE_UTILS_INTERNAL_PARAMETERLISTWITHCASEOPTION_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/UtilsTypes.h"
#include "arcane/utils/internal/ParameterCaseOption.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class ParameterList;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Liste de paramètres avec informations pour surcharger les options
 * du jeu de données.
 */
class ARCANE_UTILS_EXPORT ParameterListWithCaseOption
{
  class Impl;

 public:

  //! Construit un dictionnaire
  ParameterListWithCaseOption();
  //! Construit un dictionnaire
  ParameterListWithCaseOption(const ParameterListWithCaseOption& rhs);
  ~ParameterListWithCaseOption(); //!< Libère les ressources

 public:

  /*!
   * \brief Récupère le paramètre de nom \a param_name.
   *
   * Retourne une chaîne nulle s'il n'y aucun paramètre avec ce nom.
   *
   * Si le paramètre est présent plusieurs fois, seule la dernière
   * valeur est retournée.
   */
  String getParameterOrNull(const String& param_name) const;

  /*!
   * \brief Analyse la ligne \a line.
   *
   * La ligne doit avoir une des formes suivantes, avec \a A le
   * paramètre et \a B la valeur:
   *
   * 1. A=B,
   * 2. A:=B
   * 3. A+=B,
   * 4. A-=B
   *
   * Dans le cas (1) ou (3), la valeur de l'argument est ajoutée aux
   * occurences déjà présentes. Dans le cas (2), la valeur de
   * l'argument remplace toutes les occurences déjà présentes. Dans
   * le cas (4), l'occurence ayant la valeur \a B est supprimée si elle
   * était présente et rien ne se produit si elle était absente.
   *
   * \retval false si un paramètre a pu être analysé
   * \retval true sinon.
   */
  bool addParameterLine(const String& line);

  /*!
   * \brief Méthode permettant de récupérer un objet de type ParameterCaseOption.
   *
   * Cet objet peut être détruit après utilisation.
   *
   * \param language Le langage dans lequel est écrit le jeu de données.
   * \return Un objet de type ParameterCaseOption.
   */
  ParameterCaseOption getParameterCaseOption(const String& language) const;

  //! Ajoute les paramètres de \a parameters aux paramètres de l'instance
  void addParameters(const ParameterList& parameters);

 private:

  Impl* m_p = nullptr; //!< Implémentation
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
