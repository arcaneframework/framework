// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ParameterList.h                                             (C) 2000-2025 */
/*                                                                           */
/* Liste de paramètres.                                                      */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_PARAMETERLIST_H
#define ARCANE_UTILS_PARAMETERLIST_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/UtilsTypes.h"
#include "arcane/utils/ParameterCaseOption.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Liste de paramètres.
 *
 * Une liste de paramètres est similaire à un ensemble (clé,valeur) mais
 * une clé peut-être éventuellement présente plusieurs fois (un peu à la
 * manière de la classe std::multi_map).
 */
class ARCANE_UTILS_EXPORT ParameterList
{
 private:

  class Impl; //!< Implémentation

 public:

  //! Construit un dictionnaire
  ParameterList();
  //! Construit un dictionnaire
  ParameterList(const ParameterList& rhs);
  ~ParameterList(); //!< Libère les ressources
  
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
   * \brief Récupère la liste des paramètres et leur valeur.
   *
   * Retourne dans \a param_names la liste des noms des paramêtres et
   * dans \a values la valeur associée.
   */
  void fillParameters(StringList& param_names,StringList& values) const;

  /*!
   * \brief Méthode permettant de récupérer un objet de type ParameterCaseOption.
   *
   * Cet objet peut être détruit après utilisation.
   *
   * \param language Le langage dans lequel est écrit le jeu de données.
   * \return Un objet de type ParameterCaseOption.
   */
  ParameterCaseOption getParameterCaseOption(const String& language) const;

 private:

  Impl* m_p; //!< Implémentation
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
