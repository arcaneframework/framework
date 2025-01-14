// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CommandLineArguments.h                                      (C) 2000-2020 */
/*                                                                           */
/* Arguments de la ligne de commande.                                        */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_COMMANDLINEARGUMENTS_H
#define ARCANE_UTILS_COMMANDLINEARGUMENTS_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/ReferenceCounter.h"

#include "arcane/utils/UtilsTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class ParameterList;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Arguments de la ligne de commande.
 *
 * Cette classe utilise une sémantique par référence.
 * Les méthodes commandLineArgc() et commandLineArgv() retournent des
 * pointeurs sur des structures internes à cette classe qui ne sont
 * allouées que tant que l'instance est valide. Ils peuvent être utilisés
 * pour les méthodes classiques du C qui attendent des pointeurs sur les
 * arguments de la ligne de commande (soit l'équivalent du couple (argc,argv)
 * de la fonction main()).
 *
 * Les arguments qui commencent par '-A' sont considérés comme des paramètres
 * de type (clé,valeur) et doivent avoir la forme -A,x=y avec `x` la clé et
 * `y` la valeur. Il est ensuite possible de récupérer la valeur d'un
 * paramètre par l'intermédiaire de sa clé via la méthode getParameter();
 * Si un paramètre est présent plusieurs fois sur la ligne de commande, c'est
 * la dernière valeur qui est conservée.
 */
class ARCANE_UTILS_EXPORT CommandLineArguments
{
  class Impl;

 public:

  //! Créé une instance à partir des arguments (argc,argv)
  CommandLineArguments(int* argc, char*** argv);
  CommandLineArguments();
  explicit CommandLineArguments(const StringList& args);
  CommandLineArguments(const CommandLineArguments& rhs);
  ~CommandLineArguments();
  CommandLineArguments& operator=(const CommandLineArguments& rhs);

 public:

  int* commandLineArgc() const;
  char*** commandLineArgv() const;

  //! Remplit \a args avec arguments de la ligne de commande.
  void fillArgs(StringList& args) const;

  /*!
   * \brief Récupère le paramètre de nom \a param_name.
   *
   * Retourne une chaîne nulle s'il n'y aucun paramètre avec ce nom.
   */
  String getParameter(const String& param_name) const;

  /*!
   * \brief Ajoute un paramètre.
   * \sa ParameterList::addParameterLine()
   */
  void addParameterLine(const String& line);

  /*!
   * \brief Récupère la liste des paramètres et leur valeur.
   *
   * Retourne dans \a param_names la liste des noms des paramètres et
   * dans \a values la valeur associée.
   */
  void fillParameters(StringList& param_names,StringList& values) const;

  //! Liste des paramètres
  const ParameterList& parameters() const;

  //! Liste des paramètres
  ParameterList& parameters();

 private:

  Arccore::ReferenceCounter<Impl> m_p;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
