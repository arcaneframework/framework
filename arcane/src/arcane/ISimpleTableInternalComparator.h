// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ISimpleTableInternalComparator.h                            (C) 2000-2022 */
/*                                                                           */
/* Interface représentant un comparateur de SimpleTableInternal.             */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifndef ARCANE_ISIMPLETABLEINTERNALCOMPARATOR_H
#define ARCANE_ISIMPLETABLEINTERNALCOMPARATOR_H

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/ISimpleTableInternalMng.h"

#include "arcane/ItemTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/**
 * @brief Interface de classe représentant un comparateur de
 * SimpleTableInternal (aka STI).
 * 
 * Le principe est de comparer les valeurs d'un STI avec
 * les valeurs d'un STI de référence, en utilisant un epsilon
 * représentant la marge d'erreur acceptable.
 * 
 * Il existe deux types de manières de configurer ce comparateur :
 * - deux tableaux de String (ligne/colonne),
 * - deux expressions régulières (ligne/colonne).
 * 
 * On peut ajouter des noms de lignes/colonnes dans ces tableaux,
 * préciser si ce sont des lignes/colonnes à inclure dans la
 * comparaison au, au contraire, si ces lignes/colonnes sont
 * à exclure de la comparaison.
 * 
 * Idem pour les expressions régulières, on ajoute une expression
 * régulière lignes/colonnes et on précise si ce sont des 
 * expressions incluant ou excluant des lignes/colonnes.
 * 
 * 
 * Si les deux types de manières sont définis, les tableaux
 * priment sur les expressions régulières : d'abord on regarde
 * la présence du nom de la ligne/colonne dans le tableau correspondant.
 * 
 * Si le nom est présent, on inclut/exclut cette ligne/colonne de la 
 * comparaison.
 * Si le nom est absent mais qu'il y a une expression régulière de
 * défini, on recherche une correspondance dedans. 
 * 
 * Si aucun des types ne sont défini (tableau vide et expression 
 * régulière vide), on inclut toutes les lignes/colonnes dans
 * la comparaison.
 */
class ARCANE_CORE_EXPORT ISimpleTableInternalComparator
{
 public:
  virtual ~ISimpleTableInternalComparator() = default;

 public:
  /**
   * @brief Méthode permettant de comparer les valeurs des deux STI.
   * 
   * @param rank Le processus qui doit comparer ses résultats (-1 pour tous les processus). 
   * @param epsilon La marge d'erreur.
   * @param compare_dimension_too Si l'on doit comparer les dimensions des STI.
   * @return true S'il n'y a pas de différences.
   * @return false S'il y a au moins une différence.
   */
  virtual bool compare(Integer epsilon = 0, bool compare_dimension_too = false) = 0;

  /**
   * @brief Méthode permettant de vider les tableaux de comparaison
   * et les expressions régulières. Ne touche pas aux STI.
   * 
   */
  virtual void clearComparator() = 0;

  /**
   * @brief Méthode permettant d'ajouter une colonne dans la liste des colonnes
   * à comparer.
   * 
   * @param column_name Le nom de la colonne à comparer.
   * @return true Si le nom a bien été ajouté.
   * @return false Sinon.
   */
  virtual bool addColumnForComparing(const String& column_name) = 0;
  /**
   * @brief Méthode permettant d'ajouter une ligne dans la liste des lignes
   * à comparer.
   * 
   * @param row_name Le nom de la ligne à comparer.
   * @return true Si le nom a bien été ajouté.
   * @return false Sinon.
   */
  virtual bool addRowForComparing(const String& row_name) = 0;

  /**
   * @brief Méthode permettant de définir si le tableau de
   * colonnes représente les colonnes à inclure dans la
   * comparaison (false/par défaut) ou représente les colonnes
   * à exclure de la comparaison (true).
   * 
   * @param is_exclusive true si les colonnes doivent être
   *                     exclus.
   */
  virtual void isAnArrayExclusiveColumns(bool is_exclusive) = 0;

  /**
   * @brief Méthode permettant de définir si le tableau de
   * lignes représente les lignes à inclure dans la
   * comparaison (false/par défaut) ou représente les lignes
   * à exclure de la comparaison (true).
   * 
   * @param is_exclusive true si les lignes doivent être
   *                     exclus.
   */
  virtual void isAnArrayExclusiveRows(bool is_exclusive) = 0;

  /**
   * @brief Méthode permettant d'ajouter une expression régulière
   * permettant de déterminer les colonnes à comparer.
   * 
   * @param regex_column L'expression régulière (format ECMAScript).
   */
  virtual void editRegexColumns(const String& regex_column) = 0;
  /**
   * @brief Méthode permettant d'ajouter une expression régulière
   * permettant de déterminer les lignes à comparer.
   * 
   * @param regex_row L'expression régulière (format ECMAScript).
   */
  virtual void editRegexRows(const String& regex_row) = 0;

  /**
   * @brief Méthode permettant de demander à ce que l'expression régulière
   * exclut des colonnes au lieu d'en inclure.
   * 
   * @param is_exclusive Si l'expression régulière est excluante.
   */
  virtual void isARegexExclusiveColumns(bool is_exclusive) = 0;
  /**
   * @brief Méthode permettant de demander à ce que l'expression régulière
   * exclut des lignes au lieu d'en inclure.
   * 
   * @param is_exclusive Si l'expression régulière est excluante.
   */
  virtual void isARegexExclusiveRows(bool is_exclusive) = 0;

  /**
   * @brief Méthode permettant de récupérer le pointeur vers l'objet
   * SimpleTableInternal de référence utilisé.
   * 
   * @return SimpleTableInternal* Le pointeur utilisé. 
   */
  virtual SimpleTableInternal* internalRef() = 0;

  /**
   * @brief Méthode permettant de définir un pointeur vers
   * SimpleTableInternal de référence.
   * 
   * @warning Il est déconseillé d'utiliser cette méthode, sauf si
   * vous savez ce que vous faite. La destruction de l'objet reste
   * à la charge de l'appelant.
   * 
   * @param simple_table_reader_writer Le pointeur vers SimpleTableInternal.
   */
  virtual void setInternalRef(SimpleTableInternal* simple_table_internal) = 0;

  /**
   * @brief Méthode permettant de récupérer le pointeur vers l'objet
   * SimpleTableInternal à comparer utilisé.
   * 
   * @return SimpleTableInternal* Le pointeur utilisé. 
   */
  virtual SimpleTableInternal* internalToCompare() = 0;

  /**
   * @brief Méthode permettant de définir un pointeur vers
   * SimpleTableInternal à comparer.
   * 
   * @warning Il est déconseillé d'utiliser cette méthode, sauf si
   * vous savez ce que vous faite. La destruction de l'objet reste
   * à la charge de l'appelant.
   * 
   * @param simple_table_reader_writer Le pointeur vers SimpleTableInternal.
   */
  virtual void setInternalToCompare(SimpleTableInternal* simple_table_internal) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
