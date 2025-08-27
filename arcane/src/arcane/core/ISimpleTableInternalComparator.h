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

#include "arcane/core/ISimpleTableInternalMng.h"

#include "arcane/core/ItemTypes.h"

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
   * @param compare_dimension_too Si l'on doit comparer les dimensions des STI.
   * @return true S'il n'y a pas de différences.
   * @return false S'il y a au moins une différence.
   */
  virtual bool compare(bool compare_dimension_too = false) = 0;

  /**
   * @brief Méthode permettant de comparer uniquement un élement.
   * Les deux SimpleTableInternal sont représentés par des Ref,
   * donc toujours à jour.
   * Cette méthode peut être utilisé pendant le calcul, permettant
   * de comparer les valeurs au fur et à mesure de l'avancement du
   * calcul, au lieu de faire une comparaison final à la fin (il est
   * tout de même possible de faire les deux).
   * 
   * @param column_name Le nom de la colonne où se trouve l'élément.
   * @param row_name Le nom de la ligne où se trouve l'élément.
   * @return true Si les deux valeurs sont égales.
   * @return false Si les deux valeurs sont différentes.
   */
  virtual bool compareElem(const String& column_name, const String& row_name) = 0;

  /**
   * @brief Méthode permettant de comparer une valeur avec
   * une valeur du tableau de référence.
   * Cette méthode n'utilise pas l'internal 'toCompare'.
   * 
   * @param elem La valeur à comparer.
   * @param column_name Le nom de la colonne où se trouve l'élément de référence.
   * @param row_name Le nom de la ligne où se trouve l'élément de référence.
   * @return true Si les deux valeurs sont égales.
   * @return false Si les deux valeurs sont différentes.
   */
  virtual bool compareElem(Real elem, const String& column_name, const String& row_name) = 0;

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
   * @brief Méthode permettant de définir un epsilon pour une colonne donnée.
   * Cet epsilon doit être positif pour être pris en compte.
   * S'il y a confit avec un epsilon de ligne (défini avec addEpsilonRow()),
   * c'est l'epsilon le plus grand qui est pris en compte.
   * @note Si un epsilon a déjà été défini sur cette colonne, alors l'ancien
   * epsilon sera remplacé.
   * 
   * @param column_name Le nom de la colonne où l'epsilon sera pris en compte.
   * @param epsilon La marge d'erreur epsilon.
   * @return true Si l'epsilon a bien pu être défini.
   * @return false Si l'epsilon n'a pas pu être défini.
   */
  virtual bool addEpsilonColumn(const String& column_name, Real epsilon) = 0;

  /**
   * @brief Méthode permettant de définir un epsilon pour une ligne donnée.
   * Cet epsilon doit être positif pour être pris en compte.
   * S'il y a confit avec un epsilon de colonne (défini avec addEpsilonColumn()),
   * c'est l'epsilon le plus grand qui est pris en compte.
   * @note Si un epsilon a déjà été défini sur cette ligne, alors l'ancien
   * epsilon sera remplacé.
   * 
   * @param column_name Le nom de la ligne où l'epsilon sera pris en compte.
   * @param epsilon La marge d'erreur epsilon.
   * @return true Si l'epsilon a bien pu être défini.
   * @return false Si l'epsilon n'a pas pu être défini.
   */
  virtual bool addEpsilonRow(const String& row_name, Real epsilon) = 0;


  /**
   * @brief Méthode permettant de récupérer une référence vers l'objet
   * SimpleTableInternal "de référence" utilisé.
   * 
   * @return Ref<SimpleTableInternal> Une copie de la référence. 
   */
  virtual Ref<SimpleTableInternal> internalRef() = 0;

  /**
   * @brief Méthode permettant de définir une référence vers un
   * SimpleTableInternal "de référence".
   * 
   * @param simple_table_internal La référence vers un SimpleTableInternal.
   */
  virtual void setInternalRef(const Ref<SimpleTableInternal>& simple_table_internal) = 0;

  /**
   * @brief Méthode permettant de récupérer une référence vers l'objet
   * SimpleTableInternal "à comparer" utilisé.
   * 
   * @return Ref<SimpleTableInternal> Une copie de la référence. 
   */
  virtual Ref<SimpleTableInternal> internalToCompare() = 0;

  /**
   * @brief Méthode permettant de définir une référence vers
   * SimpleTableInternal "à comparer".
   * 
   * @param simple_table_internal La référence vers un SimpleTableInternal.
   */
  virtual void setInternalToCompare(const Ref<SimpleTableInternal>& simple_table_internal) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
