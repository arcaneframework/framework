// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ISimpleTableComparator.h                                    (C) 2000-2025 */
/*                                                                           */
/* Interface pour les services permettant de comparer un ISimpleTableOutput  */
/* et un fichier de référence.                                               */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ISIMPLETABLECOMPARATOR_H
#define ARCANE_CORE_ISIMPLETABLECOMPARATOR_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ISimpleTableOutput.h"

#include "arcane/core/ItemTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/**
 * @ingroup StandardService
 * @brief Interface de classe représentant un comparateur
 * de tableaux. À utiliser avec un service implémentant
 * ISimpleTableOutput.
 * 
 * La différence avec ISimpleTableInternalComparator est
 * que l'on compare un SimpleTableInternal contenu dans
 * un ISimpleTableOutput avec un SimpleTableInternal
 * généré à partir d'un fichier de référence.
 * 
 * Cette interface permet aussi de générer les fichiers
 * de référence en utilisant le nom de répertoire et
 * le nom de tableau du ISimpleTableOutput, permettant
 * de faciliter le processus.
 */
class ARCANE_CORE_EXPORT ISimpleTableComparator
{
 public:
  virtual ~ISimpleTableComparator() = default;

 public:
  /**
   * @brief Méthode permettant d'initialiser le service.
   * 
   * Le pointeur vers une implémentation de ISimpleTableOutput
   * doit contenir les valeurs à comparer ou à écrire en tant que
   * valeurs de référence et l'emplacement de destination des
   * fichiers de sorties, pour que soit automatiquement déterminé
   * l'emplacement des fichiers de réferences.
   * 
   * @param simple_table_output_ptr Une implémentation de ISimpleTableOutput.
   */
  virtual void init(ISimpleTableOutput* simple_table_output_ptr) = 0;

  /**
   * @brief Méthode permettant d'effacer les données lues par readReferenceFile().
   * @note Efface le SimpleTableInternal du comparateur sans toucher à celui du
   * SimpleTableOutput.
   */
  virtual void clear() = 0;

  /**
   * @brief Méthode permettant d'afficher le tableau lu.
   * 
   * @param rank Le processus qui doit afficher son tableau (-1 pour tous les processus).
   */
  virtual void print(Integer rank = 0) = 0;

  /**
   * @brief Méthode permettant de modifier le répertoire racine.
   * Cela permet d'écrire ou de rechercher des fichiers de réferences
   * autre part que dans le répertoire déterminé par l'implémentation.
   * 
   * Par défaut, pour l'implémentation csv, le répertoire racine est :
   * ./output/csv_ref/
   * 
   * @param root_directory Le nouveau répertoire racine.
   */
  virtual void editRootDirectory(const Directory& root_directory) = 0;

  /**
   * @brief Méthode permettant d'écrire les fichiers de référence.
   * 
   * @warning (Pour l'instant), cette méthode utilise l'objet pointé par 
   *          le pointeur donné lors de l'init(), donc l'écriture s'effectura 
   *          dans le format voulu par l'implémentation de ISimpleTableOutput.
   *          Si les formats de lecture et d'écriture ne correspondent
   *          pas, un appel à "compareWithReference()" retournera forcement
   *          false.
   * 
   * @param rank Le processus qui doit écrire son fichier (-1 pour tous les processus).
   * @return true Si l'écriture a bien eu lieu (et si processus appelant != rank).
   * @return false Si l'écriture n'a pas eu lieu.
   */
  virtual bool writeReferenceFile(Integer rank = -1) = 0;
  /**
   * @brief Méthode permettant de lire les fichiers de références.
   * 
   * Le type des fichiers de référence doit correspondre à l'implémentation
   * de cette interface choisi (exemple : fichier .csv -> SimpleCsvComparatorService).
   * 
   * @param rank Le processus qui doit lire son fichier (-1 pour tous les processus).
   * @return true Si le fichier a été lu (et si processus appelant != rank).
   * @return false Si le fichier n'a pas été lu.
   */
  virtual bool readReferenceFile(Integer rank = -1) = 0;

  /**
   * @brief Méthode permettant de savoir si les fichiers de réferences existent.
   * 
   * @param rank Le processus qui doit chercher son fichier (-1 pour tous les processus). 
   * @return true Si le fichier a été trouvé (et si processus appelant != rank).
   * @return false Si le fichier n'a pas été trouvé.
   */
  virtual bool isReferenceExist(Integer rank = -1) = 0;

  /**
   * @brief Méthode permettant de comparer l'objet de type ISimpleTableOutput
   * aux fichiers de réferences.
   * 
   * @param rank Le processus qui doit comparer ses résultats (-1 pour tous les processus). 
   * @param compare_dimension_too Si l'on doit aussi comparer les dimensions des tableaux de valeurs.
   * @return true S'il n'y a pas de différences (et si processus appelant != rank).
   * @return false S'il y a au moins une différence.
   */
  virtual bool compareWithReference(Integer rank = -1, bool compare_dimension_too = false) = 0;

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
   * @param rank Le processus qui doit comparer ses résultats (-1 pour tous les processus). 
   * @return true Si les deux valeurs sont égales.
   * @return false Si les deux valeurs sont différentes.
   */
  virtual bool compareElemWithReference(const String& column_name, const String& row_name, Integer rank = -1) = 0;

  /**
   * @brief Méthode permettant de comparer une valeur avec
   * une valeur du tableau de référence.
   * Cette méthode n'a pas besoin d'un internal 'toCompare' 
   * (setInternalToCompare() non nécessaire).
   * 
   * @param elem La valeur à comparer.
   * @param column_name Le nom de la colonne où se trouve l'élément de référence.
   * @param row_name Le nom de la ligne où se trouve l'élément de référence.
   * @param rank Le processus qui doit comparer ses résultats (-1 pour tous les processus). 
   * @return true Si les deux valeurs sont égales.
   * @return false Si les deux valeurs sont différentes.
   */
  virtual bool compareElemWithReference(Real elem, const String& column_name, const String& row_name, Integer rank = -1) = 0;

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
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
