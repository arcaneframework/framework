// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ISimpleTableOutput.hh                                       (C) 2000-2022 */
/*                                                                           */
/* Interface pour simples services de comparaison de tableaux générés par    */
/* ISimpleTableOutput.                                                       */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifndef ARCANE_ISIMPLETABLECOMPARATOR_H
#define ARCANE_ISIMPLETABLECOMPARATOR_H

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include <arcane/ItemTypes.h>
#include <arcane/ISimpleTableOutput.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/**
 * @ingroup StandardService
 * @brief Interface représentant un comparateur de tableau simple.
 * @warning Interface non définitive !
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
   * fichiers de sortie, pour que soit automatiquement déterminé
   * l'emplacement des fichiers de réferences.
   * 
   * @param ptr_sto Une implémentation de ISimpleTableOutput.
   */
  virtual void init(ISimpleTableOutput* ptr_sto) = 0;

  /**
   * @brief Méthode permettant de remettre à zero l'objet.
   * Necessite un appel à init() après.
   */
  virtual void clear() = 0;
  
  virtual void print(Integer only_proc = 0) = 0;

  virtual void editRootDir(Directory root_dir) = 0;

  /**
   * @brief Méthode permettant d'écrire les fichiers de référence.
   * 
   * @warning Cette méthode utilise l'objet pointé par le pointeur donné
   *          lors de l'init(), donc l'écriture s'effectura dans le format
   *          voulu par l'implémentation de ISimpleTableOutput.
   *          Si les formats de lecture et d'écriture ne correspondent
   *          pas, un appel à "compareWithRef()" retournera forcement
   *          false.
   * 
   * @param only_proc Le processus qui doit écrire son fichier (-1 pour tous les processus).
   * @return true Si l'écriture a bien eu lieu.
   * @return false Si l'écriture n'a pas eu lieu (et si processus appelant != only_proc).
   */
  virtual bool writeRefFile(Integer only_proc = -1) = 0;
  /**
   * @brief Méthode permettant de lire les fichiers de références.
   * 
   * Le type des fichiers de réference doit correspondre à l'implémentation
   * de cette interface choisi (exemple : fichier .csv -> SimpleCsvComparatorService).
   * 
   * @param only_proc Le processus qui doit lire son fichier (-1 pour tous les processus).
   * @return true Si le fichier a été lu.
   * @return false Si le fichier n'a pas été lu (et si processus appelant != only_proc).
   */
  virtual bool readRefFile(Integer only_proc = -1) = 0;

  /**
   * @brief Méthode permettant de savoir si les fichiers de réferences existent.
   * 
   * @param only_proc Le processus qui doit chercher son fichier (-1 pour tous les processus). 
   * @return true Si le fichier a été trouvé.
   * @return false Si le fichier n'a pas été trouvé (et si processus appelant != only_proc).
   */
  virtual bool isRefExist(Integer only_proc = -1) = 0;

  /**
   * @brief Méthode permettant de comparer l'objet de type ISimpleTableOutput
   * aux fichiers de réference.
   * 
   * @param only_proc Le processus qui doit comparer ses résultats (-1 pour tous les processus). 
   * @param epsilon La marge d'erreur.
   * @return true S'il n'y a pas de différences.
   * @return false S'il y a au moins une différence (et si processus appelant != only_proc).
   */
  virtual bool compareWithRef(Integer only_proc = -1, Integer epsilon = 0) = 0;

  /**
   * @brief Méthode permettant d'ajouter une colonne dans la liste des colonnes
   * à comparer.
   * 
   * @param name_column Le nom de la colonne à comparer.
   * @return true Si le nom a bien été ajouté.
   * @return false Sinon.
   */
  virtual bool addColumnToCompare(String name_column) = 0;
  /**
   * @brief Méthode permettant d'ajouter une ligne dans la liste des lignes
   * à comparer.
   * 
   * @param name_row Le nom de la ligne à comparer.
   * @return true Si le nom a bien été ajouté.
   * @return false Sinon.
   */
  virtual bool addRowToCompare(String name_row) = 0;

  /**
   * @brief Méthode permettant de supprimer une colonne de la liste des
   * colonnes à comparer.
   * 
   * @param name_column Le nom de la colonne à supprimer de la liste.
   * @return true Si la suppression a eu lieu.
   * @return false Sinon.
   */
  virtual bool removeColumnToCompare(String name_column) = 0;
  /**
   * @brief Méthode permettant de supprimer une ligne de la liste des
   * lignes à comparer.
   * 
   * @param name_row Le nom de la ligne à supprimer de la liste.
   * @return true Si la suppression a eu lieu.
   * @return false Sinon.
   */
  virtual bool removeRowToCompare(String name_row) = 0;

  /**
   * @brief Méthode permettant d'ajouter une expression régulière
   * permettant de déterminer les colonnes à comparer.
   * 
   * @param regex_column L'expression régulière (format ECMAScript).
   */
  virtual void editRegexColumns(String regex_column) = 0;
  /**
   * @brief Méthode permettant d'ajouter une expression régulière
   * permettant de déterminer les lignes à comparer.
   * 
   * @param regex_row L'expression régulière (format ECMAScript).
   */
  virtual void editRegexRows(String regex_row) = 0;

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
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
