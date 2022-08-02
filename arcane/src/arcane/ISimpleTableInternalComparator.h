// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/*                                        (C) 2000-2022 */
/*                                                                           */
/* TODO    */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifndef ARCANE_ISIMPLETABLEINTERNALCOMPARATOR_H
#define ARCANE_ISIMPLETABLEINTERNALCOMPARATOR_H

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
 * @brief TODO
 */
class ARCANE_CORE_EXPORT ISimpleTableInternalComparator
{
public:
  virtual ~ISimpleTableInternalComparator() = default;

public:

  /**
   * @brief Méthode permettant de comparer l'objet de type ISimpleTableOutput
   * aux fichiers de réference.
   * 
   * @param only_proc Le processus qui doit comparer ses résultats (-1 pour tous les processus). 
   * @param epsilon La marge d'erreur.
   * @return true S'il n'y a pas de différences.
   * @return false S'il y a au moins une différence (et si processus appelant != only_proc).
   */
  virtual bool compare(Integer epsilon = 0) = 0;

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


  virtual SimpleTableInternal* internalRef() = 0;
  virtual void setInternalRef(SimpleTableInternal* sti) = 0;
  virtual void setInternalRef(SimpleTableInternal& sti) = 0;

  virtual SimpleTableInternal* internalToCompare() = 0;
  virtual void setInternalToCompare(SimpleTableInternal* sti) = 0;
  virtual void setInternalToCompare(SimpleTableInternal& sti) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
