// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ParameterCaseOption.h                                       (C) 2000-2025 */
/*                                                                           */
/* Classe représentant l'ensemble des paramètres pouvant modifier les        */
/* options du jeu de données .                                               */
/*---------------------------------------------------------------------------*/

#ifndef ARCANE_UTILS_PARAMETERCASEOPTION_H
#define ARCANE_UTILS_PARAMETERCASEOPTION_H

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/UtilsTypes.h"
#include "arcane/utils/String.h"
#include "arcane/utils/List.h"

#include "arcane/core/ICaseMng.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ParameterOptionElementsCollection;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Classe représentant l'ensemble des paramètres pouvant modifier
 * les options du jeu de données.
 */
class ARCANE_UTILS_EXPORT
ParameterCaseOption
{

 public:

  ParameterCaseOption(ICaseMng* case_mng);
  ~ParameterCaseOption();

 public:

  /*!
   * \brief Méthode permettant de récupérer la valeur d'une option.
   *
   * L'adresse de l'option est reformée comme ceci :
   * xpath_before_index[index]/xpath_after_index
   *
   * xpath_before_index doit être de la forme suivante :
   * //case/aaa/bbb[2]/ccc
   * - le "//case/" au début (ou "//cas/" en français"),
   * - une succession de tags avec possiblement leurs indices,
   * - pas de "/" à la fin,
   * - un indice peut être mise à la fin (mais il sera remplacé
   *   par celui passé en paramètre).
   *
   * xpath_after_index doit être de la forme suivante :
   * ddd/eee
   * - pas de "/" au début ni à la fin.
   *
   * Les indices sont des indices XML et ces indices commencent par 1.
   *
   * \param xpath_before_index L'adresse avant indice.
   * \param xpath_after_index L'adresse après indice.
   * \param index L'indice à mettre entre les deux parties de l'adresse.
   * \return La valeur si trouvée, sinon chaîne null.
   */
  String getParameterOrNull(const String& xpath_before_index, const String& xpath_after_index, Integer index) const;

  /*!
   * \brief Méthode permettant de récupérer la valeur d'une option.
   *
   * L'adresse de l'option est reformée comme ceci :
   * xpath_before_index[index]
   *
   * xpath_before_index doit être de la forme suivante :
   * //case/aaa/bbb[2]/ccc
   * - le "//case/" au début (ou "//cas/" en français"),
   * - une succession de tags avec possiblement leurs indices,
   * - pas de "/" à la fin,
   * - un indice peut être mise à la fin (mais il sera remplacé
   *   par celui passé en paramètre).
   *
   * Si le paramètre allow_elems_after_index est activé, les adresses de la forme :
   * xpath_before_index[index]/aaa/bbb
   * seront aussi recherchées.
   *
   * Les indices sont des indices XML et ces indices commencent par 1.
   *
   * \param xpath_before_index L'adresse avant indice.
   * \param index L'indice à mettre après l'adresse.
   * \param allow_elems_after_index Doit-on vérifier la présence d'éléments après l'indice ?
   * \return La valeur si trouvée, sinon chaîne null.
   */
  String getParameterOrNull(const String& xpath_before_index, Integer index, bool allow_elems_after_index) const;

  /*!
   * \brief Méthode permettant de récupérer la valeur d'une option.
   *
   * L'adresse doit être de la forme suivante :
   * //case/aaa/bbb[2]/ccc
   * - le "//case/" au début (ou "//cas/" en français"),
   * - une succession de tags avec possiblement leurs indices,
   * - pas de "/" à la fin,
   * - un indice peut être mise à la fin.
   *
   * Les indices sont des indices XML et ces indices commencent par 1.
   *
   * \param full_xpath L'adresse à rechercher.
   * \return La valeur si trouvée, sinon chaîne null.
   */
  String getParameterOrNull(const String& full_xpath) const;

  /*!
   * \brief Méthode permettant de savoir si une option est présente.
   *
   * L'adresse doit être de la forme suivante :
   * //case/aaa/bbb[2]/ccc
   * - le "//case/" au début (ou "//cas/" en français"),
   * - une succession de tags avec possiblement leurs indices,
   * - pas de "/" à la fin,
   * - un indice peut être mise à la fin.
   *
   * Les indices sont des indices XML et ces indices commencent par 1.
   *
   * \param full_xpath L'adresse à rechercher.
   * \return true si l'adresse est trouvée dans la liste.
   */
  bool exist(const String& full_xpath);

  /*!
   * \brief Méthode permettant de savoir si une option est présente.
   *
   * L'adresse de l'option est reformée comme ceci :
   * xpath_before_index[ANY_INDEX]/xpath_after_index
   *
   * xpath_before_index doit être de la forme suivante :
   * //case/aaa/bbb[2]/ccc
   * - le "//case/" au début (ou "//cas/" en français"),
   * - une succession de tags avec possiblement leurs indices,
   * - pas de "/" à la fin,
   * - un indice peut être mise à la fin (mais il sera remplacé
   *   par ANY_INDEX).
   *
   * xpath_after_index doit être de la forme suivante :
   * ddd/eee
   * - pas de "/" au début ni à la fin.
   *
   * Les indices sont des indices XML et ces indices commencent par 1.
   * L'indice ANY_INDEX est un indice spécial désignant tous les indices.
   *
   * \param xpath_before_index L'adresse avant indice.
   * \param xpath_after_index L'adresse après indice.
   * \return true si l'adresse est trouvée dans la liste.
   */
  bool existAnyIndex(const String& xpath_before_index, const String& xpath_after_index) const;

  /*!
   * \brief Méthode permettant de savoir si une option est présente.
   *
   * L'adresse de l'option est reformée comme ceci :
   * full_xpath[ANY_INDEX]
   *
   * L'adresse doit être de la forme suivante :
   * //case/aaa/bbb[2]/ccc
   * - le "//case/" au début (ou "//cas/" en français"),
   * - une succession de tags avec possiblement leurs indices,
   * - pas de "/" à la fin,
   * - un indice peut être mise à la fin (mais il sera remplacé
   *   par ANY_INDEX).
   *
   * Les indices sont des indices XML et ces indices commencent par 1.
   * L'indice ANY_INDEX est un indice spécial désignant tous les indices.
   *
   * \param full_xpath L'adresse à rechercher.
   * \return true si l'adresse est trouvée dans la liste.
   */
  bool existAnyIndex(const String& full_xpath) const;

  /*!
   * \brief Méthode permettant de récupérer le ou les indices de l'option.
   *
   * L'adresse de l'option est reformée comme ceci :
   * xpath_before_index[GET_INDEX]/xpath_after_index
   *
   * xpath_before_index doit être de la forme suivante :
   * //case/aaa/bbb[2]/ccc
   * - le "//case/" au début (ou "//cas/" en français"),
   * - une succession de tags avec possiblement leurs indices,
   * - pas de "/" à la fin,
   * - un indice peut être mise à la fin (mais il sera remplacé
   *   par GET_INDEX).
   *
   * xpath_after_index doit être de la forme suivante :
   * ddd/eee
   * - pas de "/" au début ni à la fin.
   *
   * Les indices sont des indices XML et ces indices commencent par 1.
   * L'indice GET_INDEX est un indice spécial désignant les indices que l'on souhaite récupérer.
   *
   * \param xpath_before_index L'adresse avant indice.
   * \param xpath_after_index L'adresse après indice.
   * \param indexes Le tableau qui contiendra l'ensemble des indices trouvés
   * (ce tableau n'est pas effacé avant utilisation).
   */
  void indexesInParam(const String& xpath_before_index, const String& xpath_after_index, UniqueArray<Integer>& indexes) const;

  /*!
   * \brief Méthode permettant de récupérer le ou les indices de l'option.
   *
   * L'adresse de l'option est reformée comme ceci :
   * xpath_before_index[GET_INDEX]
   *
   * xpath_before_index doit être de la forme suivante :
   * //case/aaa/bbb[2]/ccc
   * - le "//case/" au début (ou "//cas/" en français"),
   * - une succession de tags avec possiblement leurs indices,
   * - pas de "/" à la fin,
   * - un indice peut être mise à la fin (mais il sera remplacé
   *   par GET_INDEX).
   *
   * Si le paramètre allow_elems_after_index est activé, les adresses de la forme :
   * xpath_before_index[GET_INDEX]/aaa/bbb
   * seront aussi recherchées.
   *
   * Les indices sont des indices XML et ces indices commencent par 1.
   * L'indice GET_INDEX est un indice spécial désignant les indices que l'on souhaite récupérer.
   *
   * \param xpath_before_index L'adresse avant indice.
   * \param indexes Le tableau qui contiendra l'ensemble des indices trouvés
   * \param allow_elems_after_index Doit-on vérifier la présence d'éléments après l'indice ?
   * (ce tableau n'est pas effacé avant utilisation).
   */
  void indexesInParam(const String& xpath_before_index, UniqueArray<Integer>& indexes, bool allow_elems_after_index) const;

  /*!
   * \brief Méthode permettant de connaitre le nombre d'indices de l'option.
   *
   * L'adresse de l'option est reformée comme ceci :
   * xpath_before_index[GET_INDEX]/xpath_after_index
   *
   * xpath_before_index doit être de la forme suivante :
   * //case/aaa/bbb[2]/ccc
   * - le "//case/" au début (ou "//cas/" en français"),
   * - une succession de tags avec possiblement leurs indices,
   * - pas de "/" à la fin,
   * - un indice peut être mise à la fin (mais il sera remplacé
   *   par GET_INDEX).
   *
   * xpath_after_index doit être de la forme suivante :
   * ddd/eee
   * - pas de "/" au début ni à la fin.
   *
   * Les indices sont des indices XML et ces indices commencent par 1.
   * L'indice GET_INDEX est un indice spécial désignant les indices que l'on souhaite récupérer.
   *
   * \param xpath_before_index L'adresse avant indice.
   * \param xpath_after_index L'adresse après indice.
   * \return Le nombre d'indices de l'option.
   */
  Integer count(const String& xpath_before_index, const String& xpath_after_index) const;

  /*!
   * \brief Méthode permettant de connaitre le nombre d'indices de l'option.
   *
   * L'adresse de l'option est reformée comme ceci :
   * xpath_before_index[GET_INDEX]
   *
   * xpath_before_index doit être de la forme suivante :
   * //case/aaa/bbb[2]/ccc
   * - le "//case/" au début (ou "//cas/" en français"),
   * - une succession de tags avec possiblement leurs indices,
   * - pas de "/" à la fin,
   * - un indice peut être mise à la fin (mais il sera remplacé
   *   par GET_INDEX).
   *
   * Les indices sont des indices XML et ces indices commencent par 1.
   * L'indice GET_INDEX est un indice spécial désignant les indices que l'on souhaite récupérer.
   *
   * \param xpath_before_index L'adresse avant indice.
   * \return Le nombre d'indices de l'option.
   */
  Integer count(const String& xpath_before_index) const;

 private:

  inline StringView _removeUselessPartInXpath(StringView xpath) const;

 private:

  StringList m_param_names;
  StringList m_values;
  String m_lang;
  ICaseMng* m_case_mng;
  ParameterOptionElementsCollection* m_lines;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
