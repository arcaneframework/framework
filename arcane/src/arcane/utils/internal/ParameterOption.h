// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ParameterOption.h                                           (C) 2000-2025 */
/*                                                                           */
/* Classe représentant l'ensemble des paramètres pouvant modifier les        */
/* options du jeu de données.                                                */
/*---------------------------------------------------------------------------*/

#ifndef ARCANE_UTILS_INTERNAL_PARAMETEROPTION_H
#define ARCANE_UTILS_INTERNAL_PARAMETEROPTION_H

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/UtilsTypes.h"
#include "arcane/utils/String.h"
#include "arcane/utils/List.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Classe représentant une partie d'une adresse d'option du jeu de données.
 * À noter qu'en XML, l'index commence à 1 et non à 0.
 *
 * Un tag spécial nommé ANY_TAG représente n'importe quel tag.
 * Deux index spéciaux sont aussi disponibles :
 * - ANY_INDEX : Représente n'importe quel index,
 * - GET_INDEX : Représente un index à récupérer (voir la classe ParameterOptionAddr).
 * Ces élements sont utiles pour l'opérateur ==.
 * À noter que ANY_TAG ne peut pas être définit sans ANY_INDEX.
 * Aussi, le tag ne peut pas être vide.
 */
class ARCANE_UTILS_EXPORT
ParameterOptionAddrPart
{
 public:
  static constexpr const char* ANY_TAG = "/";
  static constexpr Integer ANY_INDEX = -1;
  static constexpr Integer GET_INDEX = -2;

 public:

  /*!
   * \brief Constructeur. Définit le tag en ANY_TAG et l'index en ANY_INDEX.
   */
  ParameterOptionAddrPart();

  /*!
   * \brief Constructeur. Définit l'index à 1.
   * \param tag Le tag de cette partie d'adresse. Ce tag ne peut pas être ANY_TAG.
   */
  explicit ParameterOptionAddrPart(const StringView tag);

  /*!
   * \brief Constructeur.
   * \param tag Le tag de cette partie d'adresse. Ce tag ne peut pas être ANY_TAG
   * si l'index n'est pas ANY_INDEX.
   * \param index L'index de cette partie d'adresse.
   */
  ParameterOptionAddrPart(const StringView tag, const Integer index);

 public:

  StringView tag() const;
  Integer index() const;

  //! Si l'index est ANY_INDEX, le tag ne peut pas être ANY_TAG.
  //! Attention à la durée de vie de tag.
  void setTag(const StringView tag);
  void setIndex(const Integer index);

  //! isAny si ANY_TAG et ANY_INDEX.
  bool isAny() const;

  /*!
   * \brief Opérateur d'égalité.
   * Le tag ANY_TAG est égal à tous les tags.
   * L'index ANY_INDEX est égal à tous les index.
   * L'index GET_INDEX est égal à tous les index.
   */
  bool operator==(const ParameterOptionAddrPart& other) const;
  // TODO AH : À supprimer lors du passage en C++20.
  bool operator!=(const ParameterOptionAddrPart& other) const;

 private:

  StringView m_tag;
  Integer m_index;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Classe représentant une adresse d'option du jeu de données.
 * Cette adresse doit être de la forme : "tag/tag[index]/tag"
 * Les parties de l'adresse sans index auront l'index par défaut (=1).
 *
 * Cette adresse doit obéir à certaines règles :
 * - elle ne doit pas être vide,
 * - elle ne doit pas représenter l'ensemble des options ("/"),
 * - ses tags peuvent être vides ssi l'index est vide (voir après),
 * - l'index spécial ANY_INDEX ne peut être présent que si le tag est non vide,
 * - l'adresse peut terminer par un attribut ("\@name"),
 * - l'adresse donnée au constructeur ne peut pas terminer par un ANY_TAG (mais
 *   ANY_TAG peut être ajouté après avec la méthode addAddrPart()),
 *
 * Dans une chaine de caractères :
 * - le motif ANY_TAG[ANY_INDEX] peut être défini avec "//" :
 *   -> "tag/tag//tag" sera convertie ainsi : "tag[1]/tag[1]/ANY_TAG[ANY_INDEX]/tag[1]".
 * - l'index ANY_INDEX peut être défini avec un index vide "[]" :
 *   -> "tag/tag[]/\@attr" sera convertie ainsi : "tag[1]/tag[ANY_INDEX]/\@attr[1]",
 *   -> le motif "tag/[]/tag" est interdit.
 */
class ARCANE_UTILS_EXPORT
ParameterOptionAddr
{
 public:

  /*!
   * \brief Constructeur.
   * \param addr_str_view L'adresse à convertir.
   */
  explicit ParameterOptionAddr(StringView addr_str_view);

 public:

  // On ne doit pas bloquer les multiples ParameterOptionAddrPart(ANY) :
  // Construction par iteration : aaaa/bb/ANY/ANY/cc
  /*!
   * \brief Méthode permettant d'ajouter une partie à la fin de l'adresse actuelle.
   * \param part Un pointeur vers la nouvelle partie. Attention, on récupère la
   * propriété de l'objet (on gère le delete).
   */
  void addAddrPart(ParameterOptionAddrPart* part);

  /*!
   * \brief Méthode permettant de récupérer une partie de l'adresse.
   * Si l'adresse termine par un ANY_TAG[ANY_INDEX], tous index donnés en paramètre
   * supérieur au nombre de partie de l'adresse retournera le dernier élément de
   * l'adresse ("ANY_TAG[ANY_INDEX]").
   *
   * \param index_of_part L'index de la partie à récupérer.
   * \return La partie de l'adresse.
   */
  ParameterOptionAddrPart* addrPart(const Integer index_of_part) const;

  ParameterOptionAddrPart* lastAddrPart() const;

  /*!
   * \brief Méthode permettant de récupérer le nombre de partie de l'adresse.
   * Les parties égales à "ANY_TAG[ANY_INDEX]" sont comptées.
   *
   * \return Le nombre de partie de l'adresse.
   */
  Integer nbAddrPart() const;

  /*!
   * \brief Méthode permettant de récupérer un ou plusieurs indices dans l'adresse.
   *
   * Le fonctionnement de cette méthode est simple.
   * Nous avons l'adresse suivante :          "aaa[1]/bbb[2]/ccc[4]/\@name[1]".
   * L'adresse en paramètre est la suivante : "aaa[1]/bbb[GET_INDEX]/ccc[4]/\@name[1]".
   * L'indice ajouté dans la vue en paramètre sera 2.
   *
   * Si l'adresse en paramètre est :          "aaa[1]/bbb[GET_INDEX]/ccc[GET_INDEX]/\@name[1]".
   * Les indices ajoutés dans la vue seront 2 et 4.
   *
   * En revanche, un "GET_INDEX" ne peut pas être utilisé sur un "ANY_INDEX" (return false).
   * Exemple : si l'on a :              "aaa[1]/bbb[ANY_INDEX]/ccc[4]/\@name[1]".
   * Et si l'adresse en paramètre est : "aaa[1]/bbb[GET_INDEX]/ccc[GET_INDEX]/\@name[1]".
   * Le booléen retourné sera false.
   *
   * Pour avoir la bonne taille de la vue, un appel à la méthode "nbIndexToGetInAddr()"
   * peut être effectué.
   *
   * \param addr_with_get_index L'adresse contenant des indices "GET_INDEX".
   * \param indexes [OUT] La vue dans laquelle sera ajouté le ou les indices (la taille devra être correct).
   * \return true si la vue a pu être remplie correctement.
   */
  bool getIndexInAddr(const ParameterOptionAddr& addr_with_get_index, ArrayView<Integer> indexes) const;

  /*!
   * \brief Méthode permettant de savoir combien il y a de "GET_INDEX" dans l'adresse.
   * \return Le nombre de "GET_INDEX".
   */
  Integer nbIndexToGetInAddr() const;

 public:

  /*!
   * \brief Opérateur d'égalité.
   * Cet opérateur tient compte des ANY_TAG / ANY_INDEX.
   * L'adresse              "aaa[1]/bbb[2]/ANY_TAG[ANY_INDEX]"
   * sera éqale à l'adresse "aaa[1]/bbb[2]/ccc[5]/ddd[7]"
   * ou à l'adresse         "aaa[1]/bbb[ANY_INDEX]/ccc[5]/ddd[7]"
   * ou à l'adresse         "aaa[1]/bbb[2]"
   * mais pas à l'adresse   "aaa[1]"
   */
  bool operator==(const ParameterOptionAddr& other) const;

  // TODO AH : À supprimer lors du passage en C++20.
  bool operator!=(const ParameterOptionAddr& other) const;

 private:

  UniqueArray<Ref<ParameterOptionAddrPart>> m_parts;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Classe représentant un élément XML (une option Arcane).
 * Cet élement a une adresse et une valeur.
 */
class ARCANE_UTILS_EXPORT
ParameterOptionElement
{
 public:

  ParameterOptionElement(const StringView addr, const StringView value);

  ParameterOptionAddr addr() const;

  StringView value() const;

  bool operator==(const ParameterOptionAddr& addr) const;

 private:

  ParameterOptionAddr m_addr;
  StringView m_value;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Classe représentant un ensemble d'éléments XML (un ensemble d'options Arcane).
 */
class ARCANE_UTILS_EXPORT
ParameterOptionElementsCollection
{
 public:

  /*!
   * \brief Méthode permettant d'ajouter un paramètre d'option dans la liste
   * des paramètres d'options.
   *
   * \warning Les deux paramètres ne sont pas copiés ! On ne récupère qu'une vue. L'utilisateur
   * de cette classe doit gérer la durée de vie de ces objets.
   *
   * \param parameter Le paramètre d'option brut (avec les "//" au début).
   * \param value La valeur de l'option.
   */
  void addParameter(const String& parameter, const String& value);

  void addElement(StringView addr, StringView value);

  // ParameterOptionElement element(const Integer index)
  // {
  //   return m_elements[index];
  // }

  // Un StringView "vide" est éqal à un StringView "nul".
  // Comme on travaille avec des String et que la distinction
  // vide/nul est importante, on passe par un std::optional.
  std::optional<StringView> value(const ParameterOptionAddr& addr);

  /*!
   * \brief Méthode permettant de savoir si une adresse est présente dans la liste d'éléments.
   * Les ANY_TAG/ANY_INDEX sont pris en compte.
   * \param addr L'adresse à rechercher.
   * \return true si l'adresse est trouvé.
   */
  bool isExistAddr(const ParameterOptionAddr& addr);

  /*!
   * \brief Méthode permettant de savoir combien de fois une adresse est présente dans la liste d'élements.
   * Méthode particulièrement utile avec les ANY_TAG/ANY_INDEX.
   *
   * \param addr L'adresse à rechercher.
   * \return Le nombre de correspondances trouvé.
   */
  Integer countAddr(const ParameterOptionAddr& addr);

  /*!
   * \brief Méthode permettant de récupérer un ou plusieurs indices dans la liste d'adresses.
   *
   * Le fonctionnement de cette méthode est simple.
   * Nous avons les adresses suivantes :      "aaa[1]/bbb[2]/ccc[1]/\@name[1]".
   *                                          "aaa[1]/bbb[2]/ccc[2]/\@name[1]".
   *                                          "ddd[1]/eee[2]".
   *                                          "fff[1]/ggg[2]/hhh[4]".
   * L'adresse en paramètre est la suivante : "aaa[1]/bbb[2]/ccc[GET_INDEX]/\@name[1]".
   * Les indices ajoutés dans le tableau en paramètre seront 1 et 2.
   *
   * Attention : Avoir une adresse en entrée avec plusieurs "GET_INDEX" est autorisé mais
   * ça peut être dangereux si le nombre d'indices trouvé par adresse est différent pour
   * chaque adresse (s'il y a deux "GET_INDEX" mais que dans une des adresses, il n'y a
   * pas deux correspondances, ces éventuelles correspondances ne seront pas prises en
   * compte).
   *
   * \param addr_with_get_index L'adresse contenant des indices "GET_INDEX".
   * \param indexes [OUT] Le tableau dans lequel sera ajouté le ou les indices (le tableau
   * n'est pas effacé avant utilisation).
   */
  void getIndexInAddr(const ParameterOptionAddr& addr_with_get_index, UniqueArray<Integer>& indexes);

 private:

  UniqueArray<ParameterOptionElement> m_elements;
};


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
