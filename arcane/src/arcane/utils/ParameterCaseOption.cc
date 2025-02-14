// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ParameterCaseOption.cc                                      (C) 2000-2025 */
/*                                                                           */
/* Classe représentant l'ensemble des paramètres pouvant modifier les        */
/* options du jeu de données.                                                */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ParameterCaseOption.h"

#include "arcane/utils/ApplicationInfo.h"
#include "arcane/utils/CommandLineArguments.h"
#include "arcane/utils/ParameterList.h"
#include "arcane/utils/ValueConvert.h"
#include "arcane/utils/Array.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/Ref.h"

#include "arcane/core/IApplication.h"
#include "arcane/core/ICaseDocument.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{
const Arcane::String ANY_TAG_STR = "/";
const Arcane::StringView ANY_TAG = ANY_TAG_STR.view();
constexpr Arcane::Integer ANY_INDEX = -1;
constexpr Arcane::Integer GET_INDEX = -2;
} // namespace

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
class ParameterOptionAddrPart
{
 public:

  /*!
   * \brief Constructeur. Définit le tag en ANY_TAG et l'index en ANY_INDEX.
   */
  ParameterOptionAddrPart()
  : m_tag(ANY_TAG)
  , m_index(ANY_INDEX)
  {}

  /*!
   * \brief Constructeur. Définit l'index à 1.
   * \param tag Le tag de cette partie d'adresse. Ce tag ne peut pas être ANY_TAG.
   */
  explicit ParameterOptionAddrPart(const StringView tag)
  : m_tag(tag)
  , m_index(1)
  {
    ARCANE_ASSERT(tag != ANY_TAG, ("ANY_TAG without ANY_INDEX is forbidden"));
    ARCANE_ASSERT(!tag.empty(), ("tag is empty"));
  }

  /*!
   * \brief Constructeur.
   * \param tag Le tag de cette partie d'adresse. Ce tag ne peut pas être ANY_TAG
   * si l'index n'est pas ANY_INDEX.
   * \param index L'index de cette partie d'adresse.
   */
  ParameterOptionAddrPart(const StringView tag, const Integer index)
  : m_tag(tag)
  , m_index(index)
  {
    ARCANE_ASSERT(index == ANY_INDEX || tag != ANY_TAG, ("ANY_TAG without ANY_INDEX is forbidden"));
    ARCANE_ASSERT(!tag.empty(), ("tag is empty"));
  }

 public:

  StringView tag() const
  {
    return m_tag;
  }
  Integer index() const
  {
    return m_index;
  }

  //! Si l'index est ANY_INDEX, le tag ne peut pas être ANY_TAG.
  void setTag(const StringView tag)
  {
    ARCANE_ASSERT(m_index == ANY_INDEX || tag != ANY_TAG, ("ANY_TAG without ANY_INDEX is forbidden"));
    ARCANE_ASSERT(!tag.empty(), ("tag is empty"));

    m_tag = tag;
  }
  void setIndex(const Integer index)
  {
    m_index = index;
  }

  //! isAny si ANY_TAG et ANY_INDEX.
  bool isAny() const
  {
    return (m_tag == ANY_TAG && m_index == ANY_INDEX);
  }

  /*!
   * \brief Opérateur d'égalité.
   * Le tag ANY_TAG est égal à tous les tags.
   * L'index ANY_INDEX est égal à tous les index.
   * L'index GET_INDEX est égal à tous les index.
   */
  bool operator==(const ParameterOptionAddrPart& other) const
  {
    return (m_tag == other.m_tag || m_tag == ANY_TAG || other.m_tag == ANY_TAG) &&
    (m_index == other.m_index || m_index == ANY_INDEX || other.m_index == ANY_INDEX || m_index == GET_INDEX || other.m_index == GET_INDEX);
  }
  // TODO AH : À supprimer lors du passage en C++20.
  bool operator!=(const ParameterOptionAddrPart& other) const
  {
    return !operator==(other);
  }

 private:

  StringView m_tag;
  Integer m_index;
};

std::ostream& operator<<(std::ostream& o, const ParameterOptionAddrPart& h)
{
  o << (h.tag() == ANY_TAG ? "ANY" : h.tag())
    << "[" << (h.index() == ANY_INDEX ? "ANY" : (h.index() == GET_INDEX ? "GET" : std::to_string(h.index())))
    << "]";
  return o;
}

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
class ParameterOptionAddr
{
 public:

  /*!
   * \brief Constructeur.
   * \param addr_str_view L'adresse à convertir.
   */
  explicit ParameterOptionAddr(const StringView addr_str_view)
  {
    Span span_line(addr_str_view.bytes());
    Integer begin = 0;
    Integer size = 0;
    Integer index_begin = -1;
    // On interdit les options qui s'appliquent à toutes les caseoptions.
    bool have_a_no_any = false;

    // aaa[0]
    for (Integer i = 0; i < span_line.size(); ++i) {
      if (span_line[i] == '[') {
        index_begin = i + 1;
        size = i - begin;
        ARCANE_ASSERT(size != 0, ("Invalid option (empty name)"));
        ARCANE_ASSERT(index_begin < span_line.size(), ("Invalid option (']' not found)"));
      }
      else if (span_line[i] == ']') {
        ARCANE_ASSERT(index_begin != -1, ("Invalid option (']' found without '[')"));

        // Motif spécial "[]" (= ANY_INDEX)
        if (index_begin == i) {
          m_parts.add(makeRef(new ParameterOptionAddrPart(addr_str_view.subView(begin, size), ANY_INDEX)));
          have_a_no_any = true;
        }
        else {
          StringView index_str = addr_str_view.subView(index_begin, i - index_begin);
          Integer index;
          bool is_bad = builtInGetValue(index, index_str);
          if (is_bad) {
            ARCANE_FATAL("Invalid index");
          }
          m_parts.add(makeRef(new ParameterOptionAddrPart(addr_str_view.subView(begin, size), index)));
          have_a_no_any = true;
        }
      }

      else if (span_line[i] == '/') {
        ARCANE_ASSERT(i + 1 != span_line.size(), ("Invalid option ('/' found at the end of the param option)"));

        if (index_begin == -1) {
          size = i - begin;
          // Cas ou on a un any_tag any_index ("truc1//truc2").
          if (size == 0) {
            m_parts.add(makeRef(new ParameterOptionAddrPart()));
          }
          else {
            m_parts.add(makeRef(new ParameterOptionAddrPart(addr_str_view.subView(begin, size))));
            have_a_no_any = true;
          }
        }

        begin = i + 1;
        size = 0;
        index_begin = -1;
      }
    }
    if (index_begin == -1) {
      size = static_cast<Integer>(span_line.size()) - begin;
      ARCANE_ASSERT(size != 0, ("Invalid option (empty name)"));

      m_parts.add(makeRef(new ParameterOptionAddrPart(addr_str_view.subView(begin, size))));
      have_a_no_any = true;
    }
    if (!have_a_no_any) {
      ARCANE_FATAL("Invalid option");
    }
  }

 public:

  // On ne doit pas bloquer les multiples ParameterOptionAddrPart(ANY) :
  // Construction par iteration : aaaa/bb/ANY/ANY/cc
  /*!
   * \brief Méthode permettant d'ajouter une partie à la fin de l'adresse actuelle.
   * \param part Un pointeur vers la nouvelle partie. Attention, on récupère la
   * propriété de l'objet (on gère le delete).
   */
  void addAddrPart(ParameterOptionAddrPart* part)
  {
    m_parts.add(makeRef(part));
  }

  /*!
   * \brief Méthode permettant de récupérer une partie de l'adresse.
   * Si l'adresse termine par un ANY_TAG[ANY_INDEX], tous index donnés en paramètre
   * supérieur au nombre de partie de l'adresse retournera le dernier élément de
   * l'adresse ("ANY_TAG[ANY_INDEX]").
   *
   * \param index_of_part L'index de la partie à récupérer.
   * \return La partie de l'adresse.
   */
  ParameterOptionAddrPart* addrPart(const Integer index_of_part) const
  {
    if (index_of_part >= m_parts.size()) {
      if (m_parts[m_parts.size() - 1]->isAny()) {
        return lastAddrPart();
      }
      ARCANE_FATAL("Invalid index");
    }
    return m_parts[index_of_part].get();
  }

  ParameterOptionAddrPart* lastAddrPart() const
  {
    return m_parts[m_parts.size() - 1].get();
  }

  /*!
   * \brief Méthode permettant de récupérer le nombre de partie de l'adresse.
   * Les parties égales à "ANY_TAG[ANY_INDEX]" sont comptées.
   *
   * \return Le nombre de partie de l'adresse.
   */
  Integer nbAddrPart() const
  {
    return m_parts.size();
  }

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
  bool getIndexInAddr(const ParameterOptionAddr& addr_with_get_index, ArrayView<Integer> indexes) const
  {
    if (!operator==(addr_with_get_index))
      return false;

    ARCANE_ASSERT(indexes.size() == addr_with_get_index.nbIndexToGetInAddr(), ("ArrayView too small"));

    Integer index = 0;
    for (Integer i = 0; i < addr_with_get_index.nbAddrPart(); ++i) {
      if (addr_with_get_index.addrPart(i)->index() == GET_INDEX) {
        Integer index_tag = addrPart(i)->index();
        if (index_tag == ANY_INDEX)
          return false;
        indexes[index++] = index_tag;
      }
    }
    return true;
  }

  /*!
   * \brief Méthode permettant de savoir combien il y a de "GET_INDEX" dans l'adresse.
   * \return Le nombre de "GET_INDEX".
   */
  Integer nbIndexToGetInAddr() const
  {
    Integer count = 0;
    for (const auto& elem : m_parts) {
      if (elem->index() == GET_INDEX) {
        count++;
      }
    }
    return count;
  }

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
  bool operator==(const ParameterOptionAddr& other) const
  {
    Integer nb_iter = 0;
    if (lastAddrPart()->isAny()) {
      nb_iter = nbAddrPart() - 1;
    }
    else if (other.lastAddrPart()->isAny()) {
      nb_iter = other.nbAddrPart() - 1;
    }
    else if (nbAddrPart() != other.nbAddrPart()) {
      return false;
    }
    else {
      nb_iter = nbAddrPart();
    }

    for (Integer i = 0; i < nb_iter; ++i) {
      if (*addrPart(i) != *other.addrPart(i)) {
        return false;
      }
    }
    return true;
  }

 private:

  UniqueArray<Ref<ParameterOptionAddrPart>> m_parts;
};

std::ostream& operator<<(std::ostream& o, const ParameterOptionAddr& h)
{
  Integer nb_part = h.nbAddrPart();
  if (nb_part != 0)
    o << *(h.addrPart(0));
  for (Integer i = 1; i < nb_part; ++i) {
    o << "/" << *(h.addrPart(i));
  }
  return o;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Classe représentant un élément XML (une option Arcane).
 * Cet élement a une adresse et une valeur.
 */
class ParameterOptionElement
{
 public:

  ParameterOptionElement(const StringView addr, const StringView value)
  : m_addr(addr)
  , m_value(value)
  {}

  ParameterOptionAddr addr() const
  {
    return m_addr;
  }

  StringView value() const
  {
    return m_value;
  }

  bool operator==(const ParameterOptionAddr& addr) const
  {
    return m_addr == addr;
  }

 private:

  ParameterOptionAddr m_addr;
  StringView m_value;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Classe représentant un ensemble d'éléments XML (un ensemble d'options Arcane).
 */
class ParameterOptionElementsCollection
{
 public:

  void addElement(StringView addr, StringView value)
  {
    m_elements.add({ addr, value });
  }

  // ParameterOptionElement element(const Integer index)
  // {
  //   return m_elements[index];
  // }

  // Un StringView "vide" est éqal à un StringView "nul".
  // Comme on travaille avec des String et que la distinction
  // vide/nul est importante, on passe par un std::optional.
  std::optional<StringView> value(const ParameterOptionAddr& addr)
  {
    for (const auto& elem : m_elements) {
      if (elem == addr)
        return elem.value();
    }
    return {};
  }

  /*!
   * \brief Méthode permettant de savoir si une adresse est présente dans la liste d'éléments.
   * Les ANY_TAG/ANY_INDEX sont pris en compte.
   * \param addr L'adresse à rechercher.
   * \return true si l'adresse est trouvé.
   */
  bool isExistAddr(const ParameterOptionAddr& addr)
  {
    for (const auto& elem : m_elements) {
      if (elem == addr)
        return true;
    }
    return false;
  }

  /*!
   * \brief Méthode permettant de savoir combien de fois une adresse est présente dans la liste d'élements.
   * Méthode particulièrement utile avec les ANY_TAG/ANY_INDEX.
   *
   * \param addr L'adresse à rechercher.
   * \return Le nombre de correspondances trouvé.
   */
  Integer countAddr(const ParameterOptionAddr& addr)
  {
    Integer count = 0;
    for (const auto& elem : m_elements) {
      if (elem == addr)
        count++;
    }
    return count;
  }

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
  void getIndexInAddr(const ParameterOptionAddr& addr_with_get_index, UniqueArray<Integer>& indexes)
  {
    UniqueArray<Integer> new_indexes(addr_with_get_index.nbIndexToGetInAddr());
    for (const auto& elem : m_elements) {
      if (elem.addr().getIndexInAddr(addr_with_get_index, new_indexes)) {
        indexes.addRange(new_indexes);
      }
    }
  }

 private:

  UniqueArray<ParameterOptionElement> m_elements;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ParameterCaseOption::
ParameterCaseOption(ICaseMng* case_mng)
: m_case_mng(case_mng)
, m_lines(new ParameterOptionElementsCollection)
{
  m_lang = m_case_mng->caseDocumentFragment()->language();

  m_case_mng->application()->applicationInfo().commandLineArguments().parameters().fillParameters(m_param_names, m_values);

  for (Integer i = 0; i < m_param_names.count(); ++i) {
    const String& param = m_param_names[i];
    if (param.startsWith("//")) {
      m_lines->addElement(param.view().subView(2), m_values[i].view());
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ParameterCaseOption::
~ParameterCaseOption()
{
  delete m_lines;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String ParameterCaseOption::
getParameterOrNull(const String& xpath_before_index, const String& xpath_after_index, Integer index) const
{
  if (index <= 0) {
    ARCANE_FATAL("Index in XML start at 1");
  }

  ParameterOptionAddr addr{ _removeUselessPartInXpath(xpath_before_index.view()) };
  addr.lastAddrPart()->setIndex(index);
  addr.addAddrPart(new ParameterOptionAddrPart(xpath_after_index.view()));

  std::optional<StringView> value = m_lines->value(addr);
  if (value.has_value())
    return value.value();
  return {};
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String ParameterCaseOption::
getParameterOrNull(const String& xpath_before_index, Integer index, bool allow_elems_after_index) const
{
  if (index <= 0) {
    ARCANE_FATAL("Index in XML start at 1");
  }
  ParameterOptionAddr addr{ _removeUselessPartInXpath(xpath_before_index.view()) };
  addr.lastAddrPart()->setIndex(index);
  if (allow_elems_after_index) {
    addr.addAddrPart(new ParameterOptionAddrPart());
  }

  std::optional<StringView> value = m_lines->value(addr);
  if (value.has_value())
    return value.value();
  return {};
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String ParameterCaseOption::
getParameterOrNull(const String& full_xpath) const
{
  const ParameterOptionAddr addr{ _removeUselessPartInXpath(full_xpath.view()) };

  std::optional<StringView> value = m_lines->value(addr);
  if (value.has_value())
    return value.value();
  return {};
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool ParameterCaseOption::
exist(const String& full_xpath)
{
  const ParameterOptionAddr addr{ _removeUselessPartInXpath(full_xpath.view()) };
  return m_lines->isExistAddr(addr);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool ParameterCaseOption::
existAnyIndex(const String& xpath_before_index, const String& xpath_after_index) const
{
  ParameterOptionAddr addr{ _removeUselessPartInXpath(xpath_before_index.view()) };
  addr.lastAddrPart()->setIndex(ANY_INDEX);

  addr.addAddrPart(new ParameterOptionAddrPart(xpath_after_index.view()));

  return m_lines->isExistAddr(addr);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool ParameterCaseOption::
existAnyIndex(const String& full_xpath) const
{
  const ParameterOptionAddr addr{ _removeUselessPartInXpath(full_xpath.view()) };
  addr.lastAddrPart()->setIndex(ANY_INDEX);

  return m_lines->isExistAddr(addr);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParameterCaseOption::
indexesInParam(const String& xpath_before_index, const String& xpath_after_index, UniqueArray<Integer>& indexes) const
{
  ParameterOptionAddr addr{ _removeUselessPartInXpath(xpath_before_index.view()) };
  addr.lastAddrPart()->setIndex(GET_INDEX);
  addr.addAddrPart(new ParameterOptionAddrPart(xpath_after_index.view()));

  m_lines->getIndexInAddr(addr, indexes);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ParameterCaseOption::
indexesInParam(const String& xpath_before_index, UniqueArray<Integer>& indexes, bool allow_elems_after_index) const
{
  ParameterOptionAddr addr{ _removeUselessPartInXpath(xpath_before_index.view()) };
  addr.lastAddrPart()->setIndex(GET_INDEX);
  if (allow_elems_after_index) {
    addr.addAddrPart(new ParameterOptionAddrPart());
  }

  m_lines->getIndexInAddr(addr, indexes);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer ParameterCaseOption::
count(const String& xpath_before_index, const String& xpath_after_index) const
{
  ParameterOptionAddr addr{ _removeUselessPartInXpath(xpath_before_index.view()) };
  addr.lastAddrPart()->setIndex(ANY_INDEX);
  addr.addAddrPart(new ParameterOptionAddrPart(xpath_after_index.view()));

  return m_lines->countAddr(addr);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer ParameterCaseOption::
count(const String& xpath_before_index) const
{
  const ParameterOptionAddr addr{ _removeUselessPartInXpath(xpath_before_index.view()) };
  addr.lastAddrPart()->setIndex(ANY_INDEX);

  return m_lines->countAddr(addr);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

inline StringView ParameterCaseOption::
_removeUselessPartInXpath(StringView xpath) const
{
  if (m_lang == "fr")
    return xpath.subView(6);
  return xpath.subView(7);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
