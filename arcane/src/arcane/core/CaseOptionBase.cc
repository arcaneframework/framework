// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CaseOptionBase.cc                                           (C) 2000-2025 */
/*                                                                           */
/* Gestion des options du jeu de données.                                    */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/CaseOptionBase.h"

#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/StringBuilder.h"

#include "arcane/core/CaseOptionBuildInfo.h"
#include "arcane/core/StringDictionary.h"
#include "arcane/core/CaseOptions.h"
#include "arcane/core/ICaseMng.h"
#include "arcane/core/ICaseDocument.h"
#include "arcane/core/CaseOptionException.h"
#include "arcane/core/internal/ICaseOptionListInternal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 *\brief Implémentation de la classe de base d'une option du jeu de données.
 */
class CaseOptionBasePrivate
{
 public:

  explicit CaseOptionBasePrivate(const CaseOptionBuildInfo& cob);

 public:

  ICaseMng* m_case_mng = nullptr; //!< Gestionnaire du sous-domaine
  ICaseOptionList* m_parent_option_list = nullptr; //!< Parent
  ICaseDocumentFragment* m_case_document_fragment = nullptr; //!< Document associé
  XmlNode m_root_element; //!< Elément du DOM de l'option
  String m_true_name; //!< Nom de l'option
  String m_name; //!< Nom traduit de l'option
  const String m_axl_default_value; //!< Valeur par défaut initiale
  String m_default_value; //!< Valeur par défaut
  Integer m_min_occurs; //!< Nombre minimum d'occurences
  Integer m_max_occurs; //!< Nombre maximum d'occurences (-1 == unbounded)
  bool m_is_optional;
  bool m_is_initialized; //!< \a true si initialisé
  bool m_is_override_default; //!< \a true si la valeur par défaut est surchargée
  //! Liste des noms d'options par langue.
  StringDictionary m_name_translations;
  //! Liste des valeurs par défaut par catégorie.
  StringDictionary m_default_values;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CaseOptionBasePrivate::
CaseOptionBasePrivate(const CaseOptionBuildInfo& cob)
: m_case_mng(cob.caseMng())
, m_parent_option_list(cob.caseOptionList())
, m_case_document_fragment(m_parent_option_list->caseDocumentFragment())
, m_root_element()
, m_true_name(cob.name())
, m_name(m_true_name)
, m_axl_default_value(cob.defaultValue())
, m_default_value(m_axl_default_value)
, m_min_occurs(cob.minOccurs())
, m_max_occurs(cob.maxOccurs())
, m_is_optional(cob.isOptional())
, m_is_initialized(false)
, m_is_override_default(false)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CaseOptionBase::
CaseOptionBase(const CaseOptionBuildInfo& cob)
: m_p(new CaseOptionBasePrivate(cob))
{
  cob.caseOptionList()->_internalApi()->addConfig(this,cob.element());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CaseOptionBase::
~CaseOptionBase()
{
  delete m_p;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ICaseMng* CaseOptionBase::
caseMng() const
{
  return m_p->m_case_mng;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ICaseOptionList* CaseOptionBase::
parentOptionList() const
{
  return m_p->m_parent_option_list;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ITraceMng* CaseOptionBase::
traceMng() const
{
  return m_p->m_case_mng->traceMng();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ISubDomain* CaseOptionBase::
subDomain() const
{
  return m_p->m_case_mng->subDomain();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ICaseDocument* CaseOptionBase::
caseDocument() const
{
  return caseMng()->caseDocument();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ICaseDocumentFragment* CaseOptionBase::
caseDocumentFragment() const
{
  return m_p->m_case_document_fragment;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String CaseOptionBase::
_defaultValue() const
{
  return m_p->m_default_value;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseOptionBase::
_setDefaultValue(const String& def_value)
{
  m_p->m_default_value = def_value;
  m_p->m_is_override_default = true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseOptionBase::
search(bool is_phase1)
{
  _setCategoryDefaultValue();
  _setTranslatedName();
  _search(is_phase1);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String CaseOptionBase::
trueName() const
{
  return m_p->m_true_name;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String CaseOptionBase::
name() const
{
  return m_p->m_name;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer CaseOptionBase::
minOccurs() const
{
  return m_p->m_min_occurs;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer CaseOptionBase::
maxOccurs() const
{
  return m_p->m_max_occurs;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool CaseOptionBase::
isOptional() const
{
  return m_p->m_is_optional;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseOptionBase::
_setTranslatedName()
{
  String lang = caseDocumentFragment()->language();
  if (lang.null())
    m_p->m_name = m_p->m_true_name;
  else{
    String tr = m_p->m_name_translations.find(lang);
    if (!tr.null()){
      //cerr << "** TRANSLATION FOR " << m_p->m_true_name << " is " << tr << " in " << lang << '\n';
      m_p->m_name = tr;
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseOptionBase::
_setCategoryDefaultValue()
{
  // Si le développeur a surchargé l'option, ne fait rien
  if (m_p->m_is_override_default)
    return;
  String category = caseDocumentFragment()->defaultCategory();
  if (category.null())
    m_p->m_default_value = m_p->m_axl_default_value;
  else{
    String v = m_p->m_default_values.find(category);
    if (!v.null()){
      m_p->m_default_value = v;
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String CaseOptionBase::
translatedName(const String& lang) const
{
  if (!lang.null()){
    String tr = m_p->m_name_translations.find(lang);
    if (!tr.null())
      return tr;
  }
  return m_p->m_name;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseOptionBase::
setRootElement(const XmlNode& root_element)
{
  m_p->m_root_element = root_element;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

XmlNode CaseOptionBase::
rootElement() const
{
  return m_p->m_root_element;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseOptionBase::
addAlternativeNodeName(const String& lang,const String& name)
{
  m_p->m_name_translations.add(lang,name);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseOptionBase::
addDefaultValue(const String& category,const String& value)
{
  m_p->m_default_values.add(category,value);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseOptionBase::
_setIsInitialized()
{
  m_p->m_is_initialized = true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool CaseOptionBase::
_isInitialized() const
{
  return m_p->m_is_initialized;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseOptionBase::
_checkIsInitialized() const
{
  if (!_isInitialized()){
    ARCANE_THROW(CaseOptionException,"option non initialisée '{0}'",name());
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseOptionBase::
_checkMinMaxOccurs(Integer nb_occur)
{
  Integer min_occurs = m_p->m_min_occurs;
  Integer max_occurs = m_p->m_max_occurs;
  bool is_optional = m_p->m_is_optional;

  if (nb_occur == 0 && is_optional) {
    return;
  }

  if (nb_occur<min_occurs){
    StringBuilder msg = "Bad number of occurences (less than min)";
    msg += " nb_occur=";
    msg += nb_occur;
    msg += " min_occur=";
    msg += min_occurs;
    msg += " option=";
    msg += m_p->m_root_element.xpathFullName();
    msg += "/";
    msg += name();
    throw CaseOptionException(A_FUNCINFO,msg.toString(),true);
  }
  if (max_occurs>=0)
    if (nb_occur>max_occurs){
      StringBuilder msg = "Bad number of occurences (greater than max)";
      msg += " nb_occur=";
      msg += nb_occur;
      msg += " max_occur=";
      msg += max_occurs;
      msg += " option=";
      msg += m_p->m_root_element.xpathFullName();
      msg += "/";
      msg += name();
      throw CaseOptionException(A_FUNCINFO,msg.toString(),true);
    }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String CaseOptionBase::
_xpathFullName() const
{
  return m_p->m_root_element.xpathFullName();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

