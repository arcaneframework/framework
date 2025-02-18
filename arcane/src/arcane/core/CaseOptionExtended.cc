﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CaseOptionExtended.cc                                       (C) 2000-2023 */
/*                                                                           */
/* Option du jeu de données de type 'Extended'.                              */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/CaseOptionExtended.h"

#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/ApplicationInfo.h"
#include "arcane/utils/CommandLineArguments.h"
#include "arcane/utils/ParameterList.h"
#include "arcane/utils/ParameterCaseOption.h"

#include "arcane/core/IApplication.h"
#include "arcane/core/ICaseMng.h"
#include "arcane/core/CaseOptionError.h"
#include "arcane/core/ICaseDocumentVisitor.h"
#include "arcane/core/XmlNodeList.h"
#include "arcane/core/ICaseOptionList.h"
#include "arcane/core/MeshHandle.h"
#include "arcane/core/ICaseDocument.h"
#include "arcane/core/internal/StringVariableReplace.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Cherche la valeur de l'option dans le jeu de données.
 *
 * La valeur trouvée est stockée dans \a m_value.
 *
 * Si la valeur n'est pas présente dans le jeu de données, regarde s'il
 * existe une valeur par défaut et utilise cette dernière.
 */
void CaseOptionMultiExtended::
_search(bool is_phase1)
{
  ITraceMng* tm = traceMng();
  const ParameterList& params = caseMng()->application()->applicationInfo().commandLineArguments().parameters();
  const ParameterCaseOption pco{ params.getParameterCaseOption(caseDocumentFragment()->language()) };

  // !!! En XML, on commence par 1 et non 0.
  UniqueArray<Integer> option_in_param;

  pco.indexesInParam(String::format("{0}/{1}", rootElement().xpathFullName(), name()), option_in_param, false);

  XmlNodeList elem_list = rootElement().children(name());
  Integer size = elem_list.size();
  bool is_optional = isOptional();

  if (size == 0 && option_in_param.empty() && is_optional) {
    return;
  }

  Integer min_occurs = minOccurs();
  Integer max_occurs = maxOccurs();

  Integer max_in_param = 0;

  if (!option_in_param.empty()) {
    max_in_param = option_in_param[0];
    for (Integer index : option_in_param) {
      if (index > max_in_param)
        max_in_param = index;
    }
    if (max_occurs >= 0) {
      if (max_in_param > max_occurs) {
        ARCANE_FATAL("Max in param > max_occurs");
      }
    }
  }

  if (max_occurs >= 0) {
    if (size > max_occurs) {
      ARCANE_FATAL("Nb in XmlNodeList > max_occurs");
    }
  }

  Integer final_size = std::max(size, std::max(min_occurs, max_in_param));

  if (is_phase1) {
    _allocate(final_size);
    m_values.resize(final_size);
  }
  else{
    //cerr << "** MULTI SEARCH " << size << endl;
    for (Integer i = 0; i < final_size; ++i) {
      String str_val;

      if (option_in_param.contains(i + 1)) {
        str_val = pco.getParameterOrNull(String::format("{0}/{1}", rootElement().xpathFullName(), name()), i + 1, false);
      }
      else if (i < size) {
        XmlNode velem = elem_list[i];
        if (!velem.null()) {
          str_val = velem.value();
        }
      }

      if (str_val.null()) {
        str_val = _defaultValue();
      }
      else {
        // Dans un else : Le remplacement de symboles ne s'applique pas pour les valeurs par défault du .axl.
        str_val = StringVariableReplace::replaceWithCmdLineArgs(params, str_val, true);
      }

      if (str_val.null()) {
        CaseOptionError::addOptionNotFoundError(caseDocumentFragment(),A_FUNCINFO,
                                                name(),rootElement());
        continue;
      }
      tm->info(5) << "TryConvert opt=" << _xpathFullName() << " i=" << i
                  << " mesh_name=" << parentOptionList()->meshHandle().meshName()
                  << " value=" << str_val;
      bool is_bad = _tryToConvert(str_val,i);
      if (is_bad){
        m_values[i] = String();
        CaseOptionError::addInvalidTypeError(caseDocumentFragment(),A_FUNCINFO,
                                             name(),rootElement(),str_val,_typeName());
        continue;
      }
      m_values[i] = str_val;
      //ptr_value[i] = val;
      //cerr << "** FOUND " << val << endl;
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseOptionMultiExtended::
print(const String& lang,std::ostream& o) const
{
  ARCANE_UNUSED(lang);
  for( Integer i=0, s=_nbElem(); i<s; ++i )
    o << m_values[i] << " ";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseOptionMultiExtended::
visit(ICaseDocumentVisitor* visitor) const
{
  visitor->applyVisitor(this);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseOptionExtended::
setDefaultValue(const String& def_value)
{
  // Si on a une valeur donnée par l'utilisateur, on ne fait rien.
  if (isPresent())
    return;

  // Valeur déjà initialisée. Dans ce cas on remplace aussi la valeur
  // actuelle.
  if (_isInitialized()){
    bool is_bad = _tryToConvert(def_value);
    if (is_bad){
      m_value = String();
      ARCANE_FATAL("Can not convert '{0}' to type '{1}' (option='{2}')",
                   def_value,_typeName(),xpathFullName());
    }
    m_value = def_value;
  }

  // La valeur par défaut n'a pas de langue associée.
  _setDefaultValue(def_value);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Cherche la valeur de l'option dans le jeu de donnée.
 *
 * La valeur trouvée est stockée dans \a m_value.
 *
 * Si la valeur n'est pas présente dans le jeu de donnée, regarde s'il
 * existe une valeur par défaut et utilise cette dernière.
 */
void CaseOptionExtended::
_search(bool is_phase1)
{
  CaseOptionSimple::_search(is_phase1);
  if (is_phase1)
    return;
  ITraceMng* tm = traceMng();
  // Si l'option n'est pas présente dans le jeu de donnée, on prend
  // l'option par défaut.
  String str_val = (_element().null()) ? _defaultValue() : _element().value();
  bool has_valid_value = true;
  if (str_val.null()) {
    m_value = String();
    if (!isOptional()){
      CaseOptionError::addOptionNotFoundError(caseDocumentFragment(),A_FUNCINFO,
                                              name(),rootElement());
      return;
    }
    else
      has_valid_value = false;
  }
  _setHasValidValue(has_valid_value);
  if (has_valid_value){
    tm->info(5) << "TryConvert opt=" << xpathFullName()
                << " mesh_name=" << parentOptionList()->meshHandle().meshName()
                << " value=" << str_val;
    bool is_bad = _tryToConvert(str_val);
    if (is_bad){
      m_value = String();
      CaseOptionError::addInvalidTypeError(caseDocumentFragment(),A_FUNCINFO,
                                           name(),rootElement(),str_val,_typeName());
      return;
    }
    m_value = str_val;
  }
  _setIsInitialized();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseOptionExtended::
print(const String& lang,std::ostream& o) const
{
  ARCANE_UNUSED(lang);
  _checkIsInitialized();
  if (hasValidValue())
    o << m_value;
  else
    o << "undefined";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseOptionExtended::
visit(ICaseDocumentVisitor* visitor) const
{
  visitor->applyVisitor(this);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
