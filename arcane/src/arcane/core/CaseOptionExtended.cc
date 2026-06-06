// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CaseOptionExtended.cc                                       (C) 2000-2025 */
/*                                                                           */
/* Option of the 'Extended' data set type.                                   */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/CaseOptionExtended.h"

#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/StringBuilder.h"
#include "arcane/utils/internal/ParameterCaseOption.h"

#include "arcane/core/ICaseMng.h"
#include "arcane/core/CaseOptionError.h"
#include "arcane/core/ICaseDocumentVisitor.h"
#include "arcane/core/XmlNodeList.h"
#include "arcane/core/ICaseOptionList.h"
#include "arcane/core/MeshHandle.h"
#include "arcane/core/ICaseDocument.h"
#include "arcane/core/CaseOptionException.h"
#include "arcane/core/internal/StringVariableReplace.h"
#include "arcane/core/internal/ICaseMngInternal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Searches for the option value in the data set.
 *
 * The found value is stored in \a m_value.
 *
 * If the value is not present in the data set, it checks for a default
 * value and uses it.
 */
void CaseOptionMultiExtended::
_search(bool is_phase1)
{
  ITraceMng* tm = traceMng();
  const ParameterListWithCaseOption& params = caseMng()->_internalImpl()->parameters();
  const ParameterCaseOption pco{ params.getParameterCaseOption(caseDocumentFragment()->language()) };
  String full_xpath = String::format("{0}/{1}", rootElement().xpathFullName(), name());

  // !!! In XML, we start at 1 and not 0.
  UniqueArray<Integer> option_in_param;

  pco.indexesInParam(full_xpath, option_in_param, false);

  XmlNodeList elem_list = rootElement().children(name());
  Integer size = elem_list.size();
  bool is_optional = isOptional();

  if (size == 0 && option_in_param.empty() && is_optional) {
    return;
  }

  Integer min_occurs = minOccurs();
  Integer max_occurs = maxOccurs();

  Integer max_in_param = 0;

  // We check if the user provided too high an index for the option on the command line.
  if (!option_in_param.empty()) {
    max_in_param = option_in_param[0];
    for (Integer index : option_in_param) {
      if (index > max_in_param)
        max_in_param = index;
    }
    if (max_occurs >= 0) {
      if (max_in_param > max_occurs) {
        StringBuilder msg = "Bad number of occurences in command line (greater than max)";
        msg += " index_max_in_param=";
        msg += max_in_param;
        msg += " max_occur=";
        msg += max_occurs;
        msg += " option=";
        msg += full_xpath;
        throw CaseOptionException(A_FUNCINFO, msg.toString(), true);
      }
    }
  }

  if (max_occurs >= 0) {
    if (size > max_occurs) {
      StringBuilder msg = "Bad number of occurences (greater than max)";
      msg += " nb_occur=";
      msg += size;
      msg += " max_occur=";
      msg += max_occurs;
      msg += " option=";
      msg += full_xpath;
      throw CaseOptionException(A_FUNCINFO, msg.toString(), true);
    }
  }

  // There will always be at least min_occurs options.
  // If there are not enough options in the data set and in the command line parameters,
  // default values are added (if no default exists, it will crash).
  Integer final_size = std::max(size, std::max(min_occurs, max_in_param));

  if (is_phase1) {
    _allocate(final_size);
    m_values.resize(final_size);
  }
  else {
    // First, we get the data set options: since we cannot define an index
    // for options in the data set, they will necessarily be at the beginning and contiguous.
    // Then, if options are missing to reach min_occurs, default options are added.
    // If there is no default option, an exception will occur.
    // Finally, the user may have added options from the command line. We add them then.
    // If the user wishes to modify data set values from the command line, we
    // replace the options during reading.
    for (Integer i = 0; i < final_size; ++i) {
      String str_val;

      // Command line parameters part.
      if (option_in_param.contains(i + 1)) {
        str_val = pco.getParameterOrNull(full_xpath, i + 1, false);
      }

      // Data set part.
      else if (i < size) {
        XmlNode velem = elem_list[i];
        if (!velem.null()) {
          str_val = velem.value();
        }
      }

      // Default value.
      if (str_val.null()) {
        str_val = _defaultValue();
      }
      else {
        // In an else: Symbol replacement does not apply to default values from the .axl.
        str_val = StringVariableReplace::replaceWithCmdLineArgs(params, str_val, true);
      }

      // Now, this crash also concerns the case where there are no default values and there are
      // not enough options to reach min_occurs.
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
  // If a value is provided by the user, we do nothing.
  if (isPresent())
    return;

  // Value already initialized. In this case, we also replace the
  // current value.
  if (_isInitialized()){
    bool is_bad = _tryToConvert(def_value);
    if (is_bad){
      m_value = String();
      ARCANE_FATAL("Can not convert '{0}' to type '{1}' (option='{2}')",
                   def_value,_typeName(),xpathFullName());
    }
    m_value = def_value;
  }

  // The default value does not have an associated language.
  _setDefaultValue(def_value);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Searches for the option value in the data set.
 *
 * The found value is stored in \a m_value.
 *
 * If the value is not present in the data set, it checks for a default
 * value and uses it.
 */
void CaseOptionExtended::
_search(bool is_phase1)
{
  CaseOptionSimple::_search(is_phase1);
  if (is_phase1)
    return;
  ITraceMng* tm = traceMng();
  // If the option is not present in the data set, we take
  // the default option.
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
