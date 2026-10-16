// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ArcaNetReader.cc                                            (C) 2000-2026 */
/*                                                                           */
/* Reader for ArcaNet files.                                                 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/common/ArcaNetReader.h"

#include "arccore/base/FatalErrorException.h"
#include "arccore/base/PlatformUtils.h"
#include "arccore/common/JSONReader.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ArcaNetReader::
ArcaNetReader(const JSONValue& root)
: m_build_path(!platform::getEnvironmentVariable("ARCCORE_ARCANET_PATH").null())
, m_root(root)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcaNetReader::
init(const String& case_and_variations_values, const String& prog_name)
{
  m_prog_part_name = prog_name;
  if (!case_and_variations_values.empty()) {
    // Tous les éléments sont séparés par des ":".
    UniqueArray<String> split_case;
    case_and_variations_values.split(split_case, ':');

    for (const String& elem : split_case) {
      // Si on "=" est présent, c'est une variation à résoudre.
      if (elem.contains("=")) {
        UniqueArray<String> split_variation;
        elem.split(split_variation, '=');
        if (split_variation.size() != 2) {
          ARCCORE_FATAL("Bad element in Case option.");
        }
        m_variations_key_resolved.add(split_variation[0]);
        m_variations_value_resolved.add(split_variation[1]);
        // std::cout << "Add key=value : " << split_elem[0] << " = " << split_elem[1] << std::endl;
      }
      else {
        if (!m_case_name.empty()) {
          ARCCORE_FATAL("You must choose a variation for '{0}'.", elem);
        }
        if (elem.contains("~")) {
          UniqueArray<String> split_case_name_with_prog_subname;
          elem.split(split_case_name_with_prog_subname, '~');
          if (split_case_name_with_prog_subname.size() != 2) {
            ARCCORE_FATAL("Bad element in Case option.");
          }
          m_case_name = split_case_name_with_prog_subname[0];
          m_prog_custom_part_name = m_prog_part_name + "~" + split_case_name_with_prog_subname[1];
        }
        else {
          m_case_name = elem;
        }
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcaNetReader::
read()
{
  if (m_prog_part_name.empty()) {
    ARCCORE_FATAL("You must call 'init()' method before.");
  }

  // "general" part
  {
    const JSONValue cases = m_root.child("general");
    if (!cases.isNull()) {
      // std::cout << "General part" << std::endl;
      m_explored.add("general");
      _readConfigPart(cases, "general");
    }
  }
  if (!m_case_name.empty()) {
    const JSONValue cases = m_root.child("cases").child(m_case_name);
    if (cases.null()) {
      ARCCORE_FATAL("Case '{0}' not found.", m_case_name);
    }
    // std::cout << "Case part : " << case_clean << std::endl;
    m_explored.add(m_case_name);
    _readConfigPart(cases, m_case_name);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

JSONValue ArcaNetReader::
extractFilePart(const JSONValue& config_part)
{
  return config_part.child("_");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

JSONValue ArcaNetReader::
extractProgPart(const JSONValue& config_part)
{
  JSONValue arcane_part;
  if (!m_prog_custom_part_name.empty()) {
    arcane_part = config_part.child(m_prog_custom_part_name);
  }
  if (arcane_part.null()) {
    arcane_part = config_part.child(m_prog_part_name);
  }
  return arcane_part;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

JSONValue ArcaNetReader::
extractPart(const String& part_name, const JSONValue& config_part)
{
  return config_part.child(part_name);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcaNetReader::
_readDependPart(const JSONValue& depend_part)
{
  if (depend_part.null())
    return;

  const JSONValueList array = depend_part.valueAsArray();
  for (const JSONValue elem : array) {
    JSONValue dependence_node;
    String dependence_name = elem.valueAsStringView();
    {
      // S'il y a un "=", alors on est sur une variation résolue.
      if (dependence_name.contains("=")) {
        UniqueArray<String> split_variation;
        dependence_name.split(split_variation, '=');
        if (split_variation.size() != 2) {
          ARCCORE_FATAL("Element '{0}' is invalid.", dependence_name);
        }

        const JSONValue variation = m_root.child("variations").child(split_variation[0]).child(split_variation[1]);
        if (variation.null()) {
          ARCCORE_FATAL("Element '//variations/{0}/{1}' not found.", split_variation[0], split_variation[1]);
        }
        if (m_explored.contains(dependence_name)) {
          ARCCORE_FATAL("Element '//variations/{0}/{1}' already explored.", split_variation[0], split_variation[1]);
        }
        m_explored.add(dependence_name);
      }
      else {
        dependence_node = m_root.child("commons").child(dependence_name);
        if (!dependence_node.null()) {
          if (m_explored.contains(dependence_name)) {
            ARCCORE_FATAL("Element '//commons/{0}' already explored.", dependence_name);
          }
          m_explored.add(dependence_name);
        }
        // Si la dépendance n'est pas dans "commons", soit c'est une variation
        // non résolue, soit c'est une dépendance inconnue.
        else {
          bool is_excluded_variants = true;
          String cleaned_dependence_name;
          UniqueArray<String> split_dependence_name;
          // On stocke les variations exclues, s'il y en a.
          if (dependence_name.contains("!")) {
            is_excluded_variants = true;
            dependence_name.split(split_dependence_name, '!');
            cleaned_dependence_name = split_dependence_name[0];
          }
          else if (dependence_name.contains("+")) {
            is_excluded_variants = false;
            dependence_name.split(split_dependence_name, '+');
            cleaned_dependence_name = split_dependence_name[0];
          }
          else {
            cleaned_dependence_name = dependence_name;
          }
          auto pos_elem = m_variations_key_resolved.span().findFirst(cleaned_dependence_name);

          if (!pos_elem.has_value()) {
            const JSONValue error_ = m_root.child("variations").child(cleaned_dependence_name);
            if (error_.null()) {
              ARCCORE_FATAL("Element '//commons/{0}' not found.", cleaned_dependence_name);
            }
            ARCCORE_FATAL("Element '{0}' not found in 'Case' cmd line option.", cleaned_dependence_name);
          }

          String value = m_variations_value_resolved[pos_elem.value()];
          if (!split_dependence_name.empty()) {
            ArrayView excluded_or_included_values(split_dependence_name.subView(1, split_dependence_name.size() - 1));
            if (is_excluded_variants && excluded_or_included_values.contains(value)) {
              ARCCORE_FATAL("Element '//variations/{0}/{1}' is excluded.", cleaned_dependence_name, value);
            }
            if (!is_excluded_variants && !excluded_or_included_values.contains(value)) {
              ARCCORE_FATAL("Element '//variations/{0}/{1}' is not included.", cleaned_dependence_name, value);
            }
          }

          // On résout la variation.
          dependence_name = cleaned_dependence_name + "=" + value;

          dependence_node = m_root.child("variations").child(cleaned_dependence_name).child(value);
          if (dependence_node.null()) {
            ARCCORE_FATAL("Element '//variations/{0}/{1}' not found.", cleaned_dependence_name, value);
          }
          if (m_explored.contains(dependence_name)) {
            ARCCORE_FATAL("Element '//variations/{0}/{1}' already explored.", cleaned_dependence_name, value);
          }
          m_explored.add(dependence_name);
        }
      }
    }
    _readConfigPart(dependence_node, dependence_name);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcaNetReader::
_readConfigPart(const JSONValue& config_part, const String& node_name)
{
  if (m_build_path) {
    _addElemInPath("{");
    m_level_path++;
    _readFileBeforePart(config_part);
    _addElemInPath(node_name);
    m_values.add(config_part);
    _readArcanePart(config_part);
    _readFileAfterPart(config_part);
    m_level_path--;
    _addElemInPath("}");
  }
  else {
    _readFileBeforePart(config_part);
    m_values.add(config_part);
    _readFileAfterPart(config_part);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcaNetReader::
_addElemInPath(const String& elem)
{
  m_path += "\n";
  for (Integer i = 0; i < m_level_path; ++i)
    m_path += "  ";
  m_path += elem;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcaNetReader::
_readFileBeforePart(const JSONValue& config_part)
{
  const JSONValue depend_before_var = config_part.child("_").child("depend_b");
  _readDependPart(depend_before_var);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcaNetReader::
_readFileAfterPart(const JSONValue& config_part)
{
  const JSONValue depend_after_var = config_part.child("_").child("depend_a");
  _readDependPart(depend_after_var);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcaNetReader::
_readArcanePart(const JSONValue& config_part)
{
  if (!m_prog_custom_part_name.empty() && !config_part.child(m_prog_custom_part_name).null()) {
    m_path += "(";
    m_path += m_prog_custom_part_name;
    m_path += ")";
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
