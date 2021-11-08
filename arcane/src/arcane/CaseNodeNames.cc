// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CaseCodeNames.cc                                            (C) 2000-2020 */
/*                                                                           */
/* Noms des noeuds XML d'un jeu de donnée Arcane.                            */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/CaseNodeNames.h"

#include "arcane/utils/String.h"

#include "arcane/StringDictionary.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class CaseNodeNames::Impl
{
 public:
  Impl(const String& lang);
 public:
  String tr(const char* str) const
  {
    String ustr(str);
    String s = m_dict.find(ustr);
    if (s.null())
      s = ustr;
    return s;
  }
 public:
  StringDictionary m_dict;
  void _add(const char* str,const char* value)
  {
    m_dict.add(String(str),String(value));
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CaseNodeNames::Impl::
Impl(const String& lang)
{
  if (lang==String("fr")){
    _add("case","cas");
    _add("timeloop","boucle-en-temps");
    _add("title","titre");
    _add("description","description");
    _add("modules","modules");
    _add("services","services");
    _add("mesh","maillage");
    _add("meshes","meshes");
    _add("file","fichier");
    _add("partitioner","partitionneur");
    _add("userclass","userclass");
    _add("codename","codename");
    _add("codeversion","codeversion");
    _add("initialisation","initialisation");
    _add("tied-interfaces","interfaces-liees");
    _add("semi-conform","semi-conforme");
    _add("slave","esclave");
    _add("not-structured","non-structure");
    _add("functions","fonctions");
    _add("table","table");
    _add("script","script");
    _add("parameter","parametre");
    _add("value","valeur");
    _add("deltat-coef","deltat-coef");
    _add("function","fonction");
    _add("activation","activation");
    _add("interpolation","interpolation");
    _add("constant","constant-par-morceaux");
    _add("linear","lineaire");
    _add("name","nom");
    _add("real","reel");
    _add("real3","reel3");
    _add("bool","bool");
    _add("integer","entier");
    _add("time","temps");
    _add("iteration","iteration");
    _add("language","langage");
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CaseNodeNames::
CaseNodeNames(const String& lang)
: m_p(new Impl(lang))
{
  // Doit être indépendant du langage puisque sert à initialiser
  // la langue.
  lang_attribute = String("xml:lang");
  _init();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

CaseNodeNames::
~CaseNodeNames()
{
  delete m_p;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void CaseNodeNames::
_init()
{
  // NOTE: Si on ajoute ou change des noms de cette liste, il faut mettre
  // à jour la conversion correspondante dans CaseDocumentLangTranslator.

  root = m_p->tr("case");
  timeloop = m_p->tr("timeloop");
  title = m_p->tr("title");
  description = m_p->tr("description");
  modules = m_p->tr("modules");
  services = m_p->tr("services");
  mesh = m_p->tr("mesh");
  meshes = m_p->tr("meshes");
  mesh_file = m_p->tr("file");
  mesh_partitioner = m_p->tr("partitioner");
  mesh_initialisation = m_p->tr("initialisation");
  user_class = m_p->tr("userclass");
  code_name = m_p->tr("codename");
  code_version = m_p->tr("codeversion");
  code_unit = m_p->tr("codeunit");

  tied_interfaces = m_p->tr("tied-interfaces");
  tied_interfaces_semi_conform = m_p->tr("semi-conform");
  tied_interfaces_slave = m_p->tr("slave");
  tied_interfaces_not_structured = m_p->tr("not-structured");
  tied_interfaces_planar_tolerance = m_p->tr("planar-tolerance");
  
  functions = m_p->tr("functions");
  function_table = m_p->tr("table");
  function_script = m_p->tr("script");
         
  function_parameter = m_p->tr("parameter");
  function_value = m_p->tr("value");
  function_deltat_coef = m_p->tr("deltat-coef");
  function_ref = m_p->tr("function");
  function_activation_ref = m_p->tr("activation");
  function_interpolation = m_p->tr("interpolation");
  function_constant = m_p->tr("constant");
  function_linear = m_p->tr("linear");

  name_attribute = m_p->tr("name");
  real_type = m_p->tr("real");
  real3_type = m_p->tr("real3");
  bool_type = m_p->tr("bool");
  integer_type = m_p->tr("integer");
  string_type = m_p->tr("string");
         
  time_type = m_p->tr("time");
  iteration_type = m_p->tr("iteration");

  script_language_ref = m_p->tr("language");
  script_function_ref = m_p->tr("function");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
