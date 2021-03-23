// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CaseNodeNames.h                                             (C) 2000-2020 */
/*                                                                           */
/* Noms des noeuds XML d'un jeu de donnée Arcane.                            */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CASENODENAMES_H
#define ARCANE_CASENODENAMES_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/String.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 *
 * \brief Noms des noeuds XML d'un jeu de données Arcane.
 */
class ARCANE_CORE_EXPORT CaseNodeNames
{
 public:

  class Impl;

 public:

  //! Crée une instance pour le langage \a lang
  CaseNodeNames(const String& lang);
  ~CaseNodeNames();

 public:

  String root;
  String lang_attribute;
  String timeloop;
  String title;
  String description;
  String modules;
  String services;
  String mesh;
  String meshes;
  String mesh_file;
  String mesh_partitioner;
  String mesh_initialisation;
  String user_class;
  String code_name;
  String code_version;
  String code_unit;
  String tied_interfaces;
  String tied_interfaces_semi_conform;
  String tied_interfaces_slave;
  String tied_interfaces_not_structured;
  String tied_interfaces_planar_tolerance;

  String functions;
  String function_table;
  String function_script;
  String name_attribute;
  String function_parameter;
  String function_value;
  String function_deltat_coef;
  String function_interpolation;
  String function_constant;
  String function_linear;
  String function_ref;
  String function_activation_ref;

  String time_type;
  String iteration_type;

  String real_type;
  String real3_type;
  String bool_type;
  String integer_type;
  String string_type;

  String script_language_ref;
  String script_function_ref;

 private:

  Impl* m_p;

 private:

  void _init();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

