﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ParameterListPropertyReader.h                               (C) 2000-2021 */
/*                                                                           */
/* Lecture de propriétés au format JSON.                                     */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_PARAMETERLISTPROPERTYREADER_H
#define ARCANE_UTILS_PARAMETERLISTPROPERTYREADER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*
 * NOTE: Les classes de ce fichier sont en cours de mise au point.
 * NOTE: L'API peut changer à tout moment. Ne pas utiliser en dehors de Arcane.
 */

#include "arcane/utils/ParameterList.h"
#include "arcane/utils/Property.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::properties
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
//! \internal
template<typename T>
class ParameterListPropertyVisitor
: public properties::PropertyVisitor<T>
{
 public:
  ParameterListPropertyVisitor(const ParameterList& args,T& instance)
  : m_args(args), m_instance(instance){}
 private:
  const ParameterList& m_args;
  T& m_instance;
 public:
  void visit(const properties::PropertySettingBase<T>& s) override
  {
    String param_name = s.setting()->commandLineArgument();
    if (param_name.null())
      return;
    String param_value = m_args.getParameterOrNull(param_name);
    //std::cout << "GET_PARAM name='" << param_name << "' value='" << param_value << "'\n";
    if (param_value.null())
      return;
    s.setFromString(param_value,m_instance);
    //std::cout << "SET_PROP from command line:";
    //s.print(std::cout,m_instance);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Remplit les valeurs de \a instance à partir des paramètres \a args.
 */
template<typename T> inline void
readFromParameterList(const ParameterList& args,T& instance)
{
  ParameterListPropertyVisitor reader(args,instance);
  T :: applyPropertyVisitor(reader);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::properties

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
