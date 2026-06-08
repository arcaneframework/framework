// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ParameterListPropertyReader.h                               (C) 2000-2025 */
/*                                                                           */
/* Reading properties in JSON format.                                        */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_INTERNAL_PARAMETERLISTPROPERTYREADER_H
#define ARCANE_UTILS_INTERNAL_PARAMETERLISTPROPERTYREADER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*
 * NOTE: The classes in this file are under development.
 * NOTE: The API may change at any time. Do not use outside of Arcane.
 */

#include "arccore/common/ParameterList.h"
#include "arccore/common/internal/Property.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::properties
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
//! \internal
template <typename T, typename PropertyType = T>
class ParameterListPropertyVisitor
: public properties::PropertyVisitor<T>
{
 public:

  ParameterListPropertyVisitor(const ParameterList& args, T& instance)
  : m_args(args)
  , m_instance(instance)
  {}

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
    s.setFromString(param_value, m_instance);
    //std::cout << "SET_PROP from command line:";
    //s.print(std::cout,m_instance);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Fills the values of \a instance from the parameters \a args.
 */
template <typename T, typename PropertyType = T> inline void
readFromParameterList(const ParameterList& args, T& instance)
{
  ParameterListPropertyVisitor<T, PropertyType> reader(args, instance);
  PropertyType ::applyPropertyVisitor(reader);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::properties

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
