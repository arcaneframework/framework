// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VariableUtilities.h                                         (C) 2000-2015 */
/*                                                                           */
/* Fonctions utilitaires sur les variables.                                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IMPL_VARIABLEUTILITIES_H
#define ARCANE_IMPL_VARIABLEUTILITIES_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"
#include "arcane/IVariableUtilities.h"

#include <set>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IVariableMng;
class VariableDependInfo;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Fonctions utilitaires sur les variables.
 */
class VariableUtilities
: public TraceAccessor
, public IVariableUtilities
{
 public:

  VariableUtilities(IVariableMng* vm);
  virtual ~VariableUtilities();

 public:

  virtual IVariableMng* variableMng() const { return m_variable_mng; }
  virtual void dumpDependencies(IVariable* var,std::ostream& ostr,bool is_recursive);
  virtual void dumpAllVariableDependencies(std::ostream& ostr,bool is_recursive);
  virtual VariableCollection filterCommonVariables(IParallelMng* pm,
                                                   const VariableCollection input_variables,
                                                   bool dump_not_common);

 private:

  IVariableMng* m_variable_mng;

  void _dumpDependencies(IVariable* var,std::ostream& ostr,bool is_recursive);
  void _dumpDependencies(VariableDependInfo& vdi,std::ostream& ostr,bool is_recursive,
                         std::set<IVariable*>& done_vars,Integer indent_level);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

