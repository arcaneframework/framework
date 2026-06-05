// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VariableUtilities.cc                                        (C) 2000-2015 */
/*                                                                           */
/* Utility functions for variables.                                          */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Array.h"
#include "arcane/utils/Iostream.h"
#include "arcane/utils/String.h"

#include "arcane/impl/VariableUtilities.h"

#include "arcane/IVariable.h"
#include "arcane/IParallelMng.h"
#include "arcane/IVariableMng.h"
#include "arcane/VariableDependInfo.h"
#include "arcane/VariableCollection.h"
#include "arcane/SerializeBuffer.h"

#include <map>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VariableUtilities::
VariableUtilities(IVariableMng* vm)
: TraceAccessor(vm->traceMng())
, m_variable_mng(vm)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VariableUtilities::
~VariableUtilities()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableUtilities::
dumpAllVariableDependencies(std::ostream& ostr,bool is_recursive)
{
  VariableCollection used_variables = m_variable_mng->usedVariables();
  for( VariableCollection::Enumerator ivar(used_variables); ++ivar; ){
    IVariable* var = *ivar;
    dumpDependencies(var,ostr,is_recursive);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableUtilities::
dumpDependencies(IVariable* var,std::ostream& ostr,bool is_recursive)
{
  // Set of variables already processed to prevent infinite recursion
  _dumpDependencies(var,ostr,is_recursive);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableUtilities::
_dumpDependencies(IVariable* var,std::ostream& ostr,bool is_recursive)
{
  std::set<IVariable*> done_vars;
  done_vars.insert(var);

  UniqueArray<VariableDependInfo> depends;
  var->dependInfos(depends);
  Integer nb_depend = depends.size();
  if (nb_depend==0)
    return;

  ostr << var->fullName()
       << " time=" << var->modifiedTime()
       << " nb_depend=" << nb_depend
       << '\n';

  if (nb_depend!=0){
    ostr << "{\n";
    for( Integer i=0; i<nb_depend; ++i ){
      _dumpDependencies(depends[i],ostr,is_recursive,done_vars,2);
    }
    ostr << "}\n";
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableUtilities::
_dumpDependencies(VariableDependInfo& vdi,std::ostream& ostr,bool is_recursive,
                  std::set<IVariable*>& done_vars,Integer indent_level)
{
  IVariable* var = vdi.variable();
  bool no_cycle = done_vars.find(var)==done_vars.end();
  bool do_depend = no_cycle;
  if (!is_recursive)
    do_depend = false;
  done_vars.insert(var);

  std::string indent_str;
  for( Integer i=0; i<indent_level; ++i )
    indent_str.push_back(' ');

  UniqueArray<VariableDependInfo> depends;
  var->dependInfos(depends);
  Integer nb_depend = depends.size();

  ostr << indent_str
       << var->fullName()
       << " time=" << var->modifiedTime()
       << " nb_depend=" << nb_depend
       << " trace_info=" << vdi.traceInfo();
  if (!no_cycle)
    ostr << " (cycle)";
  ostr << '\n';

  if (do_depend && nb_depend!=0){
    ostr << indent_str << "{\n";
    for( Integer i=0; i<nb_depend; ++i ){
      _dumpDependencies(depends[i],ostr,true,done_vars,indent_level+2);
    }
    ostr << indent_str << "}\n";
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * The algorithm used is as follows:
 * - each rank puts the list of its variables into a SerializeBuffer.
 * - an allGather of this SerializerBuffer is performed.
 * - each rank reads the content of the allGather and counts the number
 * of occurrences of each variable.
 * - variables whose occurrence count is different from \a pm->commSize()
 * are removed.
 * - the remaining variables are sorted alphabetically.
 */
VariableCollection VariableUtilities::
filterCommonVariables(IParallelMng* pm,const VariableCollection input_variables,
                      bool dump_not_common)
{
  UniqueArray<IVariable*> vars_to_check;
  for( VariableCollection::Enumerator i(input_variables); ++i; )
    vars_to_check.add(*i);

  Integer nb_var = vars_to_check.size();
  info(4) << "CHECK: nb_variable_to_compare=" << nb_var;

  // Create a buffer to serialize the names of the variables we have
  SerializeBuffer send_buf;
  send_buf.setMode(ISerializer::ModeReserve);
  send_buf.reserveInteger(1);
  for( Integer i=0; i<nb_var; ++i ){
    send_buf.reserve(vars_to_check[i]->fullName());
  }

  send_buf.allocateBuffer();
  send_buf.setMode(ISerializer::ModePut);
  send_buf.putInteger(nb_var);
  for( Integer i=0; i<nb_var; ++i ){
    send_buf.put(vars_to_check[i]->fullName());
  }

  // Retrieve info from other PEs.
  SerializeBuffer recv_buf;
  pm->allGather(&send_buf,&recv_buf);

  std::map<String,Int32> var_occurences;

  Int32 nb_rank = pm->commSize();
  recv_buf.setMode(ISerializer::ModeGet);
  for( Integer i=0; i<nb_rank; ++i ){
    Integer nb_var_rank = recv_buf.getInteger();
    info(5) << "String recv_nb_var_rank rank=" << i << " n=" << nb_var_rank;
    for( Integer z=0; z<nb_var_rank; ++z ){
      String x;
      recv_buf.get(x);
      std::map<String,Int32>::iterator vo = var_occurences.find(x);
      if (vo==var_occurences.end())
        var_occurences.insert(std::make_pair(x,1));
      else
        vo->second = vo->second + 1;
      //info() << "String rank=" << i << " z=" << z << " name=" << x;
    }
  }

  // Iterate through the list of variables and store in \a common_vars
  // those that are available on all ranks of \a pm
  std::map<String,IVariable*> common_vars;
  {
    std::map<String,Int32>::const_iterator end_var = var_occurences.end();
    for( Integer i=0; i<nb_var; ++i ){
      IVariable* var = vars_to_check[i];
      std::map<String,Int32>::const_iterator i_var = var_occurences.find(var->fullName());
      if (i_var==end_var)
        // Should not happen
        continue;
      if (i_var->second!=nb_rank){
        if (dump_not_common)
          info() << "ERROR: can not compare variable '" << var->fullName()
                 << "' because it is not defined on all replica nb_define=" << i_var->second;
        continue;
      }
      common_vars.insert(std::make_pair(var->fullName(),var));
    }
  }

  // Create the final list by iterating over the map \a common_vars
  // and storing the values in \a sorted_common_vars. Since the map
  // is sorted alphabetically, \a sorted_common_vars will also be;
  VariableList sorted_common_vars;
  {
    std::map<String,IVariable*>::const_iterator end_var = common_vars.end();
    std::map<String,IVariable*>::const_iterator i_var = common_vars.begin();
    for( ; i_var!=end_var; ++i_var ){
      sorted_common_vars.add(i_var->second);
    }
  }

  return sorted_common_vars;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
