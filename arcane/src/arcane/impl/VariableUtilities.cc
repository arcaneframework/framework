// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VariableUtilities.cc                                        (C) 2000-2015 */
/*                                                                           */
/* Fonctions utilitaires sur les variables.                                  */
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
dumpAllVariableDependencies(ostream& ostr,bool is_recursive)
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
dumpDependencies(IVariable* var,ostream& ostr,bool is_recursive)
{
  // Ensemble des variables déjà traitées pour éviter les récursions infinies
  _dumpDependencies(var,ostr,is_recursive);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableUtilities::
_dumpDependencies(IVariable* var,ostream& ostr,bool is_recursive)
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
_dumpDependencies(VariableDependInfo& vdi,ostream& ostr,bool is_recursive,
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
 * L'algorithme utilisé est le suivant:
 * - chaque rang met dans un SerializeBuffer la liste de ses variables.
 * - on fait un allGather de ce SerializerBuffer.
 * - chaque rang lit le contenu du allGather et compte le nombre
 * d'occurence de chaque variable.
 * - les variables dont le nombre d'occurence est différent de \a pm->commSize()
 * sont retirée.
 * - les variables restantes sont triéés par ordre alphabétique.
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

  // Créé un buffer pour sérialiser les noms des variables dont on dispose
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

  // Récupère les infos des autres PE.
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

  // Parcours la liste des variables et range dans \a common_vars
  // celles qui sont disponibles sur tous les rangs de \a pm
  std::map<String,IVariable*> common_vars;
  {
    std::map<String,Int32>::const_iterator end_var = var_occurences.end();
    for( Integer i=0; i<nb_var; ++i ){
      IVariable* var = vars_to_check[i];
      std::map<String,Int32>::const_iterator i_var = var_occurences.find(var->fullName());
      if (i_var==end_var)
        // Ne devrait pas arriver
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

  // Créé la liste finale en itérant sur la map \a common_vars
  // et range les valeurs dans \a sorted_common_vars. Comme la map
  // est triée par ordre alphabétique, ce sera aussi le cas de
  // \a sorted_common_vars;
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
