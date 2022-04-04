﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VerifierService.cc                                          (C) 2000-2020 */
/*                                                                           */
/* Classe de base du service de vérification des données.                    */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/utils/List.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/OStringStream.h"

#include "arcane/VerifierService.h"
#include "arcane/ServiceBuildInfo.h"
#include "arcane/ISubDomain.h"
#include "arcane/IVariable.h"
#include "arcane/IParallelMng.h"
#include "arcane/IVariableMng.h"
#include "arcane/ArcaneException.h"
#include "arcane/CommonVariables.h"
#include "arcane/SerializeBuffer.h"
#include "arcane/VariableCollection.h"
#include "arcane/IMesh.h"

#include <map>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VerifierService::
VerifierService(const ServiceBuildInfo& sbi)
: AbstractService(sbi)
, m_sub_domain(sbi.subDomain())
, m_service_info(sbi.serviceInfo())
, m_file_name()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VerifierService::
~VerifierService()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IBase* VerifierService::
serviceParent() const
{
  return m_sub_domain;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VerifierService::
_getVariables(VariableCollection variables,bool parallel_sequential)
{
  ISubDomain* sd = subDomain();
  IParallelMng* pm_sd = sd->parallelMng();
  IVariableMng* vm = sd->variableMng();
  VariableCollection used_vars = vm->usedVariables();

  for( VariableCollection::Enumerator i(used_vars); ++i; ){
    IVariable* variable = *i;
    if (parallel_sequential){
      if (variable->property() & IVariable::PSubDomainDepend)
        continue;
      // Ne compare pas les variables tableaux lors des comparaisons parallèles/séquentielles
      // car en général ces variables sont dépendantes du découpage.
      if (variable->itemKind()==IK_Unknown)
        continue;
      // Ne compare pas les variables dont le maillage n'utilise pas le même
      // parallelMng() que le sous-domaine. En effet dans ce cas il n'est pas possible
      // de faire une vue séquentielle du maillage.
      MeshHandle mesh_handle = variable->meshHandle();
      if (mesh_handle.hasMesh()){
        IMesh* mesh = mesh_handle.mesh();
        if (mesh->parallelMng()!=pm_sd)
          continue;
      }
    }
    if (variable->property() & IVariable::PExecutionDepend)
      continue;
    variables.add(variable);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename ReaderType> void VerifierService::
_doVerif2(ReaderType reader,const VariableList& variables,bool compare_ghost)
{
  ISubDomain* sd = subDomain();
  IParallelMng* pm = sd->parallelMng();
  ITraceMng* trace = sd->traceMng();

  OStringStream not_compared_str;
  Integer nb_not_compared =0;
  Integer nb_compared = 0;

  typedef std::map<String,DiffInfo> MapDiffInfos;
  typedef std::map<String,DiffInfo>::value_type MapDiffInfosValues;
  MapDiffInfos diff_infos;

  {
    for( VariableList::Enumerator i(variables); ++i; ){
      IVariable* variable = *i;
      Integer nb_diff = 0;
      String var_name(variable->name());
      try
      {
        nb_diff = variable->checkIfSame(reader,10,compare_ghost);
        ++nb_compared;
      }
      catch(const ReaderWriterException& rw)
      {
        OStringStream ostr;
        rw.explain(ostr());
        trace->pinfo() << "Impossible to compare the variable '" << var_name << "'\n"
                     << "(Exception: " << ostr.str() << ")";
        not_compared_str() << ' ' << var_name;
        ++nb_not_compared;
      }
      diff_infos.insert(MapDiffInfosValues(var_name,DiffInfo(var_name,nb_diff)));
    }
  }
  if (nb_not_compared!=0){
    trace->warning() << "Impossible to compare " << nb_not_compared << " variable(s): "
                   << not_compared_str.str();
  }
  if (nb_compared==0){
    trace->pfatal() << "No variable has been compared";
  }
  Int32 sid = pm->commRank();
  Int32 nb_sub_domain = pm->commSize();
  bool is_master = sid==0;

  if (is_master){

    for( Integer i=0; i<nb_sub_domain; ++i ){
      if (i==sid)
        continue;
      SerializeBuffer sbuf;
      pm->recvSerializer(&sbuf,i);
      sbuf.setMode(ISerializer::ModeGet);

      {
        Int64 n = sbuf.getInt64();
        String var_name;
        for( Integer z=0; z<n; ++z ){
          Int64 nb_var_diff = sbuf.getInt64();
          sbuf.get(var_name);
          String uvar_name(var_name);
          trace->debug() << "RECEIVE: "
                       << " varname=" << var_name
                       << " nbdiff=" << nb_var_diff;
          MapDiffInfos::iterator i_map = diff_infos.find(uvar_name);
          if (i_map==diff_infos.end()){
            diff_infos.insert(MapDiffInfosValues(uvar_name,DiffInfo(uvar_name,nb_var_diff)));
          }
          else
            i_map->second.m_nb_diff += nb_var_diff;
        }
      }
    }
  }
  else{
    SerializeBuffer sbuf;
    Int64 nb_diff = diff_infos.size();
    sbuf.setMode(ISerializer::ModeReserve);
    sbuf.reserve(DT_Int64,1); // pour la taille
    for( MapDiffInfos::const_iterator i_map = diff_infos.begin(); i_map!=diff_infos.end(); ++i_map ){
      const DiffInfo& diff_info = i_map->second;
      sbuf.reserve(DT_Int64,1);
      sbuf.reserve(diff_info.m_variable_name);
    }
    sbuf.allocateBuffer();
    sbuf.setMode(ISerializer::ModePut);
    sbuf.putInt64(nb_diff);
    for( MapDiffInfos::const_iterator i_map = diff_infos.begin(); i_map!=diff_infos.end(); ++i_map ){
      const DiffInfo& diff_info = i_map->second;
      Int64 n = diff_info.m_nb_diff;
      sbuf.putInt64(n);
      sbuf.put(diff_info.m_variable_name);
    }
    pm->sendSerializer(&sbuf,0);
  }

  pm->barrier();

  if (is_master){
    Int64 total_nb_diff = 0;
    for( MapDiffInfos::const_iterator i_map = diff_infos.begin(); i_map!=diff_infos.end(); ++i_map )
      total_nb_diff += i_map->second.m_nb_diff;

    if (!m_result_file_name.empty()){
      const CommonVariables& vc = subDomain()->commonVariables();
      std::ofstream result_file(m_result_file_name.localstr());
      result_file << "<?xml version='1.0'?>\n";
      result_file << "<compare-results"
                  << " version='1.0'"
                  << " total-nb-diff='" << total_nb_diff << "'"
                  << " global-iteration='" << vc.globalIteration() << "'"
                  << " global-time='" << vc.globalTime() << "'"
                  << ">\n";
      for( MapDiffInfos::const_iterator i_map = diff_infos.begin(); i_map!=diff_infos.end(); ++i_map ){
        const DiffInfo& diff_info = i_map->second;
        result_file << " <variable>\n"
                    << "  <name>" << diff_info.m_variable_name << "</name>\n"
                    << "  <nb-diff>" << diff_info.m_nb_diff << "</nb-diff>\n"
                    << " </variable>\n";
      }
      result_file << "</compare-results>\n";
    }
    if (total_nb_diff!=0){
      trace->error() << "Some differences exist (N=" << total_nb_diff << ") with the reference file";
    }
    else{
      trace->info() << "No difference with the reference !";
    }

  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VerifierService::
_doVerif(IDataReader* reader,const VariableList& variables,bool compare_ghost)
{
  _doVerif2(reader,variables,compare_ghost);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
