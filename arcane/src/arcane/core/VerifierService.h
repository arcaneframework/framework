﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VerifierService.h                                           (C) 2000-2022 */
/*                                                                           */
/* Classe de base du service de vérification des variables.                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_VERIFIERSERVICE_H
#define ARCANE_VERIFIERSERVICE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/String.h"

#include "arcane/ArcaneTypes.h"
#include "arcane/AbstractService.h"
#include "arcane/IVerifierService.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ServiceBuildInfo;
class IDataReader;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup StandardService
 * \brief Classe de base du service de vérification des données
 * entre deux exécutions.
 */
class ARCANE_CORE_EXPORT VerifierService
: public AbstractService
, public IVerifierService
{
 private:
  struct DiffInfo
  {
   public:
    DiffInfo()
    : m_variable_name(), m_nb_diff(0)
      {}
    DiffInfo(const String& var_name,Int64 nb_diff)
    : m_variable_name(var_name), m_nb_diff(nb_diff)
      {}
   public:
    String m_variable_name;
    Int64 m_nb_diff;
  };
 public:

  explicit VerifierService(const ServiceBuildInfo& sbi);
  ~VerifierService() override;

 public:
  
  IBase* serviceParent() const override;
  IServiceInfo* serviceInfo() const override { return m_service_info; }
  IService* serviceInterface() override { return this; }

 public:

  void setFileName(const String& file_name) override { m_file_name = file_name; }
  String fileName() const override { return m_file_name; }

 public:

  void setResultFileName(const String& file_name) override { m_result_file_name = file_name; }
  String resultfileName() const override { return m_result_file_name; }

 public:
  
  void setSubDir(const String& sub_dir) override { m_sub_dir = sub_dir; }
  String subDir() const override { return m_sub_dir; }

 protected:

  virtual void _doVerif(IDataReader* reader,const VariableCollection& variables,bool compare_ghost);
  //! Remplit dans \a variables la liste des variables devant être relues
  virtual void _getVariables(VariableList variables,bool parallel_sequential);

 public:

  ISubDomain* subDomain() { return m_sub_domain; }

 protected:

 private:

  ISubDomain* m_sub_domain;
  IServiceInfo* m_service_info;
  String m_file_name;
  String m_result_file_name;
  String m_sub_dir;

 private:

  template<typename ReaderType> void
  _doVerif2(ReaderType reader,const VariableList& variables,bool compare_ghost);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
