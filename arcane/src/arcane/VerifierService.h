// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VerifierService.h                                           (C) 2000-2014 */
/*                                                                           */
/* Classe de base du service de vérification des données.                    */
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

ARCANE_BEGIN_NAMESPACE

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

  VerifierService(const ServiceBuildInfo& sbi);
  virtual ~VerifierService();

 public:
  
  virtual IBase* serviceParent() const;
  virtual IServiceInfo* serviceInfo() const { return m_service_info; }
  virtual IService* serviceInterface() { return this; }

 public:

  virtual void setFileName(const String& file_name) { m_file_name = file_name; }
  virtual const String& fileName() const { return m_file_name; }

 public:

  virtual void setResultFileName(const String& file_name) { m_result_file_name = file_name; }
  virtual const String& resultfileName() const { return m_result_file_name; }

 public:
  
  virtual void setSubDir(const String& sub_dir) { m_sub_dir = sub_dir; }
  virtual const String& subDir() const { return m_sub_dir; }

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

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
