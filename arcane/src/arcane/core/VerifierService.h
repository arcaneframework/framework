// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VerifierService.h                                           (C) 2000-2025 */
/*                                                                           */
/* Base class for the variable verification service.                         */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_VERIFIERSERVICE_H
#define ARCANE_CORE_VERIFIERSERVICE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/String.h"

#include "arcane/core/ArcaneTypes.h"
#include "arcane/core/AbstractService.h"
#include "arcane/core/IVerifierService.h"
#include "arcane/core/VariableComparer.h"

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
 * \brief Base class for the data verification service
 * between two runs.
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
    : m_variable_name()
    , m_nb_diff(0)
    {}
    DiffInfo(const String& var_name, Int64 nb_diff)
    : m_variable_name(var_name)
    , m_nb_diff(nb_diff)
    {}

   public:

    String m_variable_name;
    Int64 m_nb_diff;
  };

 public:

  explicit VerifierService(const ServiceBuildInfo& sbi);

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

  //! Desired comparison type
  void setCompareMode(eCompareMode v) override { m_compare_mode = v; }
  eCompareMode compareMode() const override { return m_compare_mode; }

  void setSubDir(const String& sub_dir) override { m_sub_dir = sub_dir; }
  String subDir() const override { return m_sub_dir; }

  void setComputeDifferenceMethod(eVariableComparerComputeDifferenceMethod v) override
  {
    m_compute_diff_method = v;
  }
  eVariableComparerComputeDifferenceMethod computeDifferenceMethod() const override
  {
    return m_compute_diff_method;
  }

 protected:

  virtual void _doVerif(IDataReader* reader, const VariableCollection& variables, bool compare_ghost);
  //! Fills the list of variables that need to be reread in \a variables
  virtual void _getVariables(VariableList variables, bool parallel_sequential);

 public:

  ISubDomain* subDomain() { return m_sub_domain; }

 protected:
 private:

  ISubDomain* m_sub_domain = nullptr;
  IServiceInfo* m_service_info = nullptr;
  String m_file_name;
  String m_result_file_name;
  String m_sub_dir;
  eCompareMode m_compare_mode = eCompareMode::Values;
  eVariableComparerComputeDifferenceMethod m_compute_diff_method = eVariableComparerComputeDifferenceMethod::Relative;

 private:

  template <typename ReaderType> void
  _doVerif2(ReaderType reader, const VariableList& variables, bool compare_ghost);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
