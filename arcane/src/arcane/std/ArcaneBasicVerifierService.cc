// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ArcaneBasicVerifierService.cc                               (C) 2000-2024 */
/*                                                                           */
/* Variable comparison service.                                              */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/StringBuilder.h"
#include "arcane/utils/ValueConvert.h"

#include "arcane/core/IParallelMng.h"
#include "arcane/core/VariableCollection.h"
#include "arcane/core/IVariableUtilities.h"
#include "arcane/core/VerifierService.h"
#include "arcane/core/ItemGroup.h"
#include "arcane/core/IVariableMng.h"
#include "arcane/core/IVariable.h"
#include "arcane/core/ISubDomain.h"
#include "arcane/core/ServiceFactory.h"
#include "arcane/core/IData.h"
#include "arcane/core/internal/IVariableInternal.h"

#include "arcane/std/internal/BasicReader.h"
#include "arcane/std/internal/BasicWriter.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
using namespace Arcane::impl;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ArcaneBasicVerifierService
: public VerifierService
{
  class GroupFinder
  : public BasicReader::IItemGroupFinder
  {
   public:

    GroupFinder(IVariableMng* vm)
    : m_variable_mng(vm)
    {}
    ItemGroup getWantedGroup(VariableMetaData* vmd) override
    {
      String full_name = vmd->fullName();
      IVariable* var = m_variable_mng->findVariableFullyQualified(full_name);
      if (!var)
        ARCANE_FATAL("Variable '{0}' not found");
      return var->itemGroup();
    }

   private:

    IVariableMng* m_variable_mng;
  };

 public:

  explicit ArcaneBasicVerifierService(const ServiceBuildInfo& sbi)
  : VerifierService(sbi)
  {
  }

 public:

  void build() override {}
  void writeReferenceFile() override;
  void doVerifFromReferenceFile(bool parallel_sequential, bool compare_ghost) override;

 protected:

  void _setFormatVersion(Int32 v)
  {
    m_wanted_format_version = v;
  }

 private:

  String m_full_file_name;
  Int32 m_wanted_format_version = 1;

 private:

  void _computeFullFileName(bool is_read)
  {
    ARCANE_UNUSED(is_read);
    StringBuilder s = fileName();
    //m_full_file_name = fileName();
    const String& sub_dir = subDir();
    if (!sub_dir.empty()) {
      s += "/";
      s += sub_dir;
    }
    m_full_file_name = s;
  }
  void _doVerifHash(BasicReader* reader, const VariableCollection& variables);
  void _writeReferenceFile(const String& file_name);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcaneBasicVerifierService::
_writeReferenceFile(const String& file_name)
{
  ISubDomain* sd = subDomain();
  IParallelMng* pm = sd->parallelMng();
  IVariableMng* vm = sd->variableMng();
  auto open_mode = BasicReaderWriterCommon::OpenModeTruncate;
  // For now, use version 1
  // Starting from January 2019, it is possible to use version 2 or 3
  // because the C# comparator supports this version.
  Int32 version = m_wanted_format_version;
  bool want_parallel = pm->isParallel();
  ScopedPtrT<BasicWriter> verif(new BasicWriter(sd->application(), pm, file_name,
                                                open_mode, version, want_parallel));
  if (compareMode() == eCompareMode::HashOnly)
    verif->setSaveValues(false);

  verif->initialize();

  // In parallel, since writing requires communication between subdomains,
  // it is essential that all PEs have the same variables. We filter them to
  // ensure this.
  VariableCollection used_variables = vm->usedVariables();
  if (pm->isParallel()) {
    bool dump_not_common = true;
    VariableCollection filtered_variables = vm->utilities()->filterCommonVariables(pm, used_variables, dump_not_common);
    vm->writeVariables(verif.get(), filtered_variables);
  }
  else {
    vm->writeVariables(verif.get(), used_variables);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcaneBasicVerifierService::
writeReferenceFile()
{
  _computeFullFileName(false);
  String dir_name = platform::getFileDirName(m_full_file_name);
  platform::recursiveCreateDirectory(m_full_file_name);
  _writeReferenceFile(m_full_file_name);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcaneBasicVerifierService::
doVerifFromReferenceFile(bool parallel_sequential, bool compare_ghost)
{
  ISubDomain* sd = subDomain();
  IParallelMng* pm = sd->parallelMng();
  IVariableMng* vm = subDomain()->variableMng();
  ITraceMng* tm = sd->traceMng();
  _computeFullFileName(true);
  bool want_parallel = pm->isParallel();
  Ref<BasicReader> reader = makeRef(new BasicReader(sd->application(), pm, A_NULL_RANK, m_full_file_name, want_parallel));
  reader->initialize();
  GroupFinder group_finder(vm);
  reader->setItemGroupFinder(&group_finder);

  VariableList read_variables;
  _getVariables(read_variables, parallel_sequential);

  // In parallel, since reading requires communication between subdomains,
  // it is essential that all PEs have the same variables. We filter them to
  // ensure this.
  if (pm->isParallel()) {
    IVariableMng* vm = sd->variableMng();
    bool dump_not_common = true;
    VariableCollection filtered_variables = vm->utilities()->filterCommonVariables(pm, read_variables, dump_not_common);
    read_variables = filtered_variables;
  }

  tm->info() << "Checking (" << m_full_file_name << ")";
  reader->beginRead(read_variables);
  if (compareMode() == eCompareMode::Values)
    _doVerif(reader.get(), read_variables, compare_ghost);
  if (compareMode() == eCompareMode::HashOnly)
    _doVerifHash(reader.get(), read_variables);
  reader->endRead();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
namespace
{
  String
  _getHashValueOrNull(const std::map<String, String>& comparison_hash_map, const String& name)
  {
    auto x = comparison_hash_map.find(name);
    if (x != comparison_hash_map.end())
      return x->second;
    return {};
  }
} // namespace

void ArcaneBasicVerifierService::
_doVerifHash(BasicReader* ref_reader, const VariableCollection& variables)
{
  ISubDomain* sd = subDomain();
  IParallelMng* pm = sd->parallelMng();
  bool is_master = pm->isMasterIO();
  const bool want_parallel = pm->isParallel();

  info() << "Check Verif Hash";
  // Retrieves the hash calculation algorithm
  IHashAlgorithm* ref_compare_hash_algo = ref_reader->comparisonHashAlgorithm();
  if (!ref_compare_hash_algo)
    // TODO: check if we should do nothing or signal an error.
    return;

  // Calculates the hash of the current variable values
  std::map<String, String> current_comparison_hash_map;
  ParallelDataWriterList parallel_data_writers;
  info() << "DoVerifHash";
  for (VariableCollection::Enumerator ivar(variables); ++ivar;) {
    IVariable* var = *ivar;
    Ref<IData> allocated_data;
    IData* data = var->data();
    // In parallel, sort entities by increasing uniqueId().
    // In sequential mode, this is always the case.
    // NOTE: for now we use the global IParallelMng but we should
    // check if we should not use the one associated with the
    // current variable's mesh \a var.
    if (want_parallel) {
      // In parallel, only compare variables on entities
      ItemGroup group = var->itemGroup();
      if (group.null())
        continue;
      Ref<ParallelDataWriter> writer = parallel_data_writers.getOrCreateWriter(group);
      allocated_data = writer->getSortedValues(data);
      data = allocated_data.get();
    }
    String hash_string = var->_internalApi()->computeComparisonHashCollective(ref_compare_hash_algo, data);
    if (!hash_string.empty())
      current_comparison_hash_map.try_emplace(var->fullName(), hash_string);
  }

  std::map<String, String> ref_comparison_hash_map;
  ref_reader->fillComparisonHash(ref_comparison_hash_map);
  if (is_master) {
    Int32 nb_variable = 0;
    Int32 nb_compared = 0;
    Int32 nb_different = 0;
    for (VariableCollection::Enumerator ivar(variables); ++ivar;) {
      IVariable* var = *ivar;
      String var_full_name = var->fullName();
      ++nb_variable;
      String ref_hash = _getHashValueOrNull(ref_comparison_hash_map, var_full_name);
      String current_hash = _getHashValueOrNull(current_comparison_hash_map, var_full_name);
      if (!ref_hash.empty() && !current_hash.empty()) {
        ++nb_compared;
        if (ref_hash != current_hash) {
          info() << "Different hash ref_hash=" << ref_hash << " current=" << current_hash
                 << " var=" << var_full_name;
          ++nb_different;
        }
        else
          info(4) << "Found Hash hash=" << ref_hash << " var=" << var_full_name;
      }
    }
    info() << "NbVariable=" << nb_variable << " nb_compared=" << nb_compared << " nb_different=" << nb_different;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ArcaneBasicVerifierService2
: public ArcaneBasicVerifierService
{
 public:

  explicit ArcaneBasicVerifierService2(const ServiceBuildInfo& sbi)
  : ArcaneBasicVerifierService(sbi)
  {
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ArcaneBasicVerifierServiceV3
: public ArcaneBasicVerifierService
{
 public:

  explicit ArcaneBasicVerifierServiceV3(const ServiceBuildInfo& sbi)
  : ArcaneBasicVerifierService(sbi)
  {
    _setFormatVersion(3);
    if (auto v = Convert::Type<Int32>::tryParseFromEnvironment("ARCANE_VERIF_HASHONLY", true)) {
      bool use_hash = (v.value() != 0);
      info() << "ArcaneBasicVerifierServiceV3: using hash?=" << use_hash;
      setCompareMode(use_hash ? eCompareMode::HashOnly : eCompareMode::Values);
    }
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE(ArcaneBasicVerifierService2,
                        ServiceProperty("ArcaneBasicVerifier2", ST_SubDomain),
                        ARCANE_SERVICE_INTERFACE(IVerifierService));

ARCANE_REGISTER_SERVICE(ArcaneBasicVerifierServiceV3,
                        ServiceProperty("ArcaneBasicVerifier3", ST_SubDomain),
                        ARCANE_SERVICE_INTERFACE(IVerifierService));

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
