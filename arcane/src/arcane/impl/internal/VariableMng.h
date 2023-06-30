// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VariableMng.h                                               (C) 2000-2023 */
/*                                                                           */
/* Classe gérant la liste des maillages.                                     */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IMPL_INTERNAL_VARIABLEMNG_H
#define ARCANE_IMPL_INTERNAL_VARIABLEMNG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"
#include "arcane/utils/String.h"
#include "arcane/utils/HashTableMap.h"
#include "arcane/utils/List.h"

#include "arcane/core/IVariableMng.h"
#include "arcane/core/IVariableFilter.h"
#include "arcane/core/VariableCollection.h"

#include <map>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class VariableMetaDataList;
class VariableReaderMng;
class XmlNode;
class VariableIOWriterMng;
class VariableIOReaderMng;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Gestionnaire de variables.
 */
class VariableMng
: public TraceAccessor
, public IVariableMng
{
  friend class VariableIOWriterMng;
  friend class VariableIOReaderMng;

  class VariableNameInfo
  {
   public:

    String m_local_name;
    String m_family_name;
    String m_mesh_name;
    bool operator==(const VariableNameInfo& vni) const
    {
      return m_local_name == vni.m_local_name && m_family_name == vni.m_family_name && m_mesh_name == vni.m_mesh_name;
    }

   public:

    Int32 hash() const
    {
      Int32 h = 0;
      if (!m_mesh_name.null()) {
        h = hash(h, m_mesh_name.localstr());
        h = hash(h, "_");
      }
      if (!m_family_name.null()) {
        h = hash(h, m_family_name.localstr());
        h = hash(h, "_");
      }
      h = hash(h, m_local_name.localstr());
      return h;
    }

   public:

    static Int32 hash(Int32 h, const char* p)
    {
      if (!p)
        return h;
      for (; *p != '\0'; ++p)
        h = (h << 5) - h + *p;
      return h;
    }
  };

  class VNIComparer
  {
   public:

    typedef const VariableNameInfo& KeyTypeConstRef;
    typedef VariableNameInfo& KeyTypeRef;
    typedef VariableNameInfo KeyTypeValue;
    typedef FalseType Printable;
    typedef Int32 HashValueType;
    static Int32 hashFunction(const VariableNameInfo& key)
    {
      // garantie que la valeur est positive
      return key.hash() & 0x7fffffff;
    }
  };

 public:

  using VNIMap = HashTableMapT<VariableNameInfo, IVariable*, VNIComparer>;

 public:

  explicit VariableMng(ISubDomain* sd);
  ~VariableMng() override;

 public:

  void build() override;
  void initialize() override;
  void removeAllVariables() override;
  void detachMeshVariables(IMesh* mesh) override;

 public:

  ISubDomain* subDomain() override { return m_sub_domain; }
  IParallelMng* parallelMng() const override { return m_parallel_mng; }
  ITraceMng* traceMng() override { return TraceAccessor::traceMng(); }
  IVariable* checkVariable(const VariableInfo& infos) override;
  void addVariableRef(VariableRef* ref) override;
  void addVariable(IVariable* var) override;
  void removeVariableRef(VariableRef*) override;
  void removeVariable(IVariable* var) override;
  void dumpList(std::ostream&, IModule*) override;
  void dumpList(std::ostream&) override;
  void initializeVariables(bool) override;
  String generateTemporaryVariableName() override;
  void variables(VariableRefCollection, IModule*) override;
  VariableCollection variables() override;
  VariableCollection usedVariables() override;
  void notifyUsedVariableChanged() override { m_used_variables_changed = true; }
  Real exportSize(const VariableCollection& vars) override;
  IObservable* writeObservable() override { return m_write_observable; }
  IObservable* readObservable() override { return m_read_observable; }
  void writeVariables(IDataWriter*, const VariableCollection& vars) override;
  void writeVariables(IDataWriter*, IVariableFilter*) override;
  void writeCheckpoint(ICheckpointWriter*) override;
  void writePostProcessing(IPostProcessorWriter* writer) override;
  void readVariables(IDataReader*, IVariableFilter*) override;
  void readCheckpoint(ICheckpointReader*) override;
  void readCheckpoint(const CheckpointReadInfo& infos) override;
  IVariable* findVariable(const String& name) override;
  IVariable* findMeshVariable(IMesh* mesh, const String& name) override;
  IVariable* findVariableFullyQualified(const String& name) override;

  void dumpStats(std::ostream& ostr, bool is_verbose) override;
  void dumpStatsJSON(JSONWriter& writer) override;
  IVariableUtilities* utilities() const override { return m_utilities; }

  EventObservable<const VariableStatusChangedEventArgs&>&
  onVariableAdded() override
  {
    return m_on_variable_added;
  }

  EventObservable<const VariableStatusChangedEventArgs&>&
  onVariableRemoved() override
  {
    return m_on_variable_removed;
  }
  ISubDomain* _internalSubDomain() const override { return m_sub_domain; }

 public:

  static bool isVariableToSave(IVariable& var);

 private:

  //! Type de la liste des variables par nom complet
  using FullNameVariableMap = std::map<String, IVariable*>;
  //! Paire de la liste des variables par nom complet
  using FullNameVariablePair = FullNameVariableMap::value_type;

  //! Type de la liste des fabriques de variables par nom complet
  using VariableFactoryMap = std::map<String, IVariableFactory*>;
  //! Paire de la liste des variables par nom complet
  using VariableFactoryPair = VariableFactoryMap::value_type;

  //! Gestionnaire de sous-domaine
  ISubDomain* m_sub_domain = nullptr;
  IParallelMng* m_parallel_mng = nullptr;
  ITimeStats* m_time_stats = nullptr;
  VariableRefList m_variables_ref; //!< Liste des variables
  VariableList m_variables;
  VariableList m_used_variables;
  bool m_variables_changed = true;
  bool m_used_variables_changed = true;
  //! Liste des variables par nom complet
  FullNameVariableMap m_full_name_variable_map;
  VNIMap m_vni_map;
  IObservable* m_write_observable = nullptr;
  IObservable* m_read_observable = nullptr;
  EventObservable<const VariableStatusChangedEventArgs&> m_on_variable_added;
  EventObservable<const VariableStatusChangedEventArgs&> m_on_variable_removed;
  List<IVariableFactory*> m_variable_factories;
  //! Liste des variables créées automatiquement lors d'une reprise
  List<VariableRef*> m_auto_create_variables;
  VariableFactoryMap m_variable_factory_map;

  Integer m_generate_name_id = 0; //!< Numéro utilisé pour générer un nom de variable

  Int64 m_nb_created_variable_reference = 0;
  Int64 m_nb_created_variable = 0;
  // Indique dans quel module une variable est créée
  std::map<IVariable*, IModule*> m_variable_creation_modules;

  IVariableUtilities* m_utilities = nullptr;
  VariableIOWriterMng* m_variable_io_writer_mng = nullptr;
  VariableIOReaderMng* m_variable_io_reader_mng = nullptr;

 private:

  //! Ecrit la valeur de la variable \a v sur le flot \a o
  void _dumpVariable(const VariableRef& v, std::ostream& o);

  static const char* _msgClassName() { return "Variable"; }
  VariableRef* _createVariableFromType(const String& full_type,
                                       const VariableBuildInfo& vbi);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Gestion de l'écriture pour les variables
 */
class VariableIOWriterMng
: public TraceAccessor
{
 private:

  class CheckpointSaveFilter
  : public IVariableFilter
  {
   public:

    bool applyFilter(IVariable& var) override
    {
      return VariableMng::isVariableToSave(var);
    }
  };

 public:

  explicit VariableIOWriterMng(VariableMng* vm);

 public:

  void writeCheckpoint(ICheckpointWriter* service);
  void writePostProcessing(IPostProcessorWriter* post_processor);
  void writeVariables(IDataWriter* writer,const VariableCollection& vars, bool use_hash);
  void writeVariables(IDataWriter* writer,IVariableFilter* filter, bool use_hash);

 private:

  VariableMng* m_variable_mng = nullptr;

 private:

  void _writeVariables(IDataWriter* writer, const VariableCollection& vars, bool use_hash);
  String _generateMetaData(const VariableCollection& vars, bool use_hash);
  void _generateVariablesMetaData(XmlNode variables_node, const VariableCollection& vars, bool use_hash);
  void _generateMeshesMetaData(XmlNode meshes_node);
  static const char* _msgClassName() { return "Variable"; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Gestion de la lecture pour les variables
 */
class VariableIOReaderMng
: public TraceAccessor
{
 public:

  class IDataReaderWrapper;
  class OldDataReaderWrapper;
  class DataReaderWrapper;
  class VariableDataTypeInfo;

 public:

  explicit VariableIOReaderMng(VariableMng* vm);

 public:

  void readCheckpoint(ICheckpointReader* service);
  void readCheckpoint(const CheckpointReadInfo& infos);
  void readVariables(IDataReader* reader, IVariableFilter* filter);

 private:

  VariableMng* m_variable_mng = nullptr;

 private:

  void _readVariablesData(VariableReaderMng& var_read_mng, IDataReaderWrapper* reader);
  void _readMetaData(VariableMetaDataList& vmd_list, Span<const Byte> bytes);
  void _checkHashFunction(const VariableMetaDataList& vmd_list);
  void _createVariablesFromMetaData(const VariableMetaDataList& vmd_list);
  void _readVariablesMetaData(VariableMetaDataList& vmd_list, const XmlNode& variables_node);
  void _readMeshesMetaData(const XmlNode& meshes_node);
  void _buildFilteredVariableList(VariableReaderMng& vars_read_mng, IVariableFilter* filter);
  void _finalizeReadVariables(const VariableList& vars_to_read);
  static const char* _msgClassName() { return "Variable"; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
