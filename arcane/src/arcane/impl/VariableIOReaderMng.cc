// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VariableIOMng.cc                                            (C) 2000-2024 */
/*                                                                           */
/* Classe gérant les entrées/sorties pour les variables.                     */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/impl/internal/VariableMng.h"

#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/ParallelFatalErrorException.h"
#include "arcane/utils/IObservable.h"
#include "arcane/utils/ScopedPtr.h"
#include "arcane/utils/StringBuilder.h"
#include "arcane/utils/MD5HashAlgorithm.h"
#include "arcane/utils/SHA1HashAlgorithm.h"
#include "arcane/utils/OStringStream.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/JSONReader.h"
#include "arcane/utils/ValueConvert.h"

#include "arcane/core/IXmlDocumentHolder.h"
#include "arcane/core/XmlNode.h"
#include "arcane/core/VariableMetaData.h"
#include "arcane/core/IData.h"
#include "arcane/core/IMesh.h"
#include "arcane/core/Properties.h"
#include "arcane/core/Timer.h"
#include "arcane/core/VarRefEnumerator.h"
#include "arcane/core/CheckpointInfo.h"
#include "arcane/core/Directory.h"
#include "arcane/core/ISerializedData.h"
#include "arcane/core/VariableBuildInfo.h"
#include "arcane/core/XmlNodeList.h"
#include "arcane/core/IMeshMng.h"
#include "arcane/core/IMeshFactoryMng.h"
#include "arcane/core/MeshBuildInfo.h"
#include "arcane/core/internal/IDataInternal.h"

#include "arcane/core/ICheckpointReader.h"
#include "arcane/core/IDataReader.h"
#include "arcane/core/IDataReader2.h"

#include "arcane/core/ISubDomain.h"
#include "arcane/core/IParallelMng.h"

#include "arcane/core/internal/IParallelMngInternal.h"
#include "arcane/core/internal/IVariableInternal.h"

// TODO: gérer le hash en version 64 bits.

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Extraction des infos de type à partir d'une chaîne de caractères
// NOTE L'extraction doit être cohérente avec la construction qui est
// dans Variable.cc
class VariableIOReaderMng::VariableDataTypeInfo
{
 public:

  explicit VariableDataTypeInfo(const String& full_type)
  : m_is_partial(false)
  {
    std::vector<String> split_strs;
    full_type.split(split_strs, '.');
    size_t nb_split = split_strs.size();
    if (nb_split == 5) {
      if (split_strs[4] != "Partial")
        ARCANE_FATAL("Invalid value for partial full_type '{0}'", full_type);
      m_is_partial = true;
      --nb_split;
    }
    if (nb_split != 4)
      ARCANE_FATAL("Invalid value for full_type '{0}'", full_type);
    m_data_type_name = split_strs[0];
    m_item_kind_name = split_strs[1];
    m_dimension = split_strs[2];
    m_multi_tag = split_strs[3];
  }

 public:

  const String& dataTypeName() const { return m_data_type_name; }
  const String& itemKindName() const { return m_item_kind_name; }
  const String& dimension() const { return m_dimension; }
  const String& multiTag() const { return m_multi_tag; }
  bool isPartial() const { return m_is_partial; }

 private:

  String m_data_type_name;
  String m_item_kind_name;
  String m_dimension;
  String m_multi_tag;
  bool m_is_partial;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface pour utiliser IDataReader ou IDataReader2.
 */
class VariableIOReaderMng::IDataReaderWrapper
{
 public:

  virtual ~IDataReaderWrapper() = default;

 public:

  virtual void beginRead(const VariableCollection& vars) = 0;
  virtual void read(VariableMetaData* vmd, IVariable* var, IData* data) = 0;
  virtual void endRead() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Wrapper pour IDataReader.
 */
class VariableIOReaderMng::OldDataReaderWrapper
: public IDataReaderWrapper
{
 public:

  explicit OldDataReaderWrapper(IDataReader* reader)
  : m_reader(reader)
  {}
  void beginRead(const VariableCollection& vars) override
  {
    return m_reader->beginRead(vars);
  }
  void read(VariableMetaData* vmd, IVariable* var, IData* data) override
  {
    ARCANE_UNUSED(vmd);
    m_reader->read(var, data);
  }
  void endRead() override
  {
    return m_reader->endRead();
  }

 private:

  IDataReader* m_reader;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Wrapper pour IDataReader2.
 */
class VariableIOReaderMng::DataReaderWrapper
: public IDataReaderWrapper
{
 public:

  explicit DataReaderWrapper(IDataReader2* reader)
  : m_reader(reader)
  {}
  void beginRead(const VariableCollection& vars) override
  {
    ARCANE_UNUSED(vars);
    return m_reader->beginRead(DataReaderInfo());
  }
  void read(VariableMetaData* vmd, IVariable* var, IData* data) override
  {
    ARCANE_UNUSED(var);
    m_reader->read(VariableDataReadInfo(vmd, data));
  }
  void endRead() override
  {
    return m_reader->endRead();
  }

 private:

  IDataReader2* m_reader;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class VariableMetaDataList
{
 public:

  typedef std::map<String, Ref<VariableMetaData>> VMDMap;

 public:

  VariableMetaDataList() = default;
  VariableMetaDataList(const VariableMetaDataList& rhs) = delete;
  VariableMetaDataList& operator=(const VariableMetaDataList& rhs) = delete;
  ~VariableMetaDataList()
  {
    clear();
  }

 public:

  VariableMetaData* add(const String& base_name, const String& mesh_name,
                        const String& family_name, const String& group_name,
                        bool is_partial)
  {
    auto vmd = makeRef(new VariableMetaData(base_name, mesh_name, family_name, group_name, is_partial));
    return add(vmd);
  }
  VariableMetaData* add(Ref<VariableMetaData> vmd)
  {
    m_vmd_map.insert(std::make_pair(vmd->fullName(), vmd));
    return vmd.get();
  }
  void clear()
  {
    m_vmd_map.clear();
  }
  VariableMetaData* findMetaData(const String& full_name)
  {
    auto x = m_vmd_map.find(full_name);
    if (x != m_vmd_map.end())
      return x->second.get();
    return nullptr;
  }
  VMDMap::const_iterator begin() const { return m_vmd_map.begin(); }
  VMDMap::const_iterator end() const { return m_vmd_map.end(); }
  void setHashAlgorithmName(const String& v) { m_hash_algorithm = v; }
  const String& hashAlgorithmName() const { return m_hash_algorithm; }

 public:

  std::map<String, Ref<VariableMetaData>> m_vmd_map;
  String m_hash_algorithm;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Gestion de la lecture des variables.
 */
class VariableReaderMng
: public TraceAccessor
{
  struct VarReadInfo
  {
   public:

    VarReadInfo(IVariable* var, IData* data, VariableMetaData* meta_data)
    : m_variable(var)
    , m_data(data)
    , m_meta_data(meta_data)
    {}

   public:

    IVariable* m_variable;
    IData* m_data;
    VariableMetaData* m_meta_data;
  };

 public:

  explicit VariableReaderMng(ITraceMng* tm)
  : TraceAccessor(tm)
  {}

 public:

  void readVariablesData(IVariableMng* vm, VariableIOReaderMng::IDataReaderWrapper* reader);
  VariableMetaDataList& variableMetaDataList() { return m_vmd_list; }
  const VariableList& variablesToRead() { return m_vars_to_read; }

 private:

  VariableMetaDataList m_vmd_list;
  VariableList m_vars_to_read;
  UniqueArray<VarReadInfo> m_var_read_info_list;

 private:

  void _buildVariablesToRead(IVariableMng* vm);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableReaderMng::
_buildVariablesToRead(IVariableMng* vm)
{
  m_vars_to_read.clear();
  m_var_read_info_list.clear();
  for (const auto& x : m_vmd_list) {
    const String& full_name = x.first;
    IVariable* var = vm->findVariableFullyQualified(full_name);
    if (!var)
      ARCANE_FATAL("Var {0} not in VariableMng", full_name);
    m_vars_to_read.add(var);
    m_var_read_info_list.add(VarReadInfo(var, var->data(), x.second.get()));
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableReaderMng::
readVariablesData(IVariableMng* vm, VariableIOReaderMng::IDataReaderWrapper* reader)
{
  _buildVariablesToRead(vm);
  reader->beginRead(m_vars_to_read);
  for (const auto& ivar : m_var_read_info_list) {
    // NOTE: var peut-être nul
    IVariable* var = ivar.m_variable;
    IData* data = ivar.m_data;
    VariableMetaData* vmd = ivar.m_meta_data;
    String exception_message;
    bool has_error = false;
    try {
      reader->read(vmd, var, data);
      if (var)
        var->notifyEndRead();
    }
    catch (const Exception& ex) {
      OStringStream ostr;
      ostr() << ex;
      exception_message = ostr.str();
      has_error = true;
    }
    catch (const std::exception& ex) {
      exception_message = ex.what();
      has_error = true;
    }
    if (has_error) {
      OStringStream ostr;
      String var_full_name = vmd->fullName();
      ostr() << "Variable = " << var_full_name;
      if (var) {
        for (VarRefEnumerator ivar(var); ivar.hasNext(); ++ivar) {
          VariableRef* ref = *ivar;
          String s = ref->assignmentStackTrace();
          if (!s.null())
            ostr() << "Stack assignement: " << s;
        }
      }

      ARCANE_FATAL("Can not read variable variable={0} exception={1} infos={2}",
                   var_full_name, exception_message, ostr.str());
    }
  }
  reader->endRead();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VariableIOReaderMng::
VariableIOReaderMng(VariableMng* vm)
: TraceAccessor(vm->traceMng())
, m_variable_mng(vm)
, m_is_use_json_metadata(true)
{
  if (auto v = Convert::Type<Int32>::tryParseFromEnvironment("ARCANE_USE_JSON_METADATA", true))
    m_is_use_json_metadata = (v.value() != 0);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableIOReaderMng::
readCheckpoint(ICheckpointReader* service)
{
  Trace::Setter mci(traceMng(), _msgClassName());

  if (!service)
    ARCANE_FATAL("No protection service specified");

  service->notifyBeginRead();

  IDataReader* data_reader = service->dataReader();
  if (!data_reader)
    ARCANE_FATAL("no dataReader()");
  String meta_data = data_reader->metaData();

  if (meta_data.null())
    ARCANE_FATAL("No meta-data in checkpoint.");

  OldDataReaderWrapper wrapper(data_reader);

  info(6) << "METADATA (ICheckpointReader): FromCheckpoint: " << meta_data;

  VariableReaderMng var_read_mng(traceMng());
  VariableMetaDataList& vmd_list = var_read_mng.variableMetaDataList();
  {
    _readMetaData(vmd_list, meta_data.bytes());
    _createVariablesFromMetaData(vmd_list);
    _readVariablesData(var_read_mng, &wrapper);
  }

  service->notifyEndRead();
  _checkHashFunction(vmd_list);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableIOReaderMng::
readCheckpoint(const CheckpointReadInfo& infos)
{
  Trace::Setter mci(traceMng(), _msgClassName());

  ICheckpointReader2* service = infos.reader();
  ARCANE_CHECK_POINTER2(service, "No checkpoint service specified");

  IParallelMng* pm = infos.parallelMng();
  ARCANE_CHECK_POINTER2(pm, "no parallelMng()");

  service->notifyBeginRead(infos);

  IDataReader2* data_reader = service->dataReader();
  ARCANE_CHECK_POINTER2(data_reader, "No dataReader()");

  UniqueArray<Byte> meta_data_bytes;
  data_reader->fillMetaData(meta_data_bytes);
  if (meta_data_bytes.empty())
    ARCANE_FATAL("No meta-data in checkpoint.");

  DataReaderWrapper wrapper(data_reader);

  info(6) << "METADATA (ICheckpointReader2): FromCheckpoint: " << String(meta_data_bytes);

  VariableReaderMng var_read_mng(traceMng());
  VariableMetaDataList& vmd_list = var_read_mng.variableMetaDataList();
  {
    _readMetaData(vmd_list, meta_data_bytes);
    _createVariablesFromMetaData(vmd_list);
    _readVariablesData(var_read_mng, &wrapper);
  }

  service->notifyEndRead();
  _checkHashFunction(vmd_list);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableIOReaderMng::
readVariables(IDataReader* reader, IVariableFilter* filter)
{
  Trace::Setter mci(traceMng(), _msgClassName());
  VariableReaderMng var_read_mng(traceMng());
  _buildFilteredVariableList(var_read_mng, filter);
  OldDataReaderWrapper wrapper(reader);
  _readVariablesData(var_read_mng, &wrapper);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Lecture des méta-données.
 *
 * En considérant que \a meta_data est un fichier XML valide, parcours
 * l'ensemble des variables le contenant et crée une référence sur
 * chacune si elles n'existent pas encore.
 *
 * Si nécessaire, pour chaque variable présente dans les méta-données,
 * la créée si elle n'existe pas encore. De plus, sauve le nom pour être
 * certain que la valeur de cette variable sera bien lue.
 */
void VariableIOReaderMng::
_readMetaData(VariableMetaDataList& vmd_list, Span<const Byte> bytes)
{
  ScopedPtrT<IXmlDocumentHolder> doc(IXmlDocumentHolder::loadFromBuffer(bytes, "meta_data", traceMng()));
  if (!doc.get())
    ARCANE_FATAL("The meta-data are invalid");
  String hash_service_name;
  JSONDocument json_reader;

  XmlNode root_node = doc->documentNode().documentElement();
  XmlNode json_node = root_node.child("json");

  // A partir de la version 3.11 de Arcane (juillet 2023), les
  // méta-données sont aussi disponibles au format JSON. On les utilise
  // si 'm_is_use_json_metadata' est vrai.
  JSONValue json_variables;
  JSONValue json_meshes;
  if (!json_node.null()) {
    String json_meta_data = json_node.value();
    info(6) << "READER_JSON=" << json_meta_data;
    json_reader.parse(json_meta_data.bytes());

    JSONValue json_meta_data_object = json_reader.root().expectedChild("arcane-checkpoint-metadata");

    // Lit toujours le nom de l'algorithme même si on n'utilise pas les meta-données
    // car on s'en sert pour les comparaisons de la valeur du hash.
    String hash_algo_name = json_meta_data_object.child("hash-algorithm-name").value();
    vmd_list.setHashAlgorithmName(hash_algo_name);

    if (m_is_use_json_metadata) {
      JSONValue json_version = json_meta_data_object.expectedChild("version");
      Int32 v = json_version.valueAsInt32();
      if (v != 1)
        ARCANE_FATAL("Bad version for JSON Meta Data (v={0}). Only version '1' is supported", v);
      json_variables = json_meta_data_object.expectedChild("variables");
      json_meshes = json_meta_data_object.expectedChild("meshes");
    }
  }
  XmlNode variables_node = root_node.child("variables");
  _readVariablesMetaData(vmd_list, json_variables, variables_node);
  XmlNode meshes_node = root_node.child("meshes");
  _readMeshesMetaData(json_meshes, meshes_node);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vérifie les valeurs des fonctions de hashage.
 *
 * Vérifie pour chaque variable que sa valeur est correcte en calculant
 * sa fonction de hashage et en la comparant à la valeur dans la
 * protection.
 * Si une variable à une valeur différente, elle est écrite dans
 * le répertoire de listing au même niveau que les logs.
 */
void VariableIOReaderMng::
_checkHashFunction(const VariableMetaDataList& vmd_list)
{
  ByteUniqueArray hash_values;
  MD5HashAlgorithm md5_hash_algorithm;
  SHA1HashAlgorithm sha1_hash_algorithm;
  // Par défaut si rien n'est spécifié, il s'agit d'une protection issue
  // d'une version antérieure à la 3.12 de Arcane. Dans ce cas l'algorithme
  // utilisé est 'MD5'.
  IHashAlgorithm* hash_algo = &md5_hash_algorithm;
  Ref<IHashAlgorithmContext> hash_context;
  String hash_service_name = vmd_list.hashAlgorithmName();
  if (!hash_service_name.empty()) {
    if (hash_service_name == "MD5")
      hash_algo = &md5_hash_algorithm;
    else if (hash_service_name == "SHA1") {
      hash_algo = &sha1_hash_algorithm;
      hash_context = sha1_hash_algorithm.createContext();
    }
    else
      ARCANE_FATAL("Not supported hash algorithm '{0}'. Valid values are 'SHA1' or 'MD5'");
  }
  Integer nb_error = 0;
  IParallelMng* pm = m_variable_mng->m_parallel_mng;
  Int32 sid = pm->commRank();
  Directory listing_dir = m_variable_mng->subDomain()->listingDirectory();
  for (const auto& i : vmd_list) {
    const VariableMetaData* vmd = i.second.get();
    Int32 hash_version = vmd->hashVersion();
    String reference_hash = (hash_version > 0) ? vmd->hash2() : vmd->hash();
    // Teste si la valeur de hashage est présente. C'est normalement
    // toujours le cas, sauf si la protection vient d'une ancienne
    // version de Arcane qui ne sauvait pas cette information.
    // Ce test pourra être supprimé plus tard.
    if (reference_hash.null())
      continue;
    const String& full_name = i.first;
    IVariable* var = m_variable_mng->findVariableFullyQualified(full_name);
    if (!var)
      // Ne devrait pas arriver
      continue;
    hash_values.clear();
    IData* data = var->data();
    String hash_str;
    bool do_compare = true;
    if (hash_version > 0) {
      ARCANE_CHECK_POINTER(hash_context.get());
      hash_context->reset();
      DataHashInfo hash_info(hash_context.get());
      data->_commonInternal()->computeHash(hash_info);
      HashAlgorithmValue hash_value;
      hash_context->computeHashValue(hash_value);
      hash_str = Convert::toHexaString(asBytes(hash_value.bytes()));
      // Ne compare si les versions de hash associées à la variable différent
      if (hash_version != hash_info.version())
        do_compare = false;
    }
    else {
      data->computeHash(hash_algo, hash_values);
      hash_str = Convert::toHexaString(hash_values);
    }
    if (do_compare && (hash_str != reference_hash)) {
      ++nb_error;
      error() << "Hash values are different. Corrumpted values."
              << " name=" << var->fullName()
              << " ref=" << reference_hash
              << " current=" << hash_str;
      Ref<ISerializedData> sdata(data->createSerializedDataRef(false));
      Span<const Byte> buf(sdata->constBytes());
      String fname = listing_dir.file(String::format("dump-{0}-sid_{1}", var->fullName(), sid));
      std::ofstream ofile(fname.localstr());
      ofile.write(reinterpret_cast<const char*>(buf.data()), buf.size());
    }
  }
  Integer total_nb_error = pm->reduce(Parallel::ReduceSum, nb_error);
  if (total_nb_error != 0) {
    bool allow_bad = !platform::getEnvironmentVariable("ARCANE_ALLOW_DIFFERENT_CHECKPOINT_HASH").null();
    if (!allow_bad)
      throw ParallelFatalErrorException(A_FUNCINFO, "hash functions differs");
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableIOReaderMng::
_createVariablesFromMetaData(const VariableMetaDataList& vmd_list)
{
  ISubDomain* sd = m_variable_mng->subDomain();
  // Récupère ou construit les variables qui n'existent pas encore.
  for (const auto& xvmd : vmd_list) {
    auto& vmd = *(xvmd.second.get());
    const String& full_name = vmd.fullName();
    IVariable* var = m_variable_mng->findVariableFullyQualified(full_name);
    if (var)
      continue;
    const String& base_name = vmd.baseName();
    Integer property = vmd.property();
    const String& mesh_name = vmd.meshName();
    const String& group_name = vmd.itemGroupName();
    const String& family_name = vmd.itemFamilyName();
    VariableBuildInfo vbi(sd, base_name, property);
    if (!mesh_name.null()) {
      if (vmd.isPartial())
        vbi = VariableBuildInfo(sd, base_name, mesh_name, family_name, group_name, property);
      else
        vbi = VariableBuildInfo(sd, base_name, mesh_name, family_name, property);
    }
    info(5) << "Create variable TYPE=" << full_name;
    VariableRef* variable_ref = m_variable_mng->_createVariableFromType(vmd.fullType(), vbi);

    if (vbi.property() & IVariable::PInShMem) {
      IParallelMng* pm{};
      // Si la variable utilise un maillage, il sera créé par _readMeshesMetaData().
      if (!mesh_name.null()) {
        MeshHandle* mesh_handle = sd->meshMng()->findMeshHandle(mesh_name, true);
        pm = mesh_handle->mesh()->parallelMng();
      }
      else {
        pm = sd->parallelMng();
      }
      variable_ref->variable()->_internalApi()->changeAllocator(MemoryAllocationOptions(pm->_internalApi()->dynamicMachineMemoryWindowMemoryAllocator()));
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableIOReaderMng::
_readVariablesMetaData(VariableMetaDataList& vmd_list, JSONValue variables_json,
                       const XmlNode& variables_node)
{
  String ustr_base_name("base-name");
  String ustr_family_name("item-family-name");
  String ustr_group_name("item-group-name");
  String ustr_mesh_name("mesh-name");
  String ustr_full_type("full-type");
  String ustr_data_type("data-type");
  String ustr_hash("hash");
  String ustr_hash2("hash2");
  String ustr_hash_version("hash-version");
  String ustr_property("property");
  String ustr_multitag("multi-tag");
  vmd_list.clear();

  struct VariableReadInfo
  {
    String full_type;
    String base_name;
    String mesh_name;
    String family_name;
    String group_name;
    String hash_value;
    String hash2_value;
    Int32 hash_version = 0;
    String multi_tag;
    String data_type;
    Int32 property = 0;
  };
  UniqueArray<VariableReadInfo> variables_info;

  // Lit les informations des variables à partir des données JSON
  // si ces dernières existent.
  if (!variables_json.null()) {
    // Lecture via JSON
    // Déclare la liste ici pour éviter de retourner un temporaire dans 'for-range'
    JSONValueList vars = variables_json.valueAsArray();
    for (const JSONValue& var : vars) {
      VariableReadInfo r;
      r.full_type = var.expectedChild(ustr_full_type).value();
      r.base_name = var.expectedChild(ustr_base_name).value();
      r.data_type = var.child(ustr_data_type).value();
      r.mesh_name = var.child(ustr_mesh_name).value();
      r.family_name = var.child(ustr_family_name).value();
      r.group_name = var.child(ustr_group_name).value();
      r.hash_value = var.child(ustr_hash).value();
      r.hash2_value = var.child(ustr_hash2).value();
      r.hash_version = var.child(ustr_hash_version).valueAsInt32();
      r.multi_tag = var.child(ustr_multitag).value();
      r.property = var.child(ustr_property).valueAsInt32();
      variables_info.add(r);
    }
  }
  else {
    // Lecture via les données XML
    XmlNodeList vars = variables_node.children("variable");
    for (const auto& var : vars) {
      VariableReadInfo r;
      r.full_type = var.attrValue(ustr_full_type);
      r.data_type = var.attrValue(ustr_data_type);
      r.base_name = var.attrValue(ustr_base_name);
      r.mesh_name = var.attrValue(ustr_mesh_name);
      r.group_name = var.attrValue(ustr_group_name);
      r.family_name = var.attrValue(ustr_family_name);
      r.hash_value = var.attrValue(ustr_hash);
      r.multi_tag = var.attrValue(ustr_multitag);
      r.property = var.attr(ustr_property).valueAsInteger();
      variables_info.add(r);
    }
  }

  for (const VariableReadInfo& r : variables_info) {
    String full_type = r.full_type;
    VariableDataTypeInfo vdti(full_type);

    // Vérifie que 'data-type' est cohérent avec la valeur dans 'full_type'
    if (vdti.dataTypeName() != r.data_type)
      ARCANE_FATAL("Incoherent value for 'data-type' name v='{0}' expected='{1}'", r.data_type, vdti.dataTypeName());

    String family_name = r.family_name;
    // Actuellement, si la variable n'est pas partielle alors son groupe
    // n'est pas sauvé dans les meta-données. Il faut donc le générer.
    String group_name = r.group_name;
    bool is_partial = vdti.isPartial();
    if (!is_partial) {
      // NOTE: Cette construction doit être cohérente avec celle de
      // DynamicMeshKindInfos. A terme il faudra toujours sauver le nom du groupe
      // dans les meta-données.
      group_name = "All" + family_name + "s";
    }
    auto vmd = vmd_list.add(r.base_name, r.mesh_name, r.family_name, group_name, is_partial);

    vmd->setFullType(full_type);
    vmd->setHash(r.hash_value);
    vmd->setHash2(r.hash2_value);
    vmd->setHashVersion(r.hash_version);
    vmd->setMultiTag(r.multi_tag);
    vmd->setProperty(r.property);

    info(5) << "CHECK VAR: "
            << " base-name=" << vmd->baseName()
            << " mesh-name=" << vmd->meshName()
            << " family-name=" << vmd->itemFamilyName()
            << " full-type=" << vmd->fullType()
            << " name=" << vmd->fullName()
            << " multitag=" << vmd->multiTag()
            << " property=" << vmd->property()
            << " hash=" << vmd->hash();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableIOReaderMng::
_readMeshesMetaData(JSONValue meshes_json, const XmlNode& meshes_node)
{
  ISubDomain* sd = m_variable_mng->subDomain();
  IMeshMng* mesh_mng = sd->meshMng();
  IMeshFactoryMng* mesh_factory_mng = mesh_mng->meshFactoryMng();

  struct MeshInfo
  {
    String name;
    String factory_name;
    bool is_sequential;
  };
  UniqueArray<MeshInfo> meshes_info;

  // Lit les informations des maillages à partir des données JSON
  // si ces dernières existent.
  if (!meshes_json.null()) {
    // Déclare la liste ici pour éviter de retourner un temporaire dans 'for-range'
    JSONValueList vars = meshes_json.valueAsArray();
    for (const JSONValue& var : vars) {
      String mesh_name = var.expectedChild("name").value();
      String mesh_factory_name = var.child("factory-name").value();
      bool is_sequential = false;
      JSONValue v = var.child("sequential");
      if (!v.null())
        is_sequential = v.valueAsBool();
      meshes_info.add({ mesh_name, mesh_factory_name, is_sequential });
    }
  }
  else {
    XmlNodeList meshes = meshes_node.children("mesh");
    for (XmlNode var : meshes) {
      String mesh_name = var.attrValue("name");
      String mesh_factory_name = var.attrValue("factory-name");
      bool is_sequential = var.attr("sequential", false).valueAsBoolean();
      meshes_info.add({ mesh_name, mesh_factory_name, is_sequential });
    }
  }

  for (const MeshInfo& mesh_info : meshes_info) {
    String mesh_name = mesh_info.name;
    String mesh_factory_name = mesh_info.factory_name;
    MeshHandle* mesh_handle = mesh_mng->findMeshHandle(mesh_name, false);
    IMesh* mesh = (mesh_handle) ? mesh_handle->mesh() : nullptr;
    if (mesh)
      continue;
    bool is_sequential = mesh_info.is_sequential;
    info() << "Creating from checkpoint mesh='" << mesh_name
           << "' sequential?=" << is_sequential
           << " factory=" << mesh_factory_name;
    // Depuis avril 2020, l'attribut 'factory-name' doit être présent
    // et sa valeur non nulle.
    if (mesh_factory_name.null())
      ARCANE_FATAL("No attribute 'factory-name' for mesh");

    {
      MeshBuildInfo mbi(mesh_name);
      mbi.addFactoryName(mesh_factory_name);
      IParallelMng* mesh_pm = m_variable_mng->m_parallel_mng;
      if (is_sequential)
        mesh_pm = mesh_pm->sequentialParallelMng();
      mbi.addParallelMng(Arccore::makeRef(mesh_pm));
      mesh_factory_mng->createMesh(mbi);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableIOReaderMng::
_buildFilteredVariableList(VariableReaderMng& var_read_mng, IVariableFilter* filter)
{
  VariableMetaDataList& vmd_list = var_read_mng.variableMetaDataList();
  for (const auto& i : m_variable_mng->m_full_name_variable_map) {
    IVariable* var = i.second;
    bool apply_me = true;
    if (filter)
      apply_me = filter->applyFilter(*var);
    info(5) << "Read variable name=" << var->fullName() << " filter=" << apply_me;
    if (apply_me) {
      Ref<VariableMetaData> vmd = var->createMetaDataRef();
      vmd_list.add(vmd);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableIOReaderMng::
_readVariablesData(VariableReaderMng& var_read_mng, IDataReaderWrapper* reader)
{
  var_read_mng.readVariablesData(m_variable_mng, reader);
  _finalizeReadVariables(var_read_mng.variablesToRead());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableIOReaderMng::
_finalizeReadVariables(const VariableList& vars_to_read)
{
  ARCANE_UNUSED(vars_to_read);

  info(4) << "VariableMng: _finalizeReadVariables()";

  // Resynchronise en lecture les valeurs de toutes les variables pour
  // être sur que les références sont toujours correctes (en cas de
  // réallocation mémoire).
  // NOTE: en théorie cela ne doit pas être utile car IVariable::notifyEndRead()
  // se charge de faire cela.
  // NOTE: de plus, il n'est nécessaire de le faire que sur les variables
  // de \a vars_to_read.
  for (const auto& i : m_variable_mng->m_full_name_variable_map)
    i.second->syncReferences();

  // Notifie les observateurs qu'une lecture vient d'être faite.
  m_variable_mng->m_read_observable->notifyAllObservers();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
