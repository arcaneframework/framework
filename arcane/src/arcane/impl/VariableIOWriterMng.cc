// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VariableIOWriterMng.cc                                      (C) 2000-2024 */
/*                                                                           */
/* Classe gérant les entrées/sorties pour les variables.                     */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/impl/internal/VariableMng.h"

#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/IObservable.h"
#include "arcane/utils/ScopedPtr.h"
#include "arcane/utils/StringBuilder.h"
#include "arcane/utils/MD5HashAlgorithm.h"
#include "arcane/utils/SHA1HashAlgorithm.h"
#include "arcane/utils/JSONWriter.h"
#include "arcane/utils/ValueConvert.h"

#include "arcane/core/IDataWriter.h"
#include "arcane/core/IXmlDocumentHolder.h"
#include "arcane/core/DomUtils.h"
#include "arcane/core/XmlNode.h"
#include "arcane/core/VariableMetaData.h"
#include "arcane/core/IData.h"
#include "arcane/core/IMesh.h"
#include "arcane/core/Properties.h"
#include "arcane/core/Timer.h"
#include "arcane/core/ICheckpointWriter.h"
#include "arcane/core/IPostProcessorWriter.h"
#include "arcane/core/internal/IDataInternal.h"

#include "arcane/core/ISubDomain.h"
#include "arcane/core/IParallelMng.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VariableIOWriterMng::
VariableIOWriterMng(VariableMng* vm)
: TraceAccessor(vm->traceMng())
, m_variable_mng(vm)
{
  if (auto v = Convert::Type<Int32>::tryParseFromEnvironment("ARCANE_VARIABLEMNG_HASHV2", true))
    m_use_hash_v2 = (v.value() != 0);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableIOWriterMng::
writeCheckpoint(ICheckpointWriter* service)
{
  if (!service)
    ARCANE_FATAL("No protection service specified");

  Trace::Setter mci(traceMng(), _msgClassName());

  Timer::Phase tp(m_variable_mng->m_time_stats, TP_InputOutput);

  CheckpointSaveFilter save_filter;

  service->notifyBeginWrite();
  IDataWriter* data_writer = service->dataWriter();
  if (!data_writer)
    ARCANE_FATAL("no writer() nor dataWriter()");
  writeVariables(data_writer, &save_filter, true);
  service->notifyEndWrite();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableIOWriterMng::
writePostProcessing(IPostProcessorWriter* post_processor)
{
  Trace::Setter mci(traceMng(), _msgClassName());

  if (!post_processor)
    ARCANE_FATAL("No post-processing service specified");

  Timer::Phase tp(m_variable_mng->m_time_stats, TP_InputOutput);

  post_processor->notifyBeginWrite();
  VariableCollection variables(post_processor->variables());
  IDataWriter* data_writer = post_processor->dataWriter();
  if (!data_writer)
    ARCANE_FATAL("no writer() nor dataWriter()");
  writeVariables(data_writer, variables, false);
  post_processor->notifyEndWrite();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableIOWriterMng::
writeVariables(IDataWriter* writer, IVariableFilter* filter, bool use_hash)
{
  Trace::Setter mci(traceMng(), _msgClassName());

  if (!writer)
    ARCANE_FATAL("No writer available for protection");

  // Calcul la liste des variables à sauver
  VariableList vars;
  for (const auto& i : m_variable_mng->m_full_name_variable_map) {
    IVariable* var = i.second;
    bool apply_var = true;
    if (filter)
      apply_var = filter->applyFilter(*var);
    if (apply_var)
      vars.add(var);
  }
  _writeVariables(writer, vars, use_hash);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \todo prendre en compte le NoDump
 */
void VariableIOWriterMng::
writeVariables(IDataWriter* writer, const VariableCollection& vars, bool use_hash)
{
  if (!writer)
    return;
  if (vars.empty()) {
    VariableList var_array;
    for (const auto& i : m_variable_mng->m_full_name_variable_map) {
      IVariable* var = i.second;
      var_array.add(var);
    }
    _writeVariables(writer, var_array, use_hash);
  }
  else
    _writeVariables(writer, vars, use_hash);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace
{
  void _writeAttribute(JSONWriter& json_writer, XmlNode var_node, const String& name, const String& value)
  {
    var_node.setAttrValue(name, value);
    json_writer.write(name, value);
  }
  void _writeAttribute(JSONWriter& json_writer, XmlNode var_node, const String& name, Int32 value)
  {
    Int64 v = value;
    var_node.setAttrValue(name, String::fromNumber(v));
    json_writer.write(name, v);
  }
  void _writeAttribute(JSONWriter& json_writer, XmlNode var_node, const String& name, bool value)
  {
    var_node.setAttrValue(name, String::fromNumber(value));
    json_writer.write(name, value);
  }

} // namespace

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableIOWriterMng::
_generateVariablesMetaData(JSONWriter& json_writer, XmlNode variables_node,
                           const VariableCollection& vars, IHashAlgorithm* hash_algo)
{
  StringBuilder var_full_type_b;
  ByteUniqueArray hash_values;
  Ref<IHashAlgorithmContext> hash_context;
  if (m_use_hash_v2) {
    SHA1HashAlgorithm sha1_hash_algo;
    hash_context = sha1_hash_algo.createContext();
  }

  json_writer.writeKey("variables");
  json_writer.beginArray();

  for (VariableCollection::Enumerator i(vars); ++i;) {
    JSONWriter::Object o(json_writer);
    IVariable* var = *i;
    Ref<VariableMetaData> vmd(var->createMetaDataRef());
    String var_full_type = vmd->fullType();
    String var_family_name = var->itemFamilyName();
    String var_mesh_name = var->meshName();
    XmlNode var_node = XmlElement(variables_node, "variable");
    _writeAttribute(json_writer, var_node, "base-name", var->name());
    if (!var_family_name.null())
      _writeAttribute(json_writer, var_node, "item-family-name", var_family_name);
    if (var->isPartial())
      _writeAttribute(json_writer, var_node, "item-group-name", var->itemGroupName());
    if (!var_mesh_name.null())
      _writeAttribute(json_writer, var_node, "mesh-name", var_mesh_name);
    _writeAttribute(json_writer, var_node, "full-type", var_full_type);
    _writeAttribute(json_writer, var_node, "data-type", String(dataTypeName(var->dataType())));
    _writeAttribute(json_writer, var_node, "dimension", var->dimension());
    _writeAttribute(json_writer, var_node, "multi-tag", var->multiTag());
    _writeAttribute(json_writer, var_node, "property", var->property());
    if (hash_algo) {
      hash_values.clear();
      var->data()->computeHash(hash_algo, hash_values);
      String hash_str = Convert::toHexaString(hash_values);
      _writeAttribute(json_writer, var_node, "hash", hash_str);
      if (hash_context.get()) {
        hash_context->reset();
        DataHashInfo hash_info(hash_context.get());
        var->data()->_commonInternal()->computeHash(hash_info);
        HashAlgorithmValue hash_value;
        hash_context->computeHashValue(hash_value);
        String hash2_str = Convert::toHexaString(asBytes(hash_value.bytes()));
        info(6) << "Hash=" << hash2_str << " old_hash="
                << hash_str << " name=" << var->name();
        _writeAttribute(json_writer, var_node, "hash2", hash2_str);
        _writeAttribute(json_writer, var_node, "hash-version", hash_info.version());
      }
    }
  }

  json_writer.endArray();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * TEMPORAIRE.
 * TODO Cela doit normalement être fait via le 'IMeshMng'.
 */
void VariableIOWriterMng::
_generateMeshesMetaData(JSONWriter& json_writer, XmlNode meshes_node)
{
  // Positionne un numéro de version pour compatibilité avec de futures versions
  meshes_node.setAttrValue("version", "1");

  json_writer.writeKey("meshes");
  json_writer.beginArray();

  ISubDomain* sd = m_variable_mng->subDomain();
  IParallelMng* pm = m_variable_mng->parallelMng();
  IMesh* default_mesh = sd->defaultMesh();
  bool is_parallel = pm->isParallel();
  IParallelMng* seq_pm = pm->sequentialParallelMng();
  ConstArrayView<IMesh*> meshes = sd->meshes();
  for (Integer i = 0, n = meshes.size(); i < n; ++i) {
    IMesh* mesh = meshes[i];
    bool do_dump = mesh->properties()->getBool("dump");
    // Sauve le maillage s'il est marqué dump ou s'il s'agit du maillage par défaut
    if (do_dump || mesh == default_mesh) {
      JSONWriter::Object o(json_writer);
      XmlNode mesh_node = XmlElement(meshes_node, "mesh");
      _writeAttribute(json_writer, mesh_node, "name", mesh->name());
      _writeAttribute(json_writer, mesh_node, "factory-name", mesh->factoryName());
      // Indique si le maillage utilise le gestionnaire de parallélisme
      // séquentiel car dans ce cas en reprise il faut le créer avec le
      // même gestionnaire.
      // TODO: il faudrait traiter les cas où un maillage est créé
      // avec un IParallelMng qui n'est ni séquentiel, ni celui du
      // sous-domaine.
      if (is_parallel && mesh->parallelMng() == seq_pm)
        _writeAttribute(json_writer, mesh_node, "sequential", true);
    }
  }

  json_writer.endArray();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String VariableIOWriterMng::
_generateMetaData(const VariableCollection& vars, IHashAlgorithm* hash_algo)
{
  JSONWriter json_writer(JSONWriter::FormatFlags::None);

  ScopedPtrT<IXmlDocumentHolder> doc(domutils::createXmlDocument());
  XmlNode doc_node = doc->documentNode();
  XmlElement root_element(doc_node, "arcane-checkpoint-metadata");
  XmlElement variables_node(root_element, "variables");
  {
    JSONWriter::Object o(json_writer);
    JSONWriter::Object o2(json_writer, "arcane-checkpoint-metadata");
    json_writer.write("version", static_cast<Int64>(1));
    if (hash_algo)
      json_writer.write("hash-algorithm-name", hash_algo->name());
    _generateVariablesMetaData(json_writer, variables_node, vars, hash_algo);
    XmlElement meshes_node(root_element, "meshes");
    _generateMeshesMetaData(json_writer, meshes_node);
  }
  {
    // Sauve la sérialisation JSON dans un élément du fichier XML.
    XmlElement json_node(root_element, "json", json_writer.getBuffer());
  }
  String s = doc->save();
  info(6) << "META_DATA=" << s;
  return s;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableIOWriterMng::
_writeVariables(IDataWriter* writer, const VariableCollection& vars, bool use_hash)
{
  if (!writer)
    return;

  m_variable_mng->m_write_observable->notifyAllObservers();
  writer->beginWrite(vars);

  // Appelle la notification de l'écriture des variables
  // Il faut le faire avant de positionner les méta-données
  // car cela autorise de changer la valeur de la variable
  // lors de cet appel.
  for (VariableCollection::Enumerator i(vars); ++i;) {
    IVariable* var = *i;
    if (var->isUsed())
      var->notifyBeginWrite();
  }

  MD5HashAlgorithm md5_hash_algo;
  SHA1HashAlgorithm sha1_hash_algo;
  IHashAlgorithm* hash_algo = &md5_hash_algo;
  if (m_use_hash_v2)
    hash_algo = &sha1_hash_algo;
  if (!use_hash)
    hash_algo = nullptr;
  String meta_data = _generateMetaData(vars, hash_algo);
  writer->setMetaData(meta_data);

  for (VariableCollection::Enumerator i(vars); ++i;) {
    IVariable* var = *i;
    if (!var->isUsed())
      continue;
    try {
      // Avec le paramètre PShMem, si on a des variables NoDump sur certains
      // processus, mais pas d'autres, on doit quand même écrire un tableau
      // vide pour les appels collectifs à la relecture.
      if (var->property() & (IVariable::PNoDump | IVariable::PTemporary)) {
        writer->write(var, var->data()->cloneEmptyRef().get());
      }
      else {
        writer->write(var, var->data());
      }
    }
    catch (const Exception& ex) {
      error() << "Exception Arcane while VariableMng::writeVariables()"
              << " var=" << var->fullName()
              << " exception=" << ex;
      throw;
    }
    catch (const std::exception& ex) {
      error() << "Exception while VariableMng::writeVariables()"
              << " var=" << var->fullName()
              << " exception=" << ex.what();
      throw;
    }
  }
  writer->endWrite();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
