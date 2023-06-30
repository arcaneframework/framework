// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VariableIOMng.cc                                            (C) 2000-2023 */
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

#include "arcane/core/ISubDomain.h"
#include "arcane/core/IParallelMng.h"

// TODO: gérer le hash en version 64 bits.

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VariableIOMng::
VariableIOMng(VariableMng* vm)
: TraceAccessor(vm->traceMng())
, m_variable_mng(vm)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableIOMng::
writeCheckpoint(ICheckpointWriter* service)
{
  if (!service)
    ARCANE_FATAL("No protection service specified");

  Trace::Setter mci(traceMng(),_msgClassName());

  Timer::Phase tp(m_variable_mng->m_time_stats,TP_InputOutput);

  CheckpointSaveFilter save_filter;

  service->notifyBeginWrite();
  IDataWriter* data_writer = service->dataWriter();
  if (!data_writer)
    ARCANE_FATAL("no writer() nor dataWriter()");
  writeVariables(data_writer,&save_filter,true);
  service->notifyEndWrite();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableIOMng::
writePostProcessing(IPostProcessorWriter* post_processor)
{
  Trace::Setter mci(traceMng(),_msgClassName());

  if (!post_processor)
    ARCANE_FATAL("No post-processing service specified");

  Timer::Phase tp(m_variable_mng->m_time_stats,TP_InputOutput);

  post_processor->notifyBeginWrite();
  VariableCollection variables(post_processor->variables());
  IDataWriter* data_writer = post_processor->dataWriter();
  if (!data_writer)
    ARCANE_FATAL("no writer() nor dataWriter()");
  writeVariables(data_writer,variables,false);
  post_processor->notifyEndWrite();
}


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableIOMng::
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
void VariableIOMng::
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

void VariableIOMng::
_generateVariablesMetaData(XmlNode variables_node, const VariableCollection& vars, bool use_hash)
{
  StringBuilder var_full_type_b;
  ByteUniqueArray hash_values;
  MD5HashAlgorithm hash_algo;

  for (VariableCollection::Enumerator i(vars); ++i;) {
    IVariable* var = *i;
    ScopedPtrT<VariableMetaData> vmd(var->createMetaData());
    String var_full_type = vmd->fullType();
    String var_family_name = var->itemFamilyName();
    String var_mesh_name = var->meshName();
    XmlNode var_node = XmlElement(variables_node, "variable");
    var_node.setAttrValue("base-name", var->name());
    if (!var_family_name.null())
      var_node.setAttrValue("item-family-name", var_family_name);
    if (var->isPartial())
      var_node.setAttrValue("item-group-name", var->itemGroupName());
    if (!var_mesh_name.null()) {
      var_node.setAttrValue("mesh-name", var_mesh_name);
    }
    var_node.setAttrValue("full-type", var_full_type);
    var_node.setAttrValue("data-type", dataTypeName(var->dataType()));
    var_node.setAttrValue("dimension", String::fromNumber(var->dimension()));
    var_node.setAttrValue("multi-tag", String::fromNumber(var->multiTag()));
    var_node.setAttrValue("property", String::fromNumber(var->property()));
    if (use_hash) {
      hash_values.clear();
      var->data()->computeHash(&hash_algo, hash_values);
      String hash_str = Convert::toHexaString(hash_values);
      var_node.setAttrValue("hash", hash_str);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * TEMPORAIRE.
 * TODO Cela doit normalement être fait via le 'IMeshMng'.
 */
void VariableIOMng::
_generateMeshesMetaData(XmlNode meshes_node)
{
  // Positionne un numéro de version pour compatibilité avec de futures versions
  meshes_node.setAttrValue("version", "1");
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
      XmlNode mesh_node = XmlElement(meshes_node, "mesh");
      mesh_node.setAttrValue("name", mesh->name());
      mesh_node.setAttrValue("factory-name", mesh->factoryName());
      // Indique si le maillage utilise le gestionnaire de parallélisme
      // séquentiel car dans ce cas en reprise il faut le créer avec le
      // même gestionnaire.
      // TODO: il faudrait traiter les cas où un maillage est créé
      // avec un IParallelMng qui n'est ni séquentiel, ni celui du
      // sous-domaine.
      if (is_parallel && mesh->parallelMng() == seq_pm)
        mesh_node.setAttrValue("sequential", "true");
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String VariableIOMng::
_generateMetaData(const VariableCollection& vars, bool use_hash)
{
  ScopedPtrT<IXmlDocumentHolder> doc(domutils::createXmlDocument());
  XmlNode doc_node = doc->documentNode();
  XmlElement root_element(doc_node, "arcane-checkpoint-metadata");
  XmlElement variables_node(root_element, "variables");
  _generateVariablesMetaData(variables_node, vars, use_hash);
  XmlElement meshes_node(root_element, "meshes");
  _generateMeshesMetaData(meshes_node);
  String s = doc->save();
  info(6) << "META_DATA=" << s;
  return s;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableIOMng::
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

  String meta_data = _generateMetaData(vars, use_hash);
  writer->setMetaData(meta_data);

  for (VariableCollection::Enumerator i(vars); ++i;) {
    IVariable* var = *i;
    if (!var->isUsed())
      continue;
    try {
      writer->write(var, var->data());
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
