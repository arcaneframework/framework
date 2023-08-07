// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshMaterialBackup.cc                                       (C) 2000-2023 */
/*                                                                           */
/* Sauvegarde/restauration des valeurs des matériaux et milieux.             */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/IDataCompressor.h"

#include "arcane/core/IVariable.h"
#include "arcane/core/IData.h"
#include "arcane/core/IMesh.h"
#include "arcane/core/IItemFamily.h"
#include "arcane/core/ServiceBuilder.h"
#include "arcane/core/internal/IDataInternal.h"

#include "arcane/materials/MeshMaterialBackup.h"
#include "arcane/materials/IMeshMaterialMng.h"
#include "arcane/materials/IMeshMaterialVariable.h"
#include "arcane/materials/MatItemEnumerator.h"

#include "arcane/materials/internal/MeshMaterialMng.h"
#include "arcane/materials/internal/MeshMaterialVariableIndexer.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

struct MeshMaterialBackup::VarData
{
 public:

  VarData() = default;
  explicit VarData(Ref<IData> d) : data(d) {}

 public:

  Ref<IData> data;
  Integer data_index = 0;
  DataCompressionBuffer m_data_buffer;
  Ref<IDataCompressor> m_compressor;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshMaterialBackup::
MeshMaterialBackup(IMeshMaterialMng* mm,bool use_unique_ids)
: TraceAccessor(mm->traceMng())
, m_material_mng(mm)
, m_use_unique_ids(use_unique_ids)
{
  m_compressor_service_name = mm->dataCompressorServiceName();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshMaterialBackup::
~MeshMaterialBackup()
{
  for( const auto& iter : m_saved_data )
    delete iter.second;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialBackup::
saveValues()
{
  _save();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialBackup::
restoreValues()
{
  _restore();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialBackup::
setCompressorServiceName(const String& name)
{
  m_compressor_service_name = name;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Indique si la variable \a var est définie sur le composant \a component.
 */
bool MeshMaterialBackup::
_isValidComponent(IMeshMaterialVariable* var,IMeshComponent* component)
{
  MatVarSpace mvs = var->space();
  if (mvs==MatVarSpace::MaterialAndEnvironment)
    return true;
  if (mvs==MatVarSpace::Environment && component->isEnvironment())
    return true;
  return false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialBackup::
_save()
{
  if (!m_compressor_service_name.null())
    m_use_v2 = true;

  ConstArrayView<MeshMaterialVariableIndexer*> indexers = m_material_mng->variablesIndexer();

  Integer nb_index = indexers.size();
  Integer nb_value = 0;
  for( Integer i=0; i<nb_index; ++i )
    nb_value += indexers[i]->cells().size();

  info(4) << "NB_EXPECTED_VALUE=" << nb_value;

  MeshMaterialMng* mm = dynamic_cast<MeshMaterialMng*>(m_material_mng);
  if (!mm)
    ARCANE_FATAL("Can not cast to MeshMaterialMng");

  // Stocke dans \a vars la liste des variables pour accès plus simple qu'avec la map
  Integer max_nb_var = arcaneCheckArraySize(mm->m_full_name_variable_map.size());
  m_vars.reserve(max_nb_var);
  for( const auto& i : mm->m_full_name_variable_map){
    IMeshMaterialVariable* mv = i.second;
    if (mv->keepOnChange() && mv->globalVariable()->isUsed())
      m_vars.add(mv);
  }
  for( IMeshMaterialVariable* mv : m_vars ){
    info(4) << "SAVE MVAR=" << mv->name() << " is_used?=" << mv->globalVariable()->isUsed();
    VarData* vd = new VarData(mv->_internalCreateSaveDataRef(nb_value));
    m_saved_data.insert(std::make_pair(mv,vd));
  }

  if (m_use_v2)
    _saveV2();
  else
    _saveV1();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialBackup::
_saveV1()
{
  ENUMERATE_COMPONENT(ic,m_material_mng->components()){
    IMeshComponent* c = *ic;
    _saveIds(c);
    for( IMeshMaterialVariable* var : m_vars ){
      if (_isValidComponent(var,c))
        var->_saveData(c,m_saved_data[var]->data.get());
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialBackup::
_saveV2()
{
  IMesh* mesh = m_material_mng->mesh();
  ServiceBuilder<IDataCompressor> sb(mesh->handle().application());
  Ref<IDataCompressor> compressor_ref;
  if (!m_compressor_service_name.empty())
    compressor_ref = sb.createReference(m_compressor_service_name);
  IDataCompressor* compressor = compressor_ref.get();
  auto components = m_material_mng->components();

  ENUMERATE_COMPONENT(ic,components){
    IMeshComponent* c = *ic;
    _saveIds(c);
  }

  for( IMeshMaterialVariable* var : m_vars ){
    VarData* var_data = m_saved_data[var];
    IData* saved_data = var_data->data.get();
    if (compressor){
      var_data->m_compressor = compressor_ref;
      var_data->m_data_buffer.m_compressor = compressor;
    }
    ENUMERATE_COMPONENT(ic,components){
      IMeshComponent* c = *ic;
      if (_isValidComponent(var,c)){
        var->_saveData(c,saved_data);
      }
    }
    if (compressor){
      IDataInternal* d = saved_data->_commonInternal();
      d->compressAndClear(var_data->m_data_buffer);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialBackup::
_saveIds(IMeshComponent* component)
{
  if (m_use_unique_ids){
    auto& ids = m_unique_ids_array[component];
    ENUMERATE_COMPONENTCELL(icell,component){
      ComponentCell ec = *icell;
      ids.add(ec.globalCell().uniqueId());
    }
    info(4) << "SAVE (uid) for component name=" << component->name()
            << " nb=" << ids.size();
  }
  else{
    Int32Array& ids = m_ids_array[component];
    ENUMERATE_COMPONENTCELL(icell,component){
      ComponentCell ec = *icell;
      ids.add(ec.globalCell().localId());
    }
    info(4) << "SAVE (lid) for component name=" << component->name()
            << " nb=" << ids.size();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialBackup::
_restore()
{
  if (m_use_unique_ids){
    info(4) << "RESTORE using uniqueIds()";
    IItemFamily* cell_family = m_material_mng->mesh()->cellFamily();
    // Si on utilise les uniqueId(), le tableau m_unique_ids_array
    // contient les valeurs des uniqueId() des mailles. Il faut
    // ensuite le convertir en localId().
    for( const auto& iter : m_unique_ids_array ){
      IMeshComponent* component = iter.first;
      auto& unique_ids = m_unique_ids_array[component];
      auto& local_ids = m_ids_array[component];
      local_ids.resize(unique_ids.size());
      cell_family->itemsUniqueIdToLocalId(local_ids,unique_ids,false);
    }
  }

  if (m_use_v2)
    _restoreV2();
  else
    _restoreV1();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialBackup::
_restoreV1()
{
  // Si on utilise les uniqueId(), alors les localId() peuvent être nuls
  // si on a supprimé des mailles.
  bool allow_null_id = m_use_unique_ids;

  ENUMERATE_COMPONENT(ic,m_material_mng->components()){
    IMeshComponent* c = *ic;
    Int32ConstArrayView ids = m_ids_array[c];
    info(4) << "RESTORE for component name=" << c->name() << " nb=" << ids.size();
    for( IMeshMaterialVariable* var : m_vars ){
      VarData* vd = m_saved_data[var];
      if (_isValidComponent(var,c)){
        var->_restoreData(c,vd->data.get(),vd->data_index,ids,allow_null_id);
        vd->data_index += ids.size();
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialBackup::
_restoreV2()
{
  // Si on utilise les uniqueId(), alors les localId() peuvent être nuls
  // si on a supprimé des mailles.
  bool allow_null_id = m_use_unique_ids;

  auto components = m_material_mng->components();

  for( IMeshMaterialVariable* var : m_vars ){
    VarData* vd = m_saved_data[var];
    // Décompresse les données si nécessaire
    IDataCompressor* compressor = vd->m_data_buffer.m_compressor;
    if (compressor){
      info(5) << "RESTORE decompress variable name=" << var->name();
      IDataInternal* d = vd->data->_commonInternal();
      d->decompressAndFill(vd->m_data_buffer);
    }
    info(4) << "RESTORE for variable name=" << var->name();
    ENUMERATE_COMPONENT(ic,components){
      IMeshComponent* c = *ic;
      Int32ConstArrayView ids = m_ids_array[c];
      if (_isValidComponent(var,c)){
        var->_restoreData(c,vd->data.get(),vd->data_index,ids,allow_null_id);
        vd->data_index += ids.size();
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
