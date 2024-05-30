// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshMaterialVariable.cc                                     (C) 2000-2024 */
/*                                                                           */
/* Variable sur un matériau du maillage.                                     */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/materials/MeshMaterialVariable.h"

#include "arcane/utils/NotImplementedException.h"
#include "arcane/utils/TraceInfo.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/Real2.h"
#include "arcane/utils/Real3.h"
#include "arcane/utils/Real2x2.h"
#include "arcane/utils/Real3x3.h"
#include "arcane/utils/Mutex.h"
#include "arcane/utils/MemoryUtils.h"
#include "arcane/utils/PlatformUtils.h"

#include "arcane/core/materials/IMeshMaterial.h"
#include "arcane/core/materials/ComponentItemVectorView.h"
#include "arcane/core/Variable.h"
#include "arcane/core/VariableDependInfo.h"
#include "arcane/core/MeshVariable.h"
#include "arcane/core/IItemFamily.h"
#include "arcane/core/IMesh.h"
#include "arcane/core/IObserver.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/Parallel.h"
#include "arcane/core/materials/internal/IMeshComponentInternal.h"
#include "arcane/core/materials/internal/IMeshMaterialMngInternal.h"

#include "arcane/materials/MaterialVariableBuildInfo.h"
#include "arcane/materials/MatItemEnumerator.h"
#include "arcane/materials/MeshMaterialVariableRef.h"
#include "arcane/materials/MeshMaterialVariableDependInfo.h"
#include "arcane/materials/IMeshMaterialVariableComputeFunction.h"
#include "arcane/materials/IMeshMaterialVariableSynchronizer.h"
#include "arcane/materials/internal/MeshMaterialVariablePrivate.h"
#include "arcane/materials/internal/MeshMaterialVariableIndexer.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshMaterialVariablePrivate::
MeshMaterialVariablePrivate(const MaterialVariableBuildInfo& v,MatVarSpace mvs,
                            MeshMaterialVariable* variable)
: m_name(v.name())
, m_material_mng(v.materialMng())
, m_var_space(mvs)
, m_variable(variable)
{
 // Pour test uniquement
 if (!platform::getEnvironmentVariable("ARCANE_NO_RECURSIVE_DEPEND").null())
   m_has_recursive_depend = false;
} 

 /*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshMaterialVariablePrivate::
~MeshMaterialVariablePrivate()
{
  if (m_global_variable_changed_observer)
    std::cerr << "WARNING: MeshMaterialVariablePrivate: in destructor: observer is not destroyed\n";
}
 
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 MeshMaterialVariablePrivate::
dataTypeSize() const
{
  return m_variable->dataTypeSize();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialVariablePrivate::
copyToBuffer(SmallSpan<const MatVarIndex> matvar_indexes,
             Span<std::byte> bytes, RunQueue* queue) const
{
  m_variable->_copyToBuffer(matvar_indexes,bytes,queue);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialVariablePrivate::
copyFromBuffer(SmallSpan<const MatVarIndex> matvar_indexes,
               Span<const std::byte> bytes, RunQueue* queue)
{
  m_variable->_copyFromBuffer(matvar_indexes,bytes,queue);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Ref<IData> MeshMaterialVariablePrivate::
internalCreateSaveDataRef(Integer nb_value)
{
  return m_variable->_internalCreateSaveDataRef(nb_value);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialVariablePrivate::
saveData(IMeshComponent* component,IData* data)
{
  m_variable->_saveData(component,data);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialVariablePrivate::
restoreData(IMeshComponent* component,IData* data,Integer data_index,
            Int32ConstArrayView ids,bool allow_null_id)
{
  m_variable->_restoreData(component,data,data_index,ids,allow_null_id);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialVariablePrivate::
copyGlobalToPartial(const CopyBetweenPartialAndGlobalArgs& args)
{
  m_variable->_copyGlobalToPartial(args);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialVariablePrivate::
copyPartialToGlobal(const CopyBetweenPartialAndGlobalArgs& args)
{
  m_variable->_copyPartialToGlobal(args);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialVariablePrivate::
initializeNewItems(const ComponentItemListBuilder& list_builder, RunQueue& queue)
{
  m_variable->_initializeNewItems(list_builder, queue);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialVariablePrivate::
syncReferences(bool check_resize)
{
  m_variable->_syncReferences(check_resize);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialVariablePrivate::
resizeForIndexer(Int32 index, RunQueue& queue)
{
  m_variable->_resizeForIndexer(index, queue);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshMaterialVariable::
MeshMaterialVariable(const MaterialVariableBuildInfo& v,MatVarSpace mvs)
: m_p(new MeshMaterialVariablePrivate(v,mvs,this))
, m_views_as_bytes(MemoryUtils::getAllocatorForMostlyReadOnlyData())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshMaterialVariable::
~MeshMaterialVariable()
{
  delete m_p;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialVariable::
incrementReference()
{
  // Cette méthode ne doit être appelé que par le constructeur de recopie
  // d'une référence. Pour les autres cas, le compteur de référence est incrémenté
  // automatiquement.
  // TODO: regarder si utiliser un AtomicInt32 pour le compteur de référence est
  // préférable ce qui permettrait de supprimer le verrou.
  Mutex::ScopedLock sl(m_p->materialMng()->variableLock());
  ++m_p->m_nb_reference;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialVariable::
addVariableRef(MeshMaterialVariableRef* ref)
{
  Mutex::ScopedLock sl(m_p->materialMng()->variableLock());
  // L'incrément de m_p->m_nb_reference est fait dans getReference()
  ref->setNextReference(m_p->m_first_reference);
  if (m_p->m_first_reference){
    MeshMaterialVariableRef* _list = m_p->m_first_reference;
    if (_list->previousReference())
      _list->previousReference()->setNextReference(ref);
    _list->setPreviousReference(ref);
  }
  else{
    ref->setPreviousReference(nullptr);
  }
  m_p->m_first_reference = ref;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialVariable::
removeVariableRef(MeshMaterialVariableRef* ref)
{
  Mutex::ScopedLock sl(m_p->materialMng()->variableLock());

  MeshMaterialVariableRef* tmp = ref;
  if (tmp->previousReference())
    tmp->previousReference()->setNextReference(tmp->nextReference());
  if (tmp->nextReference())
    tmp->nextReference()->setPreviousReference(tmp->previousReference());
  if (m_p->m_first_reference==tmp)
    m_p->m_first_reference = m_p->m_first_reference->nextReference();

  // La référence peut être utilisée par la suite donc il ne faut pas oublier
  // de supprimer le précédent et le suivant.
  ref->setNextReference(nullptr);
  ref->setPreviousReference(nullptr);

  Int32 nb_ref = --m_p->m_nb_reference;

  // Vérifie que le nombre de références est valide.
  // En cas d'erreur, on ne peut rien afficher, car il est possible que m_p ait
  // déjà été détruit.
  if (nb_ref<0)
    ARCANE_FATAL("Invalid reference number for variable");

  // Lorsqu'il n'y a plus de références sur cette variable, le signale au
  // gestionnaire de variable et se détruit
  if (nb_ref==0){
    // Attention : il faut d'abord détruire l'observable,
    // car removeVariable() peut détruire la variable globale s'il n'y
    // a plus de références dessus et comme il restera un observateur dessus,
    // cela provoquera une fuite mémoire car l'observable associé ne sera
    // pas détruit.
    delete m_p->m_global_variable_changed_observer;
    m_p->m_global_variable_changed_observer = nullptr;
    m_p->materialMng()->_internalApi()->removeVariable(this);
    delete this;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshMaterialVariableRef* MeshMaterialVariable::
firstReference() const
{
  return m_p->m_first_reference;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String MeshMaterialVariable::
name() const
{
  return m_p->name();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IVariable* MeshMaterialVariable::
materialVariable(IMeshMaterial* mat)
{
  Int32 index = mat->_internalApi()->variableIndexer()->index() + 1;
  return m_p->m_refs[index]->variable();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialVariable::
setKeepOnChange(bool v)
{
  m_p->m_keep_on_change = v;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool MeshMaterialVariable::
keepOnChange() const
{
  return  m_p->m_keep_on_change;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialVariable::
update(IMeshMaterial* mat)
{
  if (m_p->hasRecursiveDepend()){
    for( VariableDependInfo& vdi : m_p->m_depends ){
      vdi.variable()->update();
    }

    for( MeshMaterialVariableDependInfo& vdi : m_p->m_mat_depends ){
      vdi.variable()->update(mat);
    }
  }
  Int32 mat_id = mat->id();

  bool need_update = false;
  Int64 modified_time = m_p->m_modified_times[mat_id];
  for( VariableDependInfo& vdi : m_p->m_depends ){
    Int64 mt = vdi.variable()->modifiedTime();
    if (mt>modified_time){
      need_update = true;
      break;
    }
  }
  if (!need_update){
    for( MeshMaterialVariableDependInfo& vdi : m_p->m_mat_depends ){
      Int64 mt = vdi.variable()->modifiedTime(mat);
      if (mt>modified_time){
        need_update = true;
        break;
      }
    }
  }

  if (need_update){
    IMeshMaterialVariableComputeFunction* cf = m_p->m_compute_function.get();
    if (cf){
      cf->execute(mat);
    }
    else{
      ARCANE_FATAL("no compute function for variable '{0}'",name());
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialVariable::
setUpToDate(IMeshMaterial* mat)
{
  Int32 id = mat->id();
  m_p->m_modified_times[id] = IVariable::incrementModifiedTime();
}

Int64 MeshMaterialVariable::
modifiedTime(IMeshMaterial* mat)
{
  Int32 id = mat->id();
  return m_p->m_modified_times[id];
}

void MeshMaterialVariable::
addDepend(IMeshMaterialVariable* var)
{
  m_p->m_mat_depends.add(MeshMaterialVariableDependInfo(var,TraceInfo()));
}

void MeshMaterialVariable::
addDepend(IMeshMaterialVariable* var,const TraceInfo& tinfo)
{
  m_p->m_mat_depends.add(MeshMaterialVariableDependInfo(var,tinfo));
}

void MeshMaterialVariable::
removeDepend(IMeshMaterialVariable* var)
{
  ARCANE_UNUSED(var);
  throw NotImplementedException(A_FUNCINFO);
}

void MeshMaterialVariable::
addDepend(IVariable* var)
{
  m_p->m_depends.add(VariableDependInfo(var,IVariable::DPT_CurrentTime,TraceInfo()));
}

void MeshMaterialVariable::
addDepend(IVariable* var,const TraceInfo& tinfo)
{
  m_p->m_depends.add(VariableDependInfo(var,IVariable::DPT_CurrentTime,tinfo));
}

void MeshMaterialVariable::
removeDepend(IVariable* var)
{
  ARCANE_UNUSED(var);
  throw NotImplementedException(A_FUNCINFO);
}

void MeshMaterialVariable::
setComputeFunction(IMeshMaterialVariableComputeFunction* v)
{  
  m_p->m_compute_function = v;
}

IMeshMaterialVariableComputeFunction* MeshMaterialVariable::
computeFunction()
{
  return m_p->m_compute_function.get();
}

void MeshMaterialVariable::
dependInfos(Array<VariableDependInfo>& infos,
            Array<MeshMaterialVariableDependInfo>& mat_infos)
{
  for( VariableDependInfo& vdi : m_p->m_depends ){
    infos.add(vdi);
  }

  for( MeshMaterialVariableDependInfo& vdi : m_p->m_mat_depends ){
    mat_infos.add(vdi);
  }
}

ITraceMng* MeshMaterialVariable::
_traceMng() const
{
  return m_p->materialMng()->traceMng();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MatVarSpace MeshMaterialVariable::
space() const
{
  return m_p->space();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Convertit une vue de MatVarIndex en vue de Int32.
 *
 * Considère qu'un MatVarIndex contient 2 Int32.
 */
SmallSpan<const Int32> MeshMaterialVariable::
_toInt32Indexes(SmallSpan<const MatVarIndex> indexes)
{
  static_assert(sizeof(MatVarIndex)==2*sizeof(Int32),"Bad size for MatVarIndex");
  auto* ptr = reinterpret_cast<const Int32*>(indexes.data());
  return { ptr, indexes.size()*2 };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialVariable::
_copyToBuffer(SmallSpan<const MatVarIndex> matvar_indexes,
              Span<std::byte> bytes, RunQueue* queue) const
{
  const Integer one_data_size = dataTypeSize();
  SmallSpan<const Int32> indexes(_toInt32Indexes(matvar_indexes));
  const Int32 nb_item = matvar_indexes.size();
  MutableMemoryView destination_buffer(makeMutableMemoryView(bytes.data(),one_data_size,nb_item));
  ConstMultiMemoryView source_view(m_views_as_bytes.view(),one_data_size);
  source_view.copyToIndexes(destination_buffer,indexes,queue);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialVariable::
_copyFromBuffer(SmallSpan<const MatVarIndex> matvar_indexes,
                Span<const std::byte> bytes, RunQueue* queue)
{
  const Int32 one_data_size = dataTypeSize();
  SmallSpan<const Int32> indexes(_toInt32Indexes(matvar_indexes));
  const Int32 nb_item = matvar_indexes.size();
  MutableMultiMemoryView destination_view(m_views_as_bytes.view(),one_data_size);
  ConstMemoryView source_buffer(makeConstMemoryView(bytes.data(),one_data_size,nb_item));
  destination_view.copyFromIndexes(source_buffer,indexes,queue);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IMeshMaterialVariableInternal* MeshMaterialVariable::
_internalApi()
{
  return m_p->_internalApi();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
