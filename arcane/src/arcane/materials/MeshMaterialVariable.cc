// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshMaterialVariable.cc                                     (C) 2000-2025 */
/*                                                                           */
/* Variable on a mesh material.                                              */
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

#include "arcane/accelerator/RunCommandLoop.h"

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
 // For testing only
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
copyBetweenPartialAndGlobal(const CopyBetweenPartialAndGlobalArgs& args)
{
  m_variable->_copyBetweenPartialAndGlobal(args);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialVariablePrivate::
initializeNewItemsWithZero(InitializeWithZeroArgs& args)
{
  m_variable->_initializeNewItemsWithZero(args);
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
resizeForIndexer(ResizeVariableIndexerArgs& args)
{
  m_variable->_resizeForIndexer(args);
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
  // This method should only be called by the copy constructor
  // of a reference. In other cases, the reference counter is incremented
  // automatically.
  // TODO: check if using an AtomicInt32 for the reference counter is
  // preferable, which would allow removing the lock.
  Mutex::ScopedLock sl(m_p->materialMng()->variableLock());
  ++m_p->m_nb_reference;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialVariable::
addVariableRef(MeshMaterialVariableRef* ref)
{
  Mutex::ScopedLock sl(m_p->materialMng()->variableLock());
  // The increment of m_p->m_nb_reference is done in getReference()
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

  // The reference may be used later, so we must not forget
  // to remove the previous and next.
  ref->setNextReference(nullptr);
  ref->setPreviousReference(nullptr);

  Int32 nb_ref = --m_p->m_nb_reference;

  // Checks that the number of references is valid.
  // In case of an error, we cannot display anything, because it is possible that m_p has
  // already been destroyed.
  if (nb_ref<0)
    ARCANE_FATAL("Invalid reference number for variable");

  // When there are no more references on this variable, it signals to the
  // variable manager and destroys itself
  if (nb_ref==0){
    // Warning: the observer must first be destroyed,
    // because removeVariable() may destroy the global variable if there are no
    // more references on it and since an observer will remain on it,
    // this will cause a memory leak because the associated observer will not be
    // destroyed.
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
 * \brief Converts a view of MatVarIndex to a view of Int32.
 *
 * Assumes that a MatVarIndex contains 2 Int32.
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
  MemoryUtils::copyWithIndexedSource(destination_buffer,source_view,indexes,queue);
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
  MemoryUtils::copyWithIndexedDestination(destination_view, source_buffer,indexes,queue);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialVariable::
_genericCopyTo(Span<const std::byte> input,
               SmallSpan<const Int32> input_indexes,
               Span<std::byte> output,
               SmallSpan<const Int32> output_indexes,
               const RunQueue& queue, Int32 data_type_size)
{
  // TODO: verify identical index sizes
  Integer nb_value = input_indexes.size();
  auto command = makeCommand(queue);

  ARCANE_CHECK_ACCESSIBLE_POINTER(queue, output.data());
  ARCANE_CHECK_ACCESSIBLE_POINTER(queue, input.data());
  ARCANE_CHECK_ACCESSIBLE_POINTER(queue, input_indexes.data());
  ARCANE_CHECK_ACCESSIBLE_POINTER(queue, output_indexes.data());
  const Int32 dim2_size = data_type_size;

  command << RUNCOMMAND_LOOP1(iter, nb_value)
  {
    auto [i] = iter();

    Int32 output_base = output_indexes[i] * dim2_size;
    Int32 input_base = input_indexes[i] * dim2_size;
    for (Int32 j = 0; j < dim2_size; ++j)
      output[output_base + j] = input[input_base + j];
  };
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
