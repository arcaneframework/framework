// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshMaterialVariable.cc                                     (C) 2000-2016 */
/*                                                                           */
/* Variable sur un matériau du maillage.                                     */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/NotImplementedException.h"
#include "arcane/utils/TraceInfo.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/Real2.h"
#include "arcane/utils/Real3.h"
#include "arcane/utils/Real2x2.h"
#include "arcane/utils/Real3x3.h"
#include "arcane/utils/Mutex.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/ScopedPtr.h"

#include "arcane/materials/MeshMaterialVariable.h"
#include "arcane/materials/MeshMaterialVariablePrivate.h"

#include "arcane/materials/MeshMaterialVariableIndexer.h"
#include "arcane/materials/MaterialVariableBuildInfo.h"
#include "arcane/materials/IMeshMaterialMng.h"
#include "arcane/materials/IMeshMaterial.h"
#include "arcane/materials/IMeshEnvironment.h"
#include "arcane/materials/MatItemEnumerator.h"
#include "arcane/materials/MeshMaterialVariableRef.h"
#include "arcane/materials/MeshMaterialVariableDependInfo.h"
#include "arcane/materials/IMeshMaterialVariableComputeFunction.h"
#include "arcane/materials/IMeshMaterialVariableSynchronizer.h"
#include "arcane/materials/MeshMaterialVariableSynchronizerList.h"
#include "arcane/materials/ComponentItemVectorView.h"

//#include "arcane/datatype/DataTypeTraits.h"

#include "arcane/Variable.h"
#include "arcane/VariableDependInfo.h"
#include "arcane/MeshVariable.h"
#include "arcane/IItemFamily.h"
#include "arcane/IMesh.h"
#include "arcane/IObserver.h"
#include "arcane/IParallelMng.h"
#include "arcane/ItemPrinter.h"
#include "arcane/Parallel.h"

#include "arcane/IData.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE
MATERIALS_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshMaterialVariablePrivate::
MeshMaterialVariablePrivate(const MaterialVariableBuildInfo& v,MatVarSpace mvs)
: m_nb_reference(0)
, m_first_reference(0)
, m_name(v.name())
, m_material_mng(v.materialMng())
, m_keep_on_change(true)
, m_global_variable_changed_observer(0)
, m_has_recursive_depend(true)
, m_var_space(mvs)
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

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshMaterialVariable::
MeshMaterialVariable(const MaterialVariableBuildInfo& v,MatVarSpace mvs)
: m_p(new MeshMaterialVariablePrivate(v,mvs))
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
  // L'increment de m_p->m_nb_reference est fait dans getReference()
  ref->setNextReference(m_p->m_first_reference);
  if (m_p->m_first_reference){
    MeshMaterialVariableRef* _list = m_p->m_first_reference;
    if (_list->previousReference())
      _list->previousReference()->setNextReference(ref);
    _list->setPreviousReference(ref);
  }
  else{
    ref->setPreviousReference(0);
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
  ref->setNextReference(0);
  ref->setPreviousReference(0);

  Int32 nb_ref = --m_p->m_nb_reference;

  // Vérifie que le nombre de référence est valide.
  // En cas d'erreur, on ne peut rien afficher car il est possible que m_p ait
  // déjà été détruit.
  if (nb_ref<0)
    throw FatalErrorException(A_FUNCINFO,"Invalid reference number for variable");

  // Lorsqu'il n'y a plus de références sur cette variable, le signale au
  // gestionnaire de variable et se détruit
  if (nb_ref==0){
    // Attention: il faut d'abord détruire l'observable
    // car removeVariable() peut détruire la variable globale s'il n'y
    // a plus de références dessus et comme il restera un observer dessus,
    // cela provoquera une fuite mémoire car l'observable associé ne sera
    // pas détruit.
    delete m_p->m_global_variable_changed_observer;
    m_p->m_global_variable_changed_observer = nullptr;
    m_p->materialMng()->removeVariable(this);
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

const String& MeshMaterialVariable::
name() const
{
  return m_p->name();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IVariable* MeshMaterialVariable::
materialVariable(IMeshMaterial* mat)
{
  Int32 index = mat->variableIndexer()->index() + 1;
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
    for( Integer k=0,n=m_p->m_depends.size(); k<n; ++k ){
      VariableDependInfo& vdi = m_p->m_depends[k];
      vdi.variable()->update();
    }

    for( Integer k=0,n=m_p->m_mat_depends.size(); k<n; ++k ){
      MeshMaterialVariableDependInfo& vdi = m_p->m_mat_depends[k];
      vdi.variable()->update(mat);
    }
  }
  Int32 mat_id = mat->id();

  bool need_update = false;
  Int64 modified_time = m_p->m_modified_times[mat_id];
  for( Integer k=0,n=m_p->m_depends.size(); k<n; ++k ){
    VariableDependInfo& vdi = m_p->m_depends[k];
    Int64 mt = vdi.variable()->modifiedTime();
    if (mt>modified_time){
      need_update = true;
      break;
    }
  }
  if (!need_update){
    for( Integer k=0,n=m_p->m_mat_depends.size(); k<n; ++k ){
      MeshMaterialVariableDependInfo& vdi = m_p->m_mat_depends[k];
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
      String s = "no compute function for variable " + name();
      throw FatalErrorException(A_FUNCINFO,s);
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
  for( Integer k=0,n=m_p->m_depends.size(); k<n; ++k ){
    VariableDependInfo& vdi = m_p->m_depends[k];
    infos.add(vdi);
  }

  for( Integer k=0,n=m_p->m_mat_depends.size(); k<n; ++k ){
    MeshMaterialVariableDependInfo& vdi = m_p->m_mat_depends[k];
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

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MATERIALS_END_NAMESPACE
ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
