﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Array2Variable.cc                                           (C) 2000-2021 */
/*                                                                           */
/* Variable tableau 2D.                                                      */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/Array2Variable.h"

#include "arcane/utils/NotSupportedException.h"
#include "arcane/utils/ArgumentException.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/TraceInfo.h"
#include "arcane/utils/Ref.h"

#include "arcane/datatype/DataTypeTraits.h"
#include "arcane/datatype/DataStorageBuildInfo.h"

#include "arcane/VariableDiff.h"
#include "arcane/VariableBuildInfo.h"
#include "arcane/VariableInfo.h"
#include "arcane/IApplication.h"
#include "arcane/IVariableMng.h"
#include "arcane/IItemFamily.h"
#include "arcane/IVariableSynchronizer.h"
#include "arcane/IDataReader.h"
#include "arcane/ItemGroup.h"
#include "arcane/IParallelMng.h"
#include "arcane/IDataFactoryMng.h"

#include "arcane/core/internal/IDataInternal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class DataType>
class Array2VariableDiff
: public VariableDiff<DataType>
{
  typedef VariableDataTypeTraitsT<DataType> VarDataTypeTraits;
  typedef typename VariableDiff<DataType>::DiffInfo DiffInfo;

 public:

  Integer check(IVariable* var,ConstArray2View<DataType> ref,ConstArray2View<DataType> current,
                int max_print,bool compare_ghost)
  {
    typedef typename VariableDataTypeTraitsT<DataType>::IsNumeric IsNumeric;
    ITraceMng* msg = var->subDomain()->traceMng();
    ItemGroup group = var->itemGroup();
    if (group.null())
      return 0;
    GroupIndexTable * group_index_table = (var->isPartial())?group.localIdToIndex().get():0;

    int nb_diff = 0;
    Integer sid = var->subDomain()->subDomainId();
    Integer current_total_nb_element = current.totalNbElement();
    /*if (ref.totalNbElement()!=current_total_nb_element){
      msg->pinfo() << "Processeur: " << sid << " VDIFF: Variable '" << var->name()
      << "'different number of elements "
      << " current: " << current_total_nb_element
      << " ref: " << ref.totalNbElement();
      return current_total_nb_element;
      }*/
    if (current_total_nb_element==0)
      return 0;
    Integer ref_size1 = ref.dim1Size();
    Integer current_size1 = current.dim1Size();
    Integer ref_size2 = ref.dim2Size();
    Integer current_size2 = current.dim2Size();
    if (ref_size2!=current_size2){
      msg->pinfo() << "Processor: " << sid << " VDIFF: Variable '" << var->name()
                   << " bad dim2 size: ref=" << ref_size2 << " current=" << current_size2;
    }
    ENUMERATE_ITEM(i,group){
      const Item& item = *i;
      if (!item.isOwn() && !compare_ghost)
        continue;
      Integer index = item.localId();
      if (group_index_table)
        index = (*group_index_table)[index];

      if (index>=ref_size1 || index>=current_size1){
        ++nb_diff;
        msg->pinfo() << "Processor: " << sid << " VDIFF: Variable '" << var->name()
                     << "wrong number of elements : impossible comparison";
        continue;
      }
      ConstArrayView<DataType> lref = ref[index];
      ConstArrayView<DataType> lcurrent = current[index];
      for( Integer z=0; z<ref_size2; ++z ){
        DataType diff = DataType();
        DataType dref = lref[z];
        DataType dcurrent = lcurrent[z];
        if (VarDataTypeTraits::verifDifferent(dref,dcurrent,diff)){
          this->m_diffs_info.add(DiffInfo(dcurrent,dref,diff,item,z));
          ++nb_diff;
        }
      }
    }
    if (nb_diff!=0){
      this->sort(IsNumeric());
      this->dump(var,max_print);
    }
    return nb_diff;
  }

  Integer checkReplica(IParallelMng* pm,IVariable* var,ConstArray2View<DataType> var_value,
                       Integer max_print)
  {
    // Appelle la bonne spécialisation pour être sur que le type template possède
    // la réduction.
    typedef typename VariableDataTypeTraitsT<DataType>::HasReduceMinMax HasReduceMinMax;
    return _checkReplica2(pm,var,var_value,max_print,HasReduceMinMax());
  }

 private:

  Integer _checkReplica2(IParallelMng* pm,IVariable* var,ConstArray2View<DataType> var_values,
                         Integer max_print,FalseType has_reduce)
  {
    ARCANE_UNUSED(pm);
    ARCANE_UNUSED(var);
    ARCANE_UNUSED(var_values);
    ARCANE_UNUSED(max_print);
    ARCANE_UNUSED(has_reduce);
    throw NotSupportedException(A_FUNCINFO);
  }

  Integer _checkReplica2(IParallelMng* pm,IVariable* var,ConstArray2View<DataType> var_values,
                         Integer max_print,TrueType has_reduce)
  {
    ARCANE_UNUSED(has_reduce);
    typedef typename VariableDataTypeTraitsT<DataType>::IsNumeric IsNumeric;
    ITraceMng* msg = pm->traceMng();
    ItemGroup group = var->itemGroup();
    //TODO: traiter les variables qui ne sont pas sur des éléments du maillage.
    if (group.null())
      return 0;
    GroupIndexTable * group_index_table = (var->isPartial())?group.localIdToIndex().get():0;
    // Vérifie que tout les réplica ont le même nombre d'éléments dans chaque
    // dimension pour la variable.
    Integer total_nb_element = var_values.totalNbElement();
    Integer ref_size1 = var_values.dim1Size();
    Integer ref_size2 = var_values.dim2Size();

    Integer a_min_dims[2];
    ArrayView<Integer> min_dims(2,a_min_dims);
    Integer a_max_dims[2];
    ArrayView<Integer> max_dims(2,a_max_dims);
    max_dims[0] = min_dims[0] = ref_size1;
    max_dims[1] = min_dims[1] = ref_size2;

    pm->reduce(Parallel::ReduceMax,max_dims);
    pm->reduce(Parallel::ReduceMin,min_dims);
    msg->info(4) << "Array2Variable::CheckReplica2 rep_size=" << pm->commSize() << " rank=" << pm->commRank();
    if (max_dims[0]!=min_dims[0] || max_dims[1]!=min_dims[1]){
      const String& var_name = var->name();
      msg->info() << "Can not compare values on replica for variable '" << var_name << "'"
                  << " because the number of elements is not the same on all the replica "
                  << " min=" << min_dims[0] << "," << min_dims[1]
                  << " max="<< max_dims[0] << "," << max_dims[1];
      return total_nb_element;
    }
    if (total_nb_element==0)
      return 0;
    Integer nb_diff = 0;
    UniqueArray2<DataType> min_values(var_values);
    UniqueArray2<DataType> max_values(var_values);
    pm->reduce(Parallel::ReduceMax,max_values.viewAsArray());
    pm->reduce(Parallel::ReduceMin,min_values.viewAsArray());


    ENUMERATE_ITEM(i,group){
      Item item = *i;
      Integer index = item.localId();
      if (group_index_table)
        index = (*group_index_table)[index];

      if (index>=ref_size1){
        ++nb_diff;
        msg->pinfo() << "Processor: " << msg->traceId() << " VDIFF: Variable '" << var->name()
                     << "wrong number of elements : impossible comparison";
        continue;
      }
      ConstArrayView<DataType> lref = min_values[index];
      ConstArrayView<DataType> lcurrent = max_values[index];
      for( Integer z=0; z<ref_size2; ++z ){
        DataType diff = DataType();
        DataType dref = lref[z];
        DataType dcurrent = lcurrent[z];
        if (VarDataTypeTraits::verifDifferent(dref,dcurrent,diff)){
          this->m_diffs_info.add(DiffInfo(dcurrent,dref,diff,item,z));
          ++nb_diff;
        }
      }
    }

    if (nb_diff!=0){
      this->sort(IsNumeric());
      this->dump(var,max_print);
    }
    return nb_diff;
  }

};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename T> Array2VariableT<T>::
Array2VariableT(const VariableBuildInfo& vb,const VariableInfo& info)
: Variable(vb,info)
, m_data(nullptr)
{
  IDataFactoryMng* df = vb.dataFactoryMng();
  DataStorageBuildInfo storage_build_info(vb.traceMng());
  String storage_full_type = info.storageTypeInfo().fullName();
  Ref<IData> data = df->createSimpleDataRef(storage_full_type,storage_build_info);
  m_data = dynamic_cast<ValueDataType*>(data.get());
  _setData(makeRef(m_data));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename T> Array2VariableT<T>* Array2VariableT<T>::
getReference(const VariableBuildInfo& vb,const VariableInfo& vi)
{
  ThatClass* true_ptr = 0;
  IVariableMng* vm = vb.variableMng();
  IVariable* var = vm->checkVariable(vi);
  if (var)
    true_ptr = dynamic_cast<ThatClass*>(var);
  else{
    true_ptr = new ThatClass(vb,vi);
    vm->addVariable(true_ptr);
  }
  ARCANE_CHECK_PTR(true_ptr);
  return true_ptr;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename T> Array2VariableT<T>* Array2VariableT<T>::
getReference(IVariable* var)
{
  if (!var)
    throw ArgumentException(A_FUNCINFO,"null variable");
  ThatClass* true_ptr = dynamic_cast<ThatClass*>(var);
  if (!true_ptr)
    ARCANE_FATAL("Cannot build a reference from variable {0}",var->name());
  return true_ptr;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename T> void Array2VariableT<T>::
directResize(Integer s)
{
  /*info() << "RESIZE(1) " << fullName()
         << " wanted_dim1_size=" << s
         << " dim2_size=" << m_data->value().dim2Size()
         << " total=" << m_data->value().totalNbElement();*/
  m_data->_internal()->_internalDeprecatedValue().resize(s,m_data->view().dim2Size());
  syncReferences();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> void Array2VariableT<DataType>::
directResize(Integer dim1_size,Integer dim2_size)
{
  /*info() << "RESIZE(2) " << fullName()
         << " wanted_dim1_size=" << dim1_size
         << " wanted_dim2_size=" << dim2_size
         << " total=" << m_data->value().totalNbElement()
         << " dim1_size=" << m_data->value().dim1Size()
         << " dim2_size=" << m_data->value().dim2Size();*/
  m_data->_internal()->_internalDeprecatedValue().resize(dim1_size,dim2_size);
  /*info() << "RESIZE(2) AFTER " << fullName()
         << " total=" << m_data->value().totalNbElement()
         << " dim1_size=" << m_data->value().dim1Size()
         << " dim2_size=" << m_data->value().dim2Size()
         << " addr=" << m_data->value().view().unguardedBasePointer();*/
  syncReferences();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> void Array2VariableT<DataType>::
shrinkMemory()
{
  m_data->_internal()->shrink();
  syncReferences();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename T> void Array2VariableT<T>::
print(std::ostream&) const
{
  //ConstArray2View<T> x(m_data->value().constView());
  //arrayDumpValue(o,x);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename T> void Array2VariableT<T>::
synchronize()
{
  if (itemKind()==IK_Unknown)
    ARCANE_THROW(NotSupportedException,"variable '{0}' is not a mesh variable",fullName());
  IItemFamily* family = itemGroup().itemFamily();
  if (!family)
    ARCANE_FATAL("variable '{0}' without family",fullName());
  if(isPartial())
    itemGroup().synchronizer()->synchronize(this);
  else
    family->allItemsSynchronizer()->synchronize(this);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename T> Real Array2VariableT<T>::
allocatedMemory() const
{
  Real v1 = static_cast<Real>(sizeof(T));
  Real v2 = static_cast<Real>(m_data->view().totalNbElement());
  return v1*v2;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename T> Integer Array2VariableT<T>::
checkIfSync(int max_print)
{
  ValueType& data_values = m_data->_internal()->_internalDeprecatedValue();

  Integer dim1_size = valueView().dim1Size();
  if (dim1_size==0)
    return 0;

  //Integer dim2_size = value().dim2Size();
  IItemFamily* family = itemGroup().itemFamily();
  if (family){
    UniqueArray2<T> ref_array(constValueView());
    this->synchronize(); // fonctionne pour toutes les variables
    Array2VariableDiff<T> csa;
    Array2View<T> from_array(valueView());
    Integer nerror = csa.check(this,ref_array,from_array,max_print,true);
    data_values.copy(ref_array);
    return nerror;
  }
  return 0;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename T> Integer Array2VariableT<T>::
checkIfSame(IDataReader* reader,int max_print,bool compare_ghost)
{
  if (itemKind()==IK_Particle)
    return 0;
  ConstArray2View<T> from_array(valueView());

  Ref< IArray2DataT<T> > ref_data(m_data->cloneTrueEmptyRef());
  reader->read(this,ref_data.get());

  Array2VariableDiff<T> csa;
  return csa.check(this,ref_data->view(),from_array,max_print,compare_ghost);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
// Utilise une fonction Helper afin de spécialiser l'appel dans le
// cas du type 'Byte' car ArrayVariableDiff::checkReplica() utilise
// une réduction Min/Max et cela n'existe pas en MPI pour le type Byte.
namespace
{
  template<typename T> Integer
  _checkIfSameOnAllReplicaHelper(IParallelMng* pm,IVariable* var,
                                 ConstArray2View<T> values,Integer max_print)
  {
    Array2VariableDiff<T> csa;
    return csa.checkReplica(pm,var,values,max_print);
  }

  // Spécialisation pour le type 'Byte' qui ne supporte pas les réductions.
  Integer
  _checkIfSameOnAllReplicaHelper(IParallelMng* pm,IVariable* var,
                                 ConstArray2View<Byte> values,Integer max_print)
  {
    Integer dim1_size = values.dim1Size();
    Integer dim2_size = values.dim2Size();
    UniqueArray2<Integer> int_values(dim1_size,dim2_size);
    for( Integer i=0; i<dim1_size; ++i )
      for( Integer j=0; j<dim2_size; ++j )
        int_values[i][j] = values[i][j];
    Array2VariableDiff<Integer> csa;
    return csa.checkReplica(pm,var,int_values,max_print);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename T> Integer Array2VariableT<T>::
_checkIfSameOnAllReplica(IParallelMng* replica_pm,Integer max_print)
{
  return _checkIfSameOnAllReplicaHelper(replica_pm,this,constValueView(),max_print);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename T> void Array2VariableT<T>::
_internalResize(Integer new_size,Integer nb_additional_element)
{
  // Exception deplace de ItemGroupImpl::removeItems.
  // C'est un probleme d'implementation des variables partielles Arcane et non
  // du concept de variable partielle (cf ItemGroupMap qui l'implemente).
  //   if (isPartial() && new_size < value().dim1Size())
  //     throw NotSupportedException(A_FUNCINFO,"Cannot remove items to group with partial variables");

  ValueType& data_values = m_data->_internal()->_internalDeprecatedValue();
  ValueType& container_ref = data_values;

  Integer dim2_size = data_values.dim2Size();

  if (nb_additional_element!=0){
    Integer capacity = data_values.capacity();
    if (new_size>capacity)
      data_values.reserve(new_size+nb_additional_element*dim2_size);
  }

  eDataInitialisationPolicy init_policy = getGlobalDataInitialisationPolicy();
  // Si la nouvelle taille est supérieure à l'ancienne,
  // initialise les nouveaux éléments suivant
  // la politique voulue
  Integer current_size = data_values.dim1Size();

  /*info() << "RESIZE INTERNAL " << fullName()
         << " wanted_dim1_size=" << new_size
         << " dim1_size=" << value().dim1Size()
         << " dim2size=" << dim2_size
         << " total=" << value().totalNbElement();*/
  if (init_policy==DIP_InitWithDefault)
    data_values.resize(new_size,dim2_size);
  else
    data_values.resizeNoInit(new_size,dim2_size);

  if (new_size>current_size && init_policy==DIP_InitWithNan){
    for( Integer i=current_size; i<new_size; ++i )
      DataTypeTraitsT<T>::fillNan(data_values[i]);
  }

  // Compacte la mémoire si demandé
  if (_wantShrink()){
    if (container_ref.totalNbElement() < container_ref.capacity())
      container_ref.shrink();
  }

  /*info() << "RESIZE INTERNAL AFTER " << fullName()
    << " addr=" << m_data->value().view().unguardedBasePointer()
    << " dim1=" << value().dim1Size()
    << " dim2=" << value().dim2Size();*/
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename T> void Array2VariableT<T>::
copyItemsValues(Int32ConstArrayView source, Int32ConstArrayView destination)
{
  ARCANE_ASSERT(source.size()==destination.size(),
		("Unable to copy: source and destination have different sizes !"));

  const Integer dim2_size = valueView().dim2Size();
  const Integer nb_copy = source.size();
  Array2View<T> value = m_data->view();

  for( Integer i=0; i<nb_copy; ++i ){
    for( Integer j=0; j<dim2_size; ++j )
      value[destination[i]][j] = value[source[i]][j];
  }
  syncReferences();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename T> void Array2VariableT<T>::
copyItemsMeanValues(Int32ConstArrayView first_source,
                    Int32ConstArrayView second_source,
                    Int32ConstArrayView destination)
{
  ARCANE_ASSERT((first_source.size()==destination.size()) && (second_source.size()==destination.size()),
                ("Unable to copy: source and destination have different sizes !"));

  const Integer dim2_size = valueView().dim2Size();
  const Integer nb_copy = first_source.size();
  Array2View<T> value = m_data->view();

  for( Integer i=0; i<nb_copy; ++i ){
    for( Integer j=0; j<dim2_size; ++j )
      value[destination[i]][j] = (T)((value[first_source[i]][j]+value[second_source[i]][j])/2);
  }
  syncReferences();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename T> void Array2VariableT<T>::
compact(Int32ConstArrayView new_to_old_ids)
{
  if (isPartial()) {
    debug(Trace::High) << "Skip compact for partial variable " << name();
    return;
  }

  ValueType& current_value = m_data->_internal()->_internalDeprecatedValue();
  Integer current_size = current_value.dim1Size();
  if (current_size==0)
    return;

  Integer dim2_size = current_value.dim2Size();
  if (dim2_size==0)
    return;

  //TODO: eviter le clone
  UniqueArray2<T> old_value(current_value);

  Integer new_size = new_to_old_ids.size();
  current_value.resize(new_size,dim2_size);
  /*info() << "Variable: " << name() << " Compacte2D: size=" << current_size
         << " dim2_size=" << dim2_size
         << " new_size=" << new_size;*/
  
  if (arcaneIsCheck()){
    for( Integer i=0; i<new_size; ++i ){
      Integer nto = new_to_old_ids[i];
      ArrayView<T> v = current_value.at(i);
      ArrayView<T> ov = old_value.at(nto);
      for( Integer j=0; j<dim2_size; ++j )
        v.setAt(j,ov.at(j));
    }
  }
  else{
    for( Integer i=0; i<new_size; ++i ){
      Integer nto = new_to_old_ids[i];
      for( Integer j=0; j<dim2_size; ++j )
        current_value[i][j] = old_value[nto][j];
    }
  }

  syncReferences();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename T> void Array2VariableT<T>::
setIsSynchronized()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename T> void Array2VariableT<T>::
setIsSynchronized(const ItemGroup&)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> void
Array2VariableT<DataType>::
swapValues(ThatClass& rhs)
{
  _checkSwapIsValid(&rhs);
  // TODO: regarder s'il faut que les deux variables aient le même nombre
  // d'éléments mais a priori cela ne semble pas indispensable.
  m_data->swapValues(rhs.m_data);
  // Il faut mettre à jour les références pour cette variable et \a rhs.
  syncReferences();
  rhs.syncReferences();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> auto
Array2VariableT<DataType>::
value() -> ValueType&
{
  return m_data->_internal()->_internalDeprecatedValue();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template class Array2VariableT<Byte>;
template class Array2VariableT<Real>;
template class Array2VariableT<Int16>;
template class Array2VariableT<Int32>;
template class Array2VariableT<Int64>;
template class Array2VariableT<Real2>;
template class Array2VariableT<Real2x2>;
template class Array2VariableT<Real3>;
template class Array2VariableT<Real3x3>;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
