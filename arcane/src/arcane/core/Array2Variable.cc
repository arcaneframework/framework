// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Array2Variable.cc                                           (C) 2000-2025 */
/*                                                                           */
/* Variable tableau 2D.                                                      */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/Array2Variable.h"

#include "arcane/utils/NotSupportedException.h"
#include "arcane/utils/ArgumentException.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/TraceInfo.h"
#include "arcane/utils/Ref.h"
#include "arcane/utils/CheckedConvert.h"
#include "arcane/utils/ArrayShape.h"
#include "arcane/utils/ITraceMng.h"

#include "arcane/datatype/DataTypeTraits.h"
#include "arcane/datatype/DataStorageBuildInfo.h"

#include "arcane/core/VariableDiff.h"
#include "arcane/core/VariableBuildInfo.h"
#include "arcane/core/VariableInfo.h"
#include "arcane/core/IApplication.h"
#include "arcane/core/IVariableMng.h"
#include "arcane/core/IItemFamily.h"
#include "arcane/core/IVariableSynchronizer.h"
#include "arcane/core/IDataReader.h"
#include "arcane/core/ItemGroup.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/IDataFactoryMng.h"
#include "arcane/core/IMesh.h"
#include "arcane/core/VariableComparer.h"

#include "arcane/core/internal/IDataInternal.h"
#include "arcane/core/internal/IVariableMngInternal.h"
#include "arcane/core/internal/IVariableInternal.h"

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

  VariableComparerResults
  check(IVariable* var, ConstArray2View<DataType> ref, ConstArray2View<DataType> current,
        const VariableComparerArgs& compare_args)
  {
    const bool compare_ghost = compare_args.isCompareGhost();
    ItemGroup group = var->itemGroup();
    if (group.null())
      return {};
    IMesh* mesh = group.mesh();
    if (!mesh)
      return {};
    ITraceMng* msg = mesh->traceMng();
    IParallelMng* pm = mesh->parallelMng();

    GroupIndexTable * group_index_table = (var->isPartial())?group.localIdToIndex().get():0;

    int nb_diff = 0;
    Int32 sid = pm->commRank();
    Integer current_total_nb_element = current.totalNbElement();
    /*if (ref.totalNbElement()!=current_total_nb_element){
      msg->pinfo() << "Processeur: " << sid << " VDIFF: Variable '" << var->name()
      << "'different number of elements "
      << " current: " << current_total_nb_element
      << " ref: " << ref.totalNbElement();
      return current_total_nb_element;
      }*/
    if (current_total_nb_element==0)
      return {};
    Integer ref_size1 = ref.dim1Size();
    Integer current_size1 = current.dim1Size();
    Integer ref_size2 = ref.dim2Size();
    Integer current_size2 = current.dim2Size();
    if (ref_size2!=current_size2){
      msg->pinfo() << "Processor: " << sid << " VDIFF: Variable '" << var->name()
                   << " bad dim2 size: ref=" << ref_size2 << " current=" << current_size2;
    }
    ENUMERATE_ITEM(i,group){
      Item item = *i;
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
        if (VarDataTypeTraits::verifDifferent(dref,dcurrent,diff,true)){
          this->m_diffs_info.add(DiffInfo(dcurrent,dref,diff,item,z));
          ++nb_diff;
        }
      }
    }
    if (nb_diff!=0)
      this->_sortAndDump(var, pm, compare_args);

    return VariableComparerResults(nb_diff);
  }

  VariableComparerResults
  checkReplica(IVariable* var, ConstArray2View<DataType> var_values,
               const VariableComparerArgs& compare_args)
  {
    IParallelMng* replica_pm = var->_internalApi()->replicaParallelMng();
    if (!replica_pm)
      return {};
    // Appelle la bonne spécialisation pour être certain que le type template possède
    // la réduction.
    using ReduceType = typename VariableDataTypeTraitsT<DataType>::HasReduceMinMax;
    if constexpr(std::is_same<TrueType,ReduceType>::value)
      return _checkReplica2(replica_pm, var, var_values, compare_args);

    ARCANE_UNUSED(replica_pm);
    ARCANE_UNUSED(var);
    ARCANE_UNUSED(var_values);
    ARCANE_UNUSED(compare_args);
    throw NotSupportedException(A_FUNCINFO);
  }

 private:

  VariableComparerResults
  _checkReplica2(IParallelMng* pm, IVariable* var, ConstArray2View<DataType> var_values,
                 const VariableComparerArgs& compare_args)
  {
    ITraceMng* msg = pm->traceMng();
    ItemGroup group = var->itemGroup();
    //TODO: traiter les variables qui ne sont pas sur des éléments du maillage.
    if (group.null())
      return {};
    GroupIndexTable * group_index_table = (var->isPartial())?group.localIdToIndex().get():0;
    // Vérifie que tous les réplica ont le même nombre d'éléments dans chaque
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
      return VariableComparerResults(total_nb_element);
    }
    if (total_nb_element==0)
      return {};
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
        if (VarDataTypeTraits::verifDifferent(dref,dcurrent,diff,true)){
          this->m_diffs_info.add(DiffInfo(dcurrent,dref,diff,item,z));
          ++nb_diff;
        }
      }
    }

    if (nb_diff!=0)
      this->_sortAndDump(var, pm, compare_args);

    return VariableComparerResults(nb_diff);
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
  if (vb.isNull())
    return nullptr;
  ThatClass* true_ptr = nullptr;
  IVariableMng* vm = vb.variableMng();
  IVariable* var = vm->checkVariable(vi);
  if (var)
    true_ptr = dynamic_cast<ThatClass*>(var);
  else{
    true_ptr = new ThatClass(vb,vi);
    vm->_internalApi()->addVariable(true_ptr);
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
  m_data->_internal()->resizeOnlyDim1(s);
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
  m_data->_internal()->resize(dim1_size,dim2_size);
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
directResizeAndReshape(const ArrayShape& shape)
{
  Int32 dim1_size = valueView().dim1Size();
  Int32 dim2_size = CheckedConvert::toInt32(shape.totalNbElement());

  m_data->_internal()->resize(dim1_size,dim2_size);
  m_data->setShape(shape);

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

template<typename T> void Array2VariableT<T>::
synchronize(Int32ConstArrayView local_ids)
{
  if (itemKind()==IK_Unknown)
    ARCANE_THROW(NotSupportedException,"variable '{0}' is not a mesh variable",fullName());
  IItemFamily* family = itemGroup().itemFamily();
  if (!family)
    ARCANE_FATAL("variable '{0}' without family",fullName());
  if(isPartial())
    itemGroup().synchronizer()->synchronize(this, local_ids);
  else
    family->allItemsSynchronizer()->synchronize(this, local_ids);
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
// Utilise une fonction Helper afin de spécialiser l'appel dans le
// cas du type 'Byte' car ArrayVariableDiff::checkReplica() utilise
// une réduction Min/Max et cela n'existe pas en MPI pour le type Byte.
namespace
{
  template <typename T> VariableComparerResults
  _checkIfSameOnAllReplicaHelper(IVariable* var, ConstArray2View<T> values,
                                 const VariableComparerArgs& compare_args)
  {
    Array2VariableDiff<T> csa;
    return csa.checkReplica(var, values, compare_args);
  }

  // Spécialisation pour le type 'Byte' qui ne supporte pas les réductions.
  VariableComparerResults
  _checkIfSameOnAllReplicaHelper(IVariable* var, ConstArray2View<Byte> values,
                                 const VariableComparerArgs& compare_args)
  {
    Integer dim1_size = values.dim1Size();
    Integer dim2_size = values.dim2Size();
    UniqueArray2<Integer> int_values(dim1_size,dim2_size);
    for( Integer i=0; i<dim1_size; ++i )
      for( Integer j=0; j<dim2_size; ++j )
        int_values[i][j] = values[i][j];
    Array2VariableDiff<Integer> csa;
    return csa.checkReplica(var, int_values, compare_args);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename T> VariableComparerResults Array2VariableT<T>::
_compareVariable(const VariableComparerArgs& compare_args)
{
  switch (compare_args.compareMode()) {
  case eVariableComparerCompareMode::Same: {
    if (itemKind() == IK_Particle)
      return {};
    IDataReader* reader = compare_args.dataReader();
    ARCANE_CHECK_POINTER(reader);

    ConstArray2View<T> from_array(valueView());
    Ref<IArray2DataT<T>> ref_data(m_data->cloneTrueEmptyRef());
    reader->read(this, ref_data.get());

    Array2VariableDiff<T> csa;
    VariableComparerResults r = csa.check(this, ref_data->view(), from_array, compare_args);
    return r;
  } break;
  case eVariableComparerCompareMode::Sync: {
    IItemFamily* family = itemGroup().itemFamily();
    if (!family)
      return {};

    Integer dim1_size = valueView().dim1Size();
    if (dim1_size == 0)
      return {};

    ValueType& data_values = m_data->_internal()->_internalDeprecatedValue();

    UniqueArray2<T> ref_array(constValueView());
    this->synchronize(); // fonctionne pour toutes les variables
    Array2VariableDiff<T> csa;
    Array2View<T> from_array(valueView());
    VariableComparerResults results = csa.check(this, ref_array, from_array, compare_args);
    data_values.copy(ref_array);
    return results;
  }
  case eVariableComparerCompareMode::SameOnAllReplica: {
    VariableComparerResults r = _checkIfSameOnAllReplicaHelper(this, constValueView(), compare_args);
    return r;
  }
  }
  ARCANE_FATAL("Invalid value for compare mode '{0}'", (int)compare_args.compareMode());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename T> void Array2VariableT<T>::
_internalResize(const VariableResizeArgs& resize_args)
{
  Int32 new_size = resize_args.newSize();
  Int32 nb_additional_element = resize_args.nbAdditionalCapacity();
  bool use_no_init = resize_args.isUseNoInit();

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
  if (use_no_init || (init_policy!=DIP_InitWithDefault))
    data_values.resizeNoInit(new_size,dim2_size);
  else
    data_values.resize(new_size,dim2_size);

  if (new_size>current_size){
    bool use_nan = (init_policy==DIP_InitWithNan);
    bool use_nan2 = (init_policy==DIP_InitInitialWithNanResizeWithDefault) && !_hasValidData();
    if (use_nan || use_nan2){
      for( Integer i=current_size; i<new_size; ++i )
        DataTypeTraitsT<T>::fillNan(data_values[i]);
    }
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

template<typename DataType> void
Array2VariableT<DataType>::
fillShape(ArrayShape& shape_with_item)
{
  // TODO: Cette méthode est indépendante du type de donnée.
  // Il faudrait pouvoir en faire une seule version.
  ArrayShape shape = m_data->shape();
  const Int32 nb_rank = shape_with_item.nbDimension();
  //std::cout << "SHAPE=" << shape.dimensions() << " internal_rank=" << nb_rank << "\n";
  auto array_view = m_data->view();
  Int32 dim0_size = array_view.dim1Size();

  shape_with_item.setDimension(0, dim0_size);
  Int32 nb_orig_shape = shape.nbDimension();
  for (Int32 i = 0; i < nb_orig_shape; ++i) {
    shape_with_item.setDimension(i + 1, shape.dimension(i));
  }

  // Si la forme est plus petite que notre rang, remplit les dimensions
  // supplémentaires par la valeur 1.
  for (Int32 i = (nb_orig_shape + 1); i < nb_rank; ++i) {
    shape_with_item.setDimension(i, 1);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_INTERNAL_INSTANTIATE_TEMPLATE_FOR_NUMERIC_DATATYPE(Array2VariableT);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
