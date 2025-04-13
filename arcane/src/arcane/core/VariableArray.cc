// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VariableArray.cc                                            (C) 2000-2025 */
/*                                                                           */
/* Variable tableau 1D.                                                      */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/NotSupportedException.h"
#include "arcane/utils/ArgumentException.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/TraceInfo.h"
#include "arcane/utils/Ref.h"
#include "arcane/utils/MemoryAccessInfo.h"
#include "arcane/utils/MemoryAllocator.h"

#include "arcane/core/VariableDiff.h"
#include "arcane/core/VariableBuildInfo.h"
#include "arcane/core/VariableInfo.h"
#include "arcane/core/IApplication.h"
#include "arcane/core/IVariableMng.h"
#include "arcane/core/IItemFamily.h"
#include "arcane/core/IVariableSynchronizer.h"
#include "arcane/core/IDataReader.h"
#include "arcane/core/ItemGroup.h"
#include "arcane/core/IDataFactoryMng.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/IMesh.h"
#include "arcane/core/VariableComparer.h"

#include "arcane/core/datatype/DataTracer.h"
#include "arcane/core/datatype/DataTypeTraits.h"
#include "arcane/core/datatype/DataStorageBuildInfo.h"

#include "arcane/core/VariableArray.h"
#include "arcane/core/RawCopy.h"

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
class ArrayVariableDiff
: public VariableDiff<DataType>
{
  using VarDataTypeTraits = VariableDataTypeTraitsT<DataType>;
  using DiffInfo = typename VariableDiff<DataType>::DiffInfo;
  static constexpr bool IsNumeric = std::is_same_v<typename VarDataTypeTraits::IsNumeric, TrueType>;
  using NormType = typename VarDataTypeTraits::NormType;

 public:

  VariableComparerResults
  check(IVariable* var, ConstArrayView<DataType> ref, ConstArrayView<DataType> current,
        const VariableComparerArgs& compare_args)
  {
    const int max_print = compare_args.maxPrint();
    const bool compare_ghost = compare_args.isCompareGhost();
    if (var->itemKind() == IK_Unknown)
      return _checkAsArray(var, ref, current, compare_args);

    ItemGroup group = var->itemGroup();
    if (group.null())
      return {};
    IMesh* mesh = group.mesh();
    if (!mesh)
      return {};
    ITraceMng* msg = mesh->traceMng();
    IParallelMng* pm = mesh->parallelMng();

    GroupIndexTable* group_index_table = (var->isPartial()) ? group.localIdToIndex().get() : nullptr;

    int nb_diff = 0;
    bool compare_failed = false;
    eVariableComparerComputeDifferenceMethod diff_method = compare_args.computeDifferenceMethod();
    // Pas de norme max si le type n'est pas numérique.
    if (!IsNumeric)
      diff_method = eVariableComparerComputeDifferenceMethod::Relative;

    Integer ref_size = ref.size();
    NormType local_norm_max = {};

    if constexpr (IsNumeric) {
      bool is_use_local_norm = diff_method == eVariableComparerComputeDifferenceMethod::LocalNormMax;
      if (is_use_local_norm) {
        // Gros copier-coller pour calculer la norme globale
        ENUMERATE_ITEM (i, group) {
          Item item = *i;
          if (!item.isOwn() && !compare_ghost)
            continue;
          Integer index = item.localId();
          if (group_index_table) {
            index = (*group_index_table)[index];
            if (index < 0)
              continue;
          }
          if (index >= ref_size) {
            continue;
          }
          else {
            DataType dref = ref[index];
            NormType norm_max = VarDataTypeTraits::normeMax(dref);
            if (norm_max > local_norm_max) {
              local_norm_max = norm_max;
            }
          }
        }
      }
    }
    // On calcule les erreurs normalisées
    ENUMERATE_ITEM (i, group) {
      Item item = *i;
      if (!item.isOwn() && !compare_ghost)
        continue;
      Integer index = item.localId();
      if (group_index_table) {
        index = (*group_index_table)[index];
        if (index < 0)
          continue;
      }
      DataType diff = DataType();
      if (index >= ref_size) {
        ++nb_diff;
        compare_failed = true;
      }
      else {
        DataType dref = ref[index];
        DataType dcurrent = current[index];
        bool is_diff = _computeDifference(dref, dcurrent, diff, local_norm_max, diff_method);
        if (is_diff) {
          this->m_diffs_info.add(DiffInfo(dcurrent, dref, diff, item, NULL_ITEM_ID));
          ++nb_diff;
        }
      }
    }
    if (compare_failed) {
      Int32 sid = pm->commRank();
      const String& var_name = var->name();
      msg->pinfo() << "Processor " << sid << " : "
                   << "comparison impossible because the number of the elements is different "
                   << " for the variable " << var_name << " ref_size=" << ref_size;
    }
    if (nb_diff != 0)
      this->_sortAndDump(var, pm, max_print);

    return VariableComparerResults(nb_diff);
  }

  VariableComparerResults
  checkReplica(IVariable* var, ConstArrayView<DataType> var_value,
               const VariableComparerArgs& compare_args)
  {
    IParallelMng* replica_pm = compare_args.replicaParallelMng();
    ARCANE_CHECK_POINTER(replica_pm);
    // Appelle la bonne spécialisation pour être certain que le type template possède
    // la réduction.
    using ReduceType = typename VariableDataTypeTraitsT<DataType>::HasReduceMinMax;
    if constexpr(std::is_same<TrueType,ReduceType>::value)
      return _checkReplica2(replica_pm, var, var_value, compare_args);

    ARCANE_UNUSED(replica_pm);
    ARCANE_UNUSED(var);
    ARCANE_UNUSED(var_value);
    ARCANE_UNUSED(compare_args);
    throw NotSupportedException(A_FUNCINFO);
  }

 private:

  VariableComparerResults
  _checkAsArray(IVariable* var, ConstArrayView<DataType> ref, ConstArrayView<DataType> current,
                const VariableComparerArgs& compare_args)
  {
    const int max_print = compare_args.maxPrint();
    IParallelMng* pm = var->variableMng()->parallelMng();
    ITraceMng* msg = pm->traceMng();

    int nb_diff = 0;
    bool compare_failed = false;
    Integer ref_size = ref.size();
    Integer current_size = current.size();
    eVariableComparerComputeDifferenceMethod diff_method = compare_args.computeDifferenceMethod();
    // Pas de norme max si le type n'est pas numérique.
    if (!IsNumeric)
      diff_method = eVariableComparerComputeDifferenceMethod::Relative;
    NormType local_norm_max = {};

    if constexpr (IsNumeric) {
      bool is_use_local_norm = compare_args.computeDifferenceMethod() == eVariableComparerComputeDifferenceMethod::LocalNormMax;
      if (is_use_local_norm) {
        // Gros copier-coller pour calculer la norme globale
        for (Integer index = 0; index < current_size; ++index) {
          if (index >= ref_size) {
            continue;
          }
          else {
            DataType dref = ref[index];
            typename VarDataTypeTraits::NormType norm_max = VarDataTypeTraits::normeMax(dref);
            if (norm_max > local_norm_max) {
              local_norm_max = norm_max;
            }
          }
        }
      }
    }
    // On calcule les erreurs normalisées
    for( Integer index=0; index<current_size; ++index ){
      DataType diff = DataType();
      if (index>=ref_size){
        ++nb_diff;
        compare_failed = true;
      }
      else{
        DataType dref = ref[index];
        DataType dcurrent = current[index];
        if (_computeDifference(dref, dcurrent, diff, local_norm_max, diff_method)) {
          this->m_diffs_info.add(DiffInfo(dcurrent,dref,diff,index,NULL_ITEM_ID));
          ++nb_diff;
        }
      }
    }
    if (compare_failed){
      Int32 sid = pm->commRank();
      const String& var_name = var->name();
      msg->pinfo() << "Processor " << sid << " : "
                   << " comparaison impossible car nombre d'éléments différents"
                   << " pour la variable " << var_name << " ref_size=" << ref_size;
        
    }
    if (nb_diff!=0)
      this->_sortAndDump(var,pm,max_print);

    return VariableComparerResults(nb_diff);
  }

  VariableComparerResults
  _checkReplica2(IParallelMng* pm, IVariable* var, ConstArrayView<DataType> var_values,
                 const VariableComparerArgs& compare_args)
  {
    const int max_print = compare_args.maxPrint();
    ITraceMng* msg = pm->traceMng();
    Integer size = var_values.size();
    // Vérifie que tous les réplica ont le même nombre d'éléments pour la variable.
    Integer max_size = pm->reduce(Parallel::ReduceMax,size);
    Integer min_size = pm->reduce(Parallel::ReduceMin,size);
    msg->info(5) << "CheckReplica2 rep_size=" << pm->commSize() << " rank=" << pm->commRank();
    if (max_size!=min_size){
      const String& var_name = var->name();
      msg->info() << "Can not compare values on replica for variable '" << var_name << "'"
                  << " because the number of elements is not the same on all the replica "
                  << " min=" << min_size << " max="<< max_size;
      return VariableComparerResults(max_size);
    }
    Integer nb_diff = 0;
    UniqueArray<DataType> min_values(var_values);
    UniqueArray<DataType> max_values(var_values);
    pm->reduce(Parallel::ReduceMax,max_values);
    pm->reduce(Parallel::ReduceMin,min_values);

    for( Integer index=0; index<size; ++index ){
      DataType diff = DataType();
      DataType min_val = min_values[index];
      DataType max_val = max_values[index];
      if (VarDataTypeTraits::verifDifferent(min_val,max_val,diff,true)){
        this->m_diffs_info.add(DiffInfo(min_val,max_val,diff,index,NULL_ITEM_ID));
        ++nb_diff;
      }
    }
    if (nb_diff!=0)
      this->_sortAndDump(var,pm,max_print);

    return VariableComparerResults(nb_diff);
  }
  bool _computeDifference(const DataType& dref, const DataType& dcurrent, DataType& diff,
                          const NormType& local_norm_max,
                          eVariableComparerComputeDifferenceMethod diff_method)
  {
    bool is_diff = false;
    switch (diff_method) {
    case eVariableComparerComputeDifferenceMethod::Relative:
      is_diff = VarDataTypeTraits::verifDifferent(dref, dcurrent, diff, true);
      break;
    case eVariableComparerComputeDifferenceMethod::LocalNormMax:
      is_diff = VarDataTypeTraits::verifDifferentNorm(dref, dcurrent, diff, local_norm_max, true);
      break;
    }
    return is_diff;
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename T> VariableArrayT<T>::
VariableArrayT(const VariableBuildInfo& vb,const VariableInfo& info)
: Variable(vb,info)
, m_value(nullptr)
{
  IDataFactoryMng* df = vb.dataFactoryMng();
  DataStorageBuildInfo storage_build_info(vb.traceMng());
  String storage_full_type = info.storageTypeInfo().fullName();
  Ref<IData> data = df->createSimpleDataRef(storage_full_type,storage_build_info);
  m_value = dynamic_cast<ValueDataType*>(data.get());
  ARCANE_CHECK_POINTER(m_value);
  _setData(makeRef(m_value));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename T> VariableArrayT<T>::
~VariableArrayT()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename T> VariableArrayT<T>* VariableArrayT<T>::
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

template<typename T> VariableArrayT<T>* VariableArrayT<T>::
getReference(IVariable* var)
{
  if (!var)
    throw ArgumentException(A_FUNCINFO,"null variable");
  auto* true_ptr = dynamic_cast<ThatClass*>(var);
  if (!true_ptr)
    ARCANE_FATAL("Can not build a reference from variable {0}",var->name());
  return true_ptr;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename T> void VariableArrayT<T>::
print(std::ostream& o) const
{
	ConstArrayView<T> x(m_value->view());
	Integer size = x.size();
	o << "(dimension=" << size << ") ";
	if (size<=150){
		for( auto& i : x ){
			o << i << '\n';
		}
	}
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename T> void VariableArrayT<T>::
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

template<typename T> void VariableArrayT<T>::
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

template<typename T> Real VariableArrayT<T>::
allocatedMemory() const
{
  Real v1 = (Real)(sizeof(T));
  Real v2 = (Real)(m_value->view().size());
  return v1 * v2;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
// Utilise une fonction Helper afin de spécialiser l'appel dans le
// cas du type 'Byte' car ArrayVariableDiff::checkReplica() utilise
// une réduction Min/Max et cela n'existe pas en MPI pour le type Byte.
namespace
{
  template <typename T> VariableComparerResults
  _checkIfSameOnAllReplicaHelper(IVariable* var, ConstArrayView<T> values,
                                 const VariableComparerArgs& compare_args)
  {
    ArrayVariableDiff<T> csa;
    return csa.checkReplica(var, values, compare_args);
  }

  // Spécialisation pour le type 'Byte' qui ne supporte pas les réductions.
  VariableComparerResults
  _checkIfSameOnAllReplicaHelper(IVariable* var, ConstArrayView<Byte> values,
                                 const VariableComparerArgs& compare_args)
  {
    Integer size = values.size();
    UniqueArray<Integer> int_values(size);
    for( Integer i=0; i<size; ++i )
      int_values[i] = values[i];
    ArrayVariableDiff<Integer> csa;
    return csa.checkReplica(var, int_values, compare_args);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename T> VariableComparerResults VariableArrayT<T>::
_compareVariable(const VariableComparerArgs& compare_args)
{
  switch (compare_args.compareMode()) {
  case eVariableComparerCompareMode::Same: {

    if (itemKind() == IK_Particle)
      return {};
    IDataReader* reader = compare_args.dataReader();
    ARCANE_CHECK_POINTER(reader);

    ArrayView<T> from_array(valueView());

    Ref<IArrayDataT<T>> ref_data(m_value->cloneTrueEmptyRef());
    reader->read(this, ref_data.get());

    ArrayVariableDiff<T> csa;
    VariableComparerResults r = csa.check(this, ref_data->view(), from_array, compare_args);
    return r;
  }
  case eVariableComparerCompareMode::Sync: {
    IItemFamily* family = itemGroup().itemFamily();
    if (!family)
      return {};
    ValueType& data_values = m_value->_internal()->_internalDeprecatedValue();
    UniqueArray<T> ref_array(constValueView());
    this->synchronize(); // fonctionne pour toutes les variables
    ArrayVariableDiff<T> csa;
    ConstArrayView<T> from_array(constValueView());
    VariableComparerResults r = csa.check(this, ref_array, from_array, compare_args);
    data_values.copy(ref_array);
    return r;
  }
  case eVariableComparerCompareMode::SameReplica: {
    VariableComparerResults r = _checkIfSameOnAllReplicaHelper(this, constValueView(), compare_args);
    return r;
  }
  }
  ARCANE_FATAL("Invalid value for compare mode '{0}'", (int)compare_args.compareMode());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Initialise la variable.
 *
 Initialise la variable avec la valeur \a value sur le groupe \a group.
 
 La valeur étant passée sous forme d'une chaine de caractère, vérifie que
 la conversion en le type de la variable est possible. De même vérifie
 que le groupe \a group est du type #GroupType. Si l'un de ces deux points
 n'est pas respecté, l'initialisation échoue.

 \retval true en cas d'erreur,
 \retval false en cas de succés.
*/
template<typename T> bool VariableArrayT<T>::
initialize(const ItemGroup& group,const String& value)
{
  //TODO: peut-être vérifier que la variable est utilisée ?

  // Tente de convertir value en une valeur du type de la variable.
  T v = T();
  bool is_bad = VariableDataTypeTraitsT<T>::getValue(v,value);

  if (is_bad){
    error() << String::format("Can not convert the string '{0}' to type '{1}'",
                              value,dataType());
    return true;
  }

  bool is_ok = false;

  ArrayView<T> values(m_value->view());
  if (group.itemFamily()==itemFamily()){
    is_ok = true;
    // TRES IMPORTANT
    //TODO doit utiliser une indirection et une hierarchie entre groupe
    // Enfin, affecte la valeur \a v à toutes les entités du groupe.
    //ValueType& var_value = this->value();
    ENUMERATE_ITEM(i,group){
      Item elem = *i;
      values[elem.localId()] = v;
    }
  }

  if (is_ok)
    return false;

  eItemKind group_kind = group.itemKind();

  error() << "The type of elements (" << itemKindName(group_kind)
          << ") of the group `" << group.name() << "' does not match \n"
          << "the type of the variable (" << itemKindName(this->itemKind()) << ").";
  return true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename T> void VariableArrayT<T>::
copyItemsValues(Int32ConstArrayView source, Int32ConstArrayView destination)
{
  ARCANE_ASSERT(source.size()==destination.size(),
                ("Impossible to copy: source and destination of different sizes !"));
  ArrayView<T> value =  m_value->view();
  const Integer size = source.size();
  for(Integer i=0; i<size; ++i )
    value[destination[i]] = value[source[i]];
  syncReferences();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename T> void VariableArrayT<T>::
copyItemsMeanValues(Int32ConstArrayView first_source,
                    Int32ConstArrayView second_source,
                    Int32ConstArrayView destination)
{
  ARCANE_ASSERT((first_source.size()==destination.size()) && (second_source.size()==destination.size()),
                ("Impossible to copy: source and destination of different sizes !"));
  ArrayView<T> value =  m_value->view();
  const Integer size = first_source.size();
  for(Integer i=0; i<size; ++i ) {
    value[destination[i]] = (T)((value[first_source[i]]+value[second_source[i]])/2);
  }
  syncReferences();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<> void VariableArrayT<String>::
copyItemsMeanValues(Int32ConstArrayView first_source,
                    Int32ConstArrayView second_source,
                    Int32ConstArrayView destination);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename T> void VariableArrayT<T>::
compact(Int32ConstArrayView new_to_old_ids)
{
  if (isPartial()) {
    debug(Trace::High) << "Skip compact for partial variable " << name();
    return;
  }

  UniqueArray<T> old_value(constValueView());
  Integer new_size = new_to_old_ids.size();
  m_value->resize(new_size);
  ArrayView<T> current_value = m_value->view();
  if (arcaneIsCheck()){
    for( Integer i=0; i<new_size; ++i )
      current_value.setAt(i,old_value.at(new_to_old_ids[i]));
  }
  else{
    for( Integer i=0; i<new_size; ++i )
			RawCopy<T>::copy(current_value[i], old_value[ new_to_old_ids[i] ]); // current_value[i] = old_value[ new_to_old_ids[i] ];
  }
  syncReferences();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename T> void VariableArrayT<T>::
setIsSynchronized()
{
  setIsSynchronized(itemGroup());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename T> void VariableArrayT<T>::
setIsSynchronized(const ItemGroup& group)
{
  ARCANE_UNUSED(group);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename T> void VariableArrayT<T>::
_internalResize(const VariableResizeArgs& resize_args)
{
  Int32 new_size = resize_args.newSize();
  Int32 nb_additional_element = resize_args.nbAdditionalCapacity();
  bool use_no_init = resize_args.isUseNoInit();

  auto* value_internal = m_value->_internal();

  if (nb_additional_element!=0){
    Integer capacity = value_internal->capacity();
    if (new_size>capacity)
      value_internal->reserve(new_size+nb_additional_element);
  }
  eDataInitialisationPolicy init_policy = getGlobalDataInitialisationPolicy();
  // Si la nouvelle taille est supérieure à l'ancienne,
  // initialise les nouveaux éléments suivant
  // la politique voulue
  Integer current_size = m_value->view().size();
  if (!isUsed()){
    // Si la variable n'est plus utilisée, libérée la mémoire
    // associée.
    value_internal->dispose();
  }
  if (use_no_init)
    value_internal->_internalDeprecatedValue().resizeNoInit(new_size);
  else
    value_internal->resize(new_size);
  if (new_size>current_size){
    if (init_policy==DIP_InitWithDefault){
      ArrayView<T> values = this->valueView();
      for(Integer i=current_size; i<new_size; ++i)
        values[i] = T();
    }
    else{
      bool use_nan = (init_policy==DIP_InitWithNan);
      bool use_nan2 = (init_policy==DIP_InitInitialWithNanResizeWithDefault) && !_hasValidData();
      if (use_nan || use_nan2){
        ArrayView<T> view = this->valueView();
        DataTypeTraitsT<T>::fillNan(view.subView(current_size,new_size-current_size));
      }
    }
  }

  // Compacte la mémoire si demandé
  if (_wantShrink()){
    if (m_value->view().size() < value_internal->capacity()){
      value_internal->shrink();
    }
  }

  // Controle si toutes les modifs après le dispose n'ont pas altéré l'état de l'allocation
  // Dans le cas d'une variable non utilisée, la capacité max autorisée est
  // égale à celle d'un vecteur Simd de la plateforme.
  // (cela ne peut pas être 0 car la classe Array doit allouer au moins un
  // élément si on utilise un allocateur spécifique ce qui est le cas
  // pour les variables.
  Int64 capacity = value_internal->capacity();
  if ( !((isUsed() || capacity<=AlignedMemoryAllocator::simdAlignment())) )
    ARCANE_FATAL("Wrong unused data size {0}",capacity);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> void VariableArrayT<DataType>::
resizeWithReserve(Integer n,Integer nb_additional)
{
  _resize(VariableResizeArgs(n,nb_additional));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> void VariableArrayT<DataType>::
shrinkMemory()
{
  m_value->_internal()->shrink();
  syncReferences();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> Integer VariableArrayT<DataType>::
capacity()
{
  return m_value->_internal()->capacity();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> void VariableArrayT<DataType>::
fill(const DataType& value)
{
  m_value->view().fill(value);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> void VariableArrayT<DataType>::
fill(const DataType& value,const ItemGroup& group)
{
  ARCANE_UNUSED(group);
  this->fill(value);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> void
VariableArrayT<DataType>::
swapValues(ThatClass& rhs)
{
  _checkSwapIsValid(&rhs);
  // TODO: regarder s'il faut que les deux variables aient le même nombre
  // d'éléments mais a priori cela ne semble pas indispensable.
  m_value->swapValues(rhs.m_value);
  // Il faut mettre à jour les références pour cette variable et \a rhs.
  syncReferences();
  rhs.syncReferences();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// SDP: Specialisation
template<> void VariableArrayT<String>::
copyItemsMeanValues(Int32ConstArrayView first_source,
                    Int32ConstArrayView second_source,
                    Int32ConstArrayView destination)
{
  Integer dsize = destination.size();
  bool is_ok = (first_source.size()==dsize) && (second_source.size()==dsize);
  if (!is_ok)
		ARCANE_FATAL("Unable to copy: source and destination of different sizes !");

  ArrayView<String> value =  m_value->view();
  const Integer size = first_source.size();
  for(Integer i=0; i<size; ++i )
    value[destination[i]] = value[first_source[i]];
  syncReferences();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> auto VariableArrayT<DataType>::
value() -> ValueType&
{
  return m_value->_internal()->_internalDeprecatedValue();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_INTERNAL_INSTANTIATE_TEMPLATE_FOR_NUMERIC_DATATYPE(VariableArrayT);
template class VariableArrayT<String>;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
