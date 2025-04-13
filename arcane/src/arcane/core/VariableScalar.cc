// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VariableScalar.cc                                           (C) 2000-2025 */
/*                                                                           */
/* Variable scalaire.                                                        */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/VariableScalar.h"

#include "arcane/utils/NotSupportedException.h"
#include "arcane/utils/ArgumentException.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/TraceInfo.h"
#include "arcane/utils/Ref.h"
#include "arcane/utils/ITraceMng.h"

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
#include "arcane/core/internal/IVariableMngInternal.h"

#include "arcane/core/datatype/DataStorageBuildInfo.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<class DataType>
class ScalarVariableDiff
: public VariableDiff<DataType>
{
  typedef VariableDataTypeTraitsT<DataType> VarDataTypeTraits;
  typedef typename VariableDiff<DataType>::DiffInfo DiffInfo;

 public:
  //TODO: a simplifier car recopie de ArrayVariableDiff
  // mais pour les variables scalaires il n'y a qu'une valeur et aucun groupe
  // associé.
  VariableComparerResults
  check(IVariable* var,ConstArrayView<DataType> ref,ConstArrayView<DataType> current,
        const VariableComparerArgs& compare_args)
  {
    const int max_print = compare_args.maxPrint();
    const bool compare_ghost = compare_args.isCompareGhost();
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
    Integer ref_size = ref.size();
    ENUMERATE_ITEM(i,group){
      const Item& item = *i;
      if (!item.isOwn() && !compare_ghost)
        continue;
      Integer index = item.localId();
      if (group_index_table){
        index = (*group_index_table)[index];
        if (index<0)
          continue;
      }

      DataType diff = DataType();
      if (index>=ref_size){
        ++nb_diff;
        compare_failed = true;
      }
      else{
        DataType dref = ref[index];
        DataType dcurrent = current[index];
        if (VarDataTypeTraits::verifDifferent(dref,dcurrent,diff,true)){
          this->m_diffs_info.add(DiffInfo(dcurrent,dref,diff,item,NULL_ITEM_ID));
          ++nb_diff;
        }
      }
    }
    if (compare_failed){
      Int32 sid = pm->commRank();
      const String& var_name = var->name();
      msg->pinfo() << "Processor " << sid << " : "
                   << " Unable to compare : elements numbers are different !"
                   << " pour la variable " << var_name << " ref_size=" << ref_size;
        
    }
    if (nb_diff!=0)
      this->_sortAndDump(var,pm,max_print);

    return VariableComparerResults(nb_diff);
  }

  VariableComparerResults
  checkReplica(IVariable* var, const DataType& var_value,
               const VariableComparerArgs& compare_args)
  {
    IParallelMng* replica_pm = compare_args.replicaParallelMng();
    ARCANE_CHECK_POINTER(replica_pm);
    const int max_print = compare_args.maxPrint();
    // Appelle la bonne spécialisation pour être certain que le type template possède
    // la réduction.
    using ReduceType = typename VariableDataTypeTraitsT<DataType>::HasReduceMinMax;
    if constexpr(std::is_same<TrueType,ReduceType>::value)
      return _checkReplica2(replica_pm, var_value);
    ARCANE_UNUSED(replica_pm);
    ARCANE_UNUSED(var);
    ARCANE_UNUSED(var_value);
    ARCANE_UNUSED(max_print);
    throw NotSupportedException(A_FUNCINFO);
  }

 private:

  VariableComparerResults
  _checkReplica2(IParallelMng* pm, const DataType& var_value)
  {
    Int32 nb_rank = pm->commSize();
    if (nb_rank==1)
      return {};

    DataType max_value = pm->reduce(Parallel::ReduceMax,var_value);
    DataType min_value = pm->reduce(Parallel::ReduceMin,var_value);

    Integer nb_diff = 0;
    DataType diff = DataType();
    if (VarDataTypeTraits::verifDifferent(min_value,max_value,diff,true)){
      this->m_diffs_info.add(DiffInfo(min_value,max_value,diff,0,NULL_ITEM_ID));
      ++nb_diff;
    }
    return VariableComparerResults(nb_diff);
  }

};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename T> VariableScalarT<T>::
VariableScalarT(const VariableBuildInfo& v,const VariableInfo& info)
: Variable(v,info)
, m_value(nullptr)
{
  IDataFactoryMng* df = v.dataFactoryMng();
  DataStorageBuildInfo storage_build_info(v.traceMng());
  String storage_full_type = info.storageTypeInfo().fullName();
  Ref<IData> data = df->createSimpleDataRef(storage_full_type,storage_build_info);
  m_value = dynamic_cast<ValueDataType*>(data.get());
  _setData(makeRef(m_value));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename T> auto VariableScalarT<T>::
getReference(const VariableBuildInfo& vb,const VariableInfo& vi) -> ThatClass*
{
  ThatClass* true_ptr = 0;
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

template<typename T> VariableScalarT<T>* VariableScalarT<T>::
getReference(IVariable* var)
{
  if (!var)
    throw ArgumentException(A_FUNCINFO,"null variable");
  ThatClass* true_ptr = dynamic_cast<ThatClass*>(var);
  if (!true_ptr)
    ARCANE_FATAL("Can not build a reference from variable {0}",var->name());
  return true_ptr;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
// Utilise une fonction Helper afin de spécialiser l'appel dans le
// cas du type 'Byte' car ArrayVariableDiff::checkReplica() utilise
// une réduction Min/Max et cela n'existe pas en MPI pour le type Byte.
namespace
{
  template <typename T> VariableComparerResults
  _checkIfSameOnAllReplicaHelper(IVariable* var, const T& value,
                                 const VariableComparerArgs& compare_args)
  {
    ScalarVariableDiff<T> csa;
    return csa.checkReplica(var, value, compare_args);
  }

  // Spécialisation pour le type 'Byte' qui ne supporte pas les réductions.
  VariableComparerResults
  _checkIfSameOnAllReplicaHelper(IVariable* var, const Byte& value,
                                 const VariableComparerArgs& compare_args)
  {
    Integer int_value = value;
    ScalarVariableDiff<Integer> csa;
    return csa.checkReplica(var, int_value, compare_args);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename T> VariableComparerResults VariableScalarT<T>::
_compareVariable(const VariableComparerArgs& compare_args)
{
  switch (compare_args.compareMode()) {
  case VariableComparerArgs::eCompareMode::Same: {

    if (itemKind() == IK_Particle)
      return {};
    IDataReader* reader = compare_args.dataReader();
    ARCANE_CHECK_POINTER(reader);
    T from(value());
    T ref = T();
    Ref<IScalarDataT<T>> ref_data(m_value->cloneTrueEmptyRef());
    reader->read(this, ref_data.get());
    ref = ref_data->value();
    ConstArrayView<T> from_array(1, &from);
    ConstArrayView<T> ref_array(1, &ref);
    ScalarVariableDiff<T> csa;
    VariableComparerResults r = csa.check(this, ref_array, from_array, compare_args);
    return r;
  }
  case VariableComparerArgs::eCompareMode::Sync:
    return {};
  case VariableComparerArgs::eCompareMode::SameReplica: {
    VariableComparerResults r = _checkIfSameOnAllReplicaHelper(this, value(), compare_args);
    return r;
  }
  }
  ARCANE_FATAL("Invalid value for compare mode '{0}'", (int)compare_args.compareMode());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename T> void VariableScalarT<T>::
print(std::ostream& o) const
{
  o << m_value->value();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename T> void VariableScalarT<T>::
synchronize()
{
  // Rien à faire pour les variables scalaires
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename T> void VariableScalarT<T>::
synchronize(Int32ConstArrayView local_ids)
{
  // Rien à faire pour les variables scalaires
  ARCANE_UNUSED(local_ids);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename T> Real VariableScalarT<T>::
allocatedMemory() const
{
  return static_cast<Real>(sizeof(T));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename T> void VariableScalarT<T>::
copyItemsValues(Int32ConstArrayView source, Int32ConstArrayView destination)
{
  ARCANE_UNUSED(source);
  ARCANE_UNUSED(destination);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename T> void VariableScalarT<T>::
copyItemsMeanValues(Int32ConstArrayView first_source,
                    Int32ConstArrayView second_source,
                    Int32ConstArrayView destination)
{
  ARCANE_UNUSED(first_source);
  ARCANE_UNUSED(second_source);
  ARCANE_UNUSED(destination);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename T> void VariableScalarT<T>::
compact(Int32ConstArrayView new_to_old_ids)
{
  ARCANE_UNUSED(new_to_old_ids);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename T> void VariableScalarT<T>::
setIsSynchronized()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename T> void VariableScalarT<T>::
setIsSynchronized(const ItemGroup& group)
{
  ARCANE_UNUSED(group);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> void
VariableScalarT<DataType>::
swapValues(ThatClass& rhs)
{
  _checkSwapIsValid(&rhs);
  m_value->swapValues(rhs.m_value);
  // Il faut mettre à jour les références pour cette variable et \a rhs.
  syncReferences();
  rhs.syncReferences();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_INTERNAL_INSTANTIATE_TEMPLATE_FOR_NUMERIC_DATATYPE(VariableScalarT);
template class VariableScalarT<String>;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
