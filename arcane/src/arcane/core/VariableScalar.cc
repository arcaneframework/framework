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
  Integer
  check(IVariable* var,ConstArrayView<DataType> ref,ConstArrayView<DataType> current,
        int max_print,bool compare_ghost)
  {
    ItemGroup group = var->itemGroup();
    if (group.null())
      return 0;
    IMesh* mesh = group.mesh();
    if (!mesh)
      return 0;
    ITraceMng* msg = mesh->traceMng();
    IParallelMng* pm = mesh->parallelMng();

    GroupIndexTable * group_index_table = (var->isPartial())?group.localIdToIndex().get():0;

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

    return nb_diff;
  }

  Integer checkReplica(IParallelMng* pm,IVariable* var,const DataType& var_value,
                       Integer max_print)
  {
    // Appelle la bonne spécialisation pour être sur que le type template possède
    // la réduction.
    using ReduceType = typename VariableDataTypeTraitsT<DataType>::HasReduceMinMax;
    if constexpr(std::is_same<TrueType,ReduceType>::value)
      return _checkReplica2(pm,var_value);
    ARCANE_UNUSED(pm);
    ARCANE_UNUSED(var);
    ARCANE_UNUSED(var_value);
    ARCANE_UNUSED(max_print);
    throw NotSupportedException(A_FUNCINFO);
  }

 private:

  Integer _checkReplica2(IParallelMng* pm,const DataType& var_value)
  {
    Int32 nb_rank = pm->commSize();
    if (nb_rank==1)
      return 0;

    DataType max_value = pm->reduce(Parallel::ReduceMax,var_value);
    DataType min_value = pm->reduce(Parallel::ReduceMin,var_value);

    Integer nb_diff = 0;
    DataType diff = DataType();
    if (VarDataTypeTraits::verifDifferent(min_value,max_value,diff,true)){
      this->m_diffs_info.add(DiffInfo(min_value,max_value,diff,0,NULL_ITEM_ID));
      ++nb_diff;
    }
    return nb_diff;
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

template<typename T> Integer VariableScalarT<T>::
checkIfSame(IDataReader* reader,int max_print,bool compare_ghost)
{
  if (itemKind()==IK_Particle)
    return 0;
  T from(value());
  T ref = T();
  Ref< IScalarDataT<T> > ref_data(m_value->cloneTrueEmptyRef());
  reader->read(this,ref_data.get());
  ref = ref_data->value();
  ConstArrayView<T> from_array(1,&from);
  ConstArrayView<T> ref_array(1,&ref);
  ScalarVariableDiff<T> csa;
  return csa.check(this,ref_array,from_array,max_print,compare_ghost);
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
                                 const T& value,Integer max_print)
  {
    ScalarVariableDiff<T> csa;
    return csa.checkReplica(pm,var,value,max_print);
  }

  // Spécialisation pour le type 'Byte' qui ne supporte pas les réductions.
  Integer
  _checkIfSameOnAllReplicaHelper(IParallelMng* pm,IVariable* var,
                                 const Byte& value,Integer max_print)
  {
    Integer int_value = value;
    ScalarVariableDiff<Integer> csa;
    return csa.checkReplica(pm,var,int_value,max_print);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename T> Integer VariableScalarT<T>::
_checkIfSameOnAllReplica(IParallelMng* replica_pm,Integer max_print)
{
  return _checkIfSameOnAllReplicaHelper(replica_pm,this,value(),max_print);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename T> VariableComparerResults VariableScalarT<T>::
_compareVariable(const VariableComparerArgs& compare_args)
{
  ARCANE_FATAL("NotImplemented");
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
