// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshMaterialVariableArray.cc                                (C) 2000-2024 */
/*                                                                           */
/* Variable tableau sur un matériau du maillage.                             */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/NotImplementedException.h"
#include "arcane/utils/Real2.h"
#include "arcane/utils/Real3.h"
#include "arcane/utils/Real2x2.h"
#include "arcane/utils/Real3x3.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/CheckedConvert.h"

#include "arcane/materials/MeshMaterialVariable.h"
#include "arcane/materials/MaterialVariableBuildInfo.h"
#include "arcane/materials/MeshMaterialVariableSynchronizerList.h"
#include "arcane/materials/MatItemEnumerator.h"
#include "arcane/materials/ComponentItemVectorView.h"
#include "arcane/materials/internal/MeshMaterialVariablePrivate.h"

#include "arcane/core/Array2Variable.h"
#include "arcane/core/VariableRefArray2.h"
#include "arcane/core/MeshVariable.h"
#include "arcane/core/ISerializer.h"
#include "arcane/core/internal/IVariableInternal.h"

#include "arcane/materials/ItemMaterialVariableBaseT.H"

#include "arcane/datatype/DataTypeTraits.h"
#include "arcane/datatype/DataStorageBuildInfo.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> void
MaterialVariableArrayTraits<DataType>::
copyTo(SmallSpan2<const DataType> input, SmallSpan<const Int32> input_indexes,
       SmallSpan2<DataType> output, SmallSpan<const Int32> output_indexes,
       const RunQueue& queue)
{
  // TODO: vérifier tailles des indexes et des dim2Size() identiques
  Integer nb_value = input_indexes.size();
  Integer dim2_size = input.dim2Size();

  auto command = makeCommand(queue);
  command << RUNCOMMAND_LOOP1(iter, nb_value)
  {
    auto [i] = iter();
    auto xo = output[output_indexes[i]];
    auto xi = input[ input_indexes[i] ];
    for( Integer j=0; j<dim2_size; ++j )
      xo[j] = xi[j];
  };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> void
MaterialVariableArrayTraits<DataType>::
resizeAndFillWithDefault(ValueDataType* data,ContainerType& container,Integer dim1_size)
{
  ContainerType& values = data->_internal()->_internalDeprecatedValue();
  Integer dim2_size = values.dim2Size();

  //TODO: faire une version de Array2 qui spécifie une valeur à donner
  // pour initialiser lors d'un resize() (comme pour Array::resize()).
  container.resize(dim1_size,dim2_size);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> void
MaterialVariableArrayTraits<DataType>::
resizeWithReserve(PrivatePartType* var, Integer dim1_size, Real reserve_ratio)
{
  // Pour éviter de réallouer à chaque fois qu'il y a une augmentation du
  // nombre de mailles matériaux, alloue un petit peu plus que nécessaire.
  // Par défaut, on alloue 5% de plus.
  Int32 nb_add = static_cast<Int32>(dim1_size * reserve_ratio);
  var->_internalApi()->resizeWithReserve(dim1_size, nb_add);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> void
MaterialVariableArrayTraits<DataType>::
saveData(IMeshComponent* component,IData* data,
         Array<ContainerViewType>& cviews)
{
  // Pour les tableaux 2D, on redimensionne directement
  // \a values en ajoutant le nombre d'éléments pour notre
  // component.
  ConstArrayView<Array2View<DataType>> views = cviews;
  if (views.empty())
    return;
  auto* true_data = dynamic_cast<ValueDataType*>(data);
  ARCANE_CHECK_POINTER(true_data);
  ContainerType& values = true_data->_internal()->_internalDeprecatedValue();
  ComponentItemVectorView component_view = component->view();

  Integer dim2_size = views[0].dim2Size();
  Integer current_dim1_size = values.dim1Size();
  Integer added_dim1_size = component_view.nbItem();
  values.resizeNoInit(current_dim1_size+added_dim1_size,dim2_size);
  ConstArrayView<MatVarIndex> mvi_indexes = component_view._matvarIndexes();

  for( Integer i=0, n=mvi_indexes.size(); i<n; ++i ){
    MatVarIndex mvi = mvi_indexes[i];
    values[i+current_dim1_size].copy(views[mvi.arrayIndex()][mvi.valueIndex()]);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> ItemMaterialVariableArray<DataType>::
ItemMaterialVariableArray(const MaterialVariableBuildInfo& v,PrivatePartType* global_var,
                          VariableRef* global_var_ref,MatVarSpace mvs)
: BaseClass(v,global_var,global_var_ref,mvs)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> void
ItemMaterialVariableArray<DataType>::
dumpValues(std::ostream& ostr)
{
  ARCANE_UNUSED(ostr);
  throw NotImplementedException(A_FUNCINFO);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> void
ItemMaterialVariableArray<DataType>::
dumpValues(std::ostream& ostr,AllEnvCellVectorView view)
{
  ARCANE_UNUSED(ostr);
  ARCANE_UNUSED(view);
  throw NotImplementedException(A_FUNCINFO);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> void
ItemMaterialVariableArray<DataType>::
serialize(ISerializer* sbuf,Int32ConstArrayView ids)
{
  // TODO: essayer de fusionner cette methode avec la variante scalaire.
  IItemFamily* family = m_global_variable->itemFamily();
  if (!family)
    return;
  IMeshMaterialMng* mat_mng = m_p->materialMng();
  const Int32 nb_count = DataTypeTraitsT<DataType>::nbBasicType();
  typedef typename DataTypeTraitsT<DataType>::BasicType BasicType;
  const eDataType data_type = DataTypeTraitsT<BasicType>::type();
  ItemVectorView ids_view(family->view(ids));
  Int32 dim2_size = m_global_variable->valueView().dim2Size();
  bool has_mat = this->space()!=MatVarSpace::Environment;
  switch(sbuf->mode()){
  case ISerializer::ModeReserve:
    {
      Int64 nb_val = 0;
      ENUMERATE_ALLENVCELL(iallenvcell,mat_mng,ids_view){
        ENUMERATE_CELL_ENVCELL(ienvcell,(*iallenvcell)){
          ++nb_val; // 1 valeur pour le milieu
          EnvCell envcell = *ienvcell;
          if (has_mat)
            nb_val += envcell.nbMaterial(); // 1 valeur par matériau du milieu.
        }
      }
      sbuf->reserve(m_global_variable->fullName());
      sbuf->reserve(DT_Int64,1);  // Pour le nombre de valeurs.
      Int64 nb_basic_value = nb_val * nb_count * dim2_size;
      sbuf->reserveSpan(data_type,nb_basic_value);  // Pour le nombre de valeurs.
    }
    break;
  case ISerializer::ModePut:
    {
      UniqueArray<DataType> values;
      ENUMERATE_ALLENVCELL(iallenvcell,mat_mng,ids_view){
        ENUMERATE_CELL_ENVCELL(ienvcell,(*iallenvcell)){
          values.addRange(value(ienvcell._varIndex()));
          if (has_mat){
            ENUMERATE_CELL_MATCELL(imatcell,(*ienvcell)){
              values.addRange(value(imatcell._varIndex()));
            }
          }
        }
      }
      Int64 nb_value = values.largeSize();
      Span<const BasicType> basic_values(reinterpret_cast<BasicType*>(values.data()),nb_value*nb_count);
      sbuf->put(m_global_variable->fullName());
      sbuf->putInt64(nb_value);
      sbuf->putSpan(basic_values);
    }
    break;
  case ISerializer::ModeGet:
    {
      UniqueArray<BasicType> basic_values;
      String ref_name;
      sbuf->get(ref_name);
      if (m_global_variable->fullName()!=ref_name)
        ARCANE_FATAL("Bad serialization expected={0} found={1}",m_global_variable->fullName(),ref_name);
      Int32 nb_value = CheckedConvert::toInt32(sbuf->getInt64());
      Int32 nb_basic_value = nb_value * nb_count;
      basic_values.resize(nb_basic_value);
      sbuf->getSpan(basic_values);
      if (dim2_size!=0){
        SmallSpan2<DataType> data_values(reinterpret_cast<DataType*>(basic_values.data()),nb_value,dim2_size);
        Int32 index = 0;
        ENUMERATE_ALLENVCELL(iallenvcell,mat_mng,ids_view){
          ENUMERATE_CELL_ENVCELL(ienvcell,(*iallenvcell)){
            EnvCell envcell = *ienvcell;
            setValue(ienvcell._varIndex(),data_values[index]);
            ++index;
            if (has_mat){
              ENUMERATE_CELL_MATCELL(imatcell,envcell){
                setValue(imatcell._varIndex(),data_values[index]);
                ++index;
              }
            }
          }
        }
      }
    }
    break;
  default:
    ARCANE_THROW(NotSupportedException,"Invalid serialize");
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> void
ItemMaterialVariableArray<DataType>::
synchronize()
{
  MeshMaterialVariableSynchronizerList mmvsl(m_p->materialMng());
  mmvsl.add(this);
  mmvsl.apply();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> void
ItemMaterialVariableArray<DataType>::
synchronize(MeshMaterialVariableSynchronizerList& sync_list)
{
  sync_list.add(this);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> void
ItemMaterialVariableArray<DataType>::
resize(Integer dim2_size)
{
  for( PrivatePartType* pvar : m_vars ){
    if (pvar){
      Integer dim1_size = pvar->valueView().dim1Size();
      pvar->directResize(dim1_size,dim2_size);
    }
  }

  //m_data->value().resize(dim1_size,dim2_size);
  /*info() << "RESIZE(2) AFTER " << fullName()
         << " total=" << m_data->value().totalNbElement()
         << " dim1_size=" << m_data->value().dim1Size()
         << " dim2_size=" << m_data->value().dim2Size()
         << " addr=" << m_data->value().view().unguardedBasePointer();*/
  this->syncReferences();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> Int32
ItemMaterialVariableArray<DataType>::
dataTypeSize() const
{
  Int32 dim2_size = m_vars[0]->valueView().dim2Size();
  return (Int32)sizeof(DataType) * dim2_size;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> void
ItemMaterialVariableArray<DataType>::
_copyToBufferLegacy(SmallSpan<const MatVarIndex> matvar_indexes,Span<std::byte> bytes) const
{
  const Integer one_data_size = dataTypeSize();
  Integer dim2_size = m_vars[0]->valueView().dim2Size();

  // TODO: Vérifier que la taille est un multiple de 'one_data_size' et que
  // l'alignement est correct.
  const Int32 value_size = CheckedConvert::toInt32(bytes.size() / one_data_size);
  Array2View<DataType> values(reinterpret_cast<DataType*>(bytes.data()),value_size,dim2_size);
  for( Integer z=0; z<value_size; ++z ){
    values[z].copy(value(matvar_indexes[z]));
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> void
ItemMaterialVariableArray<DataType>::
copyToBuffer(ConstArrayView<MatVarIndex> matvar_indexes,ByteArrayView bytes) const
{
  auto* ptr = reinterpret_cast<std::byte*>(bytes.data());
  return _copyToBufferLegacy(matvar_indexes,{ptr,bytes.size()});
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> void
ItemMaterialVariableArray<DataType>::
_copyFromBufferLegacy(SmallSpan<const MatVarIndex> matvar_indexes,Span<const std::byte> bytes)
{
  const Integer one_data_size = dataTypeSize();
  Integer dim2_size = m_vars[0]->valueView().dim2Size();

  // TODO: Vérifier que la taille est un multiple de 'one_data_size' et que
  // l'alignement est correct.
  const Integer value_size = CheckedConvert::toInt32(bytes.size() / one_data_size);
  ConstArray2View<DataType> values(reinterpret_cast<const DataType*>(bytes.data()),value_size,dim2_size);
  for( Integer z=0; z<value_size; ++z ){
    setValue(matvar_indexes[z],values[z]);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> void
ItemMaterialVariableArray<DataType>::
copyFromBuffer(ConstArrayView<MatVarIndex> matvar_indexes,ByteConstArrayView bytes)
{
  auto* ptr = reinterpret_cast<const std::byte*>(bytes.data());
  return _copyFromBufferLegacy(matvar_indexes,{ptr,bytes.size()});
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename ItemType,typename DataType>
MeshMaterialVariableArray<ItemType,DataType>::
MeshMaterialVariableArray(const MaterialVariableBuildInfo& v,PrivatePartType* global_var,
                          VariableRefType* global_var_ref,MatVarSpace mvs)
: ItemMaterialVariableArray<DataType>(v,global_var,global_var_ref,mvs)
, m_true_global_variable_ref(global_var_ref) // Sera détruit par la classe de base
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#define ARCANE_INSTANTIATE_MAT(type) \
  template class ItemMaterialVariableBase< MaterialVariableArrayTraits<type> >;\
  template class ItemMaterialVariableArray<type>;\
  template class MeshMaterialVariableArray<Cell,type>;\
  template class MeshMaterialVariableCommonStaticImpl<MeshMaterialVariableArray<Cell,type>>

ARCANE_INSTANTIATE_MAT(Byte);
ARCANE_INSTANTIATE_MAT(Int16);
ARCANE_INSTANTIATE_MAT(Int32);
ARCANE_INSTANTIATE_MAT(Int64);
ARCANE_INSTANTIATE_MAT(Real);
ARCANE_INSTANTIATE_MAT(Real2);
ARCANE_INSTANTIATE_MAT(Real3);
ARCANE_INSTANTIATE_MAT(Real2x2);
ARCANE_INSTANTIATE_MAT(Real3x3);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
