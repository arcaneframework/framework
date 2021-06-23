// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshMaterialVariableArray.cc                                (C) 2000-2018 */
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
#include "arcane/materials/MeshMaterialVariablePrivate.h"
#include "arcane/materials/MatItemEnumerator.h"
#include "arcane/materials/ComponentItemVectorView.h"

#include "arcane/Array2Variable.h"
#include "arcane/VariableRefArray2.h"
#include "arcane/MeshVariable.h"
#include "arcane/ISerializer.h"

#include "arcane/materials/ItemMaterialVariableBaseT.H"

#include "arcane/datatype/DataTypeTraits.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE
MATERIALS_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> void
MaterialVariableArrayTraits<DataType>::
copyTo(ConstArray2View<DataType> input,Int32ConstArrayView input_indexes,
       Array2View<DataType> output,Int32ConstArrayView output_indexes)
{
  // TODO: vérifier tailles des indexes et des dim2Size() identiques
  Integer nb_value = input_indexes.size();
  Integer dim2_size = input.dim2Size();
  for( Integer i=0; i<nb_value; ++i ){
    auto xo = output[ output_indexes[i] ];
    auto xi = input[ input_indexes[i] ];
    for( Integer j=0; j<dim2_size; ++j )
      xo[j] = xi[j];
  }
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
  ConstArrayView<MatVarIndex> mvi_indexes = component_view.matvarIndexes();

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
dumpValues(ostream& ostr)
{
  ARCANE_UNUSED(ostr);
  throw NotImplementedException(A_FUNCINFO);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> void
ItemMaterialVariableArray<DataType>::
dumpValues(ostream& ostr,AllEnvCellVectorView view)
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
  const Integer nb_count = DataTypeTraitsT<DataType>::nbBasicType();
  typedef typename DataTypeTraitsT<DataType>::BasicType BasicType;
  const eDataType data_type = DataTypeTraitsT<BasicType>::type();
  ItemVectorView ids_view(family->itemsInternal(),ids);
  Int64 dim2_size = m_global_variable->value().dim2Size();
  bool has_mat = this->space()!=MatVarSpace::Environment;
  switch(sbuf->mode()){
  case ISerializer::ModeReserve:
    {
      Int64 nb_val = 0;
      ENUMERATE_ALLENVCELL(iallenvcell,mat_mng,ids_view){
        ++nb_val; // 1 valeur pour le milieu
        ENUMERATE_CELL_ENVCELL(ienvcell,(*iallenvcell)){
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
      Int64 nb_value = sbuf->getInt64();
      Int64 nb_basic_value = nb_value * nb_count;
      basic_values.resize(nb_basic_value);
      sbuf->getSpan(basic_values);
      if (dim2_size!=0){
        Span2<DataType> data_values(reinterpret_cast<DataType*>(basic_values.data()),nb_value,dim2_size);
        Int64 index = 0;
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
  for( PrivatePartType* pvar : m_vars.range() ){
    if (pvar){
      Integer dim1_size = pvar->value().dim1Size();
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
  Integer dim2_size = m_vars[0]->value().dim2Size();
  return (Int32)sizeof(DataType) * dim2_size;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> void
ItemMaterialVariableArray<DataType>::
copyToBuffer(ConstArrayView<MatVarIndex> matvar_indexes,ByteArrayView bytes) const
{
  Integer dim2_size = m_vars[0]->value().dim2Size();
  Integer one_data_size = dataTypeSize();

  // TODO: Vérifier que la taille est un multiple de 'one_data_size' et que
  // l'alignement est correct.
  const Integer value_size = bytes.size() / one_data_size;
  Array2View<DataType> values((DataType*)bytes.unguardedBasePointer(),value_size,dim2_size);
  for( Integer z=0; z<value_size; ++z ){
    values[z].copy(value(matvar_indexes[z]));
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> void
ItemMaterialVariableArray<DataType>::
copyFromBuffer(ConstArrayView<MatVarIndex> matvar_indexes,ByteConstArrayView bytes)
{
  Integer dim2_size = m_vars[0]->value().dim2Size();
  Integer one_data_size = dataTypeSize();

  // TODO: Vérifier que la taille est un multiple de 'one_data_size' et que
  // l'alignement est correct.
  const Integer value_size = bytes.size() / one_data_size;
  ConstArray2View<DataType> values((const DataType*)bytes.unguardedBasePointer(),value_size,dim2_size);
  for( Integer z=0; z<value_size; ++z ){
    setValue(matvar_indexes[z],values[z]);
  }
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

template<typename ItemType,typename DataType>
MeshMaterialVariableArray<ItemType,DataType>*
MeshMaterialVariableArray<ItemType,DataType>::
getReference(const MaterialVariableBuildInfo& v,MatVarSpace mvs)
{
  return getReference(v,v.materialMng(),mvs);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename ItemType,typename DataType>
MeshMaterialVariableArray<ItemType,DataType>*
MeshMaterialVariableArray<ItemType,DataType>::
getReference(const VariableBuildInfo& v,IMeshMaterialMng* mm,MatVarSpace mvs)
{
  return ReferenceGetter::getReference(v,mm,mvs);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template class ItemMaterialVariableBase< MaterialVariableArrayTraits<Real> >;
template class ItemMaterialVariableBase< MaterialVariableArrayTraits<Byte> >;
template class ItemMaterialVariableBase< MaterialVariableArrayTraits<Int16> >;
template class ItemMaterialVariableBase< MaterialVariableArrayTraits<Int32> >;
template class ItemMaterialVariableBase< MaterialVariableArrayTraits<Int64> >;
template class ItemMaterialVariableBase< MaterialVariableArrayTraits<Real2> >;
template class ItemMaterialVariableBase< MaterialVariableArrayTraits<Real3> >;
template class ItemMaterialVariableBase< MaterialVariableArrayTraits<Real2x2> >;
template class ItemMaterialVariableBase< MaterialVariableArrayTraits<Real3x3> >;

template class ItemMaterialVariableArray<Real>;
template class ItemMaterialVariableArray<Byte>;
template class ItemMaterialVariableArray<Int16>;
template class ItemMaterialVariableArray<Int32>;
template class ItemMaterialVariableArray<Int64>;
template class ItemMaterialVariableArray<Real2>;
template class ItemMaterialVariableArray<Real3>;
template class ItemMaterialVariableArray<Real2x2>;
template class ItemMaterialVariableArray<Real3x3>;

template class MeshMaterialVariableArray<Cell,Byte>;
template class MeshMaterialVariableArray<Cell,Real>;
template class MeshMaterialVariableArray<Cell,Int16>;
template class MeshMaterialVariableArray<Cell,Int32>;
template class MeshMaterialVariableArray<Cell,Int64>;
template class MeshMaterialVariableArray<Cell,Real2>;
template class MeshMaterialVariableArray<Cell,Real3>;
template class MeshMaterialVariableArray<Cell,Real2x2>;
template class MeshMaterialVariableArray<Cell,Real3x3>;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MATERIALS_END_NAMESPACE
ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
