// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshMaterialVariable.h                                      (C) 2000-2022 */
/*                                                                           */
/* Variable sur un matériau du maillage.                                     */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MATERIALS_MESHMATERIALVARIABLE_H
#define ARCANE_MATERIALS_MESHMATERIALVARIABLE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/NotImplementedException.h"
#include "arcane/utils/Array.h"

#include "arcane/core/materials/IMeshMaterialVariable.h"
#include "arcane/core/materials/MatVarIndex.h"

#include "arcane/materials/MaterialsGlobal.h"

// A supprimer
#include "arcane/materials/MeshMaterialVariableFactoryRegisterer.h"
#include "arcane/core/materials/MaterialVariableBuildInfo.h"
#include "arcane/core/materials/IMeshMaterialMng.h"
#include "arcane/core/materials/IMeshMaterialVariableFactoryMng.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class VariableInfo;
class VariableRef;
class VariableBuildInfo;
template <typename ItemType,typename DataTypeT> class MeshVariableScalarRefT;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class MaterialVariableBuildInfo;
class MeshMaterialVariablePrivate;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \ingroup ArcaneMaterials
 * \brief Classe de base des variables matériaux.
 *
 * Cette classe contient l'implémentation des variables matériaux. Elle
 * est interne à Arcane. En général, c'est IMeshMaterialVariable qu'il
 * faut utiliser.
 */
class ARCANE_MATERIALS_EXPORT MeshMaterialVariable
: public IMeshMaterialVariable
{
 public:

  MeshMaterialVariable(const MaterialVariableBuildInfo& v,MatVarSpace mvs);
  ~MeshMaterialVariable();

 public:

  const String& name() const override;
  void addVariableRef(MeshMaterialVariableRef* var_ref) override;
  void removeVariableRef(MeshMaterialVariableRef* var_ref) override;
  MeshMaterialVariableRef* firstReference() const override;
  IVariable* materialVariable(IMeshMaterial* mat) override;

  void setKeepOnChange(bool v) override;
  bool keepOnChange() const override;

  MatVarSpace space() const override;

 public:

  //! @name Gestion des dépendances
  //@{
  void update(IMeshMaterial* mat) override;
  void setUpToDate(IMeshMaterial* mat) override;
  Int64 modifiedTime(IMeshMaterial* mat) override;
  void addDepend(IMeshMaterialVariable* var) override;
  void addDepend(IMeshMaterialVariable* var,const TraceInfo& tinfo) override;
  void addDepend(IVariable* var) override;
  void addDepend(IVariable* var,const TraceInfo& tinfo) override;
  void removeDepend(IMeshMaterialVariable* var) override;
  void removeDepend(IVariable* var) override;
  void setComputeFunction(IMeshMaterialVariableComputeFunction* v) override;
  IMeshMaterialVariableComputeFunction* computeFunction() override;
  void dependInfos(Array<VariableDependInfo>& infos,
                   Array<MeshMaterialVariableDependInfo>& mat_infos) override;
  //@}

 public:

  //! @name Fonctions publiques mais réservées à Arcane pour gérer les synchronisations
  //@{
  virtual Int32 dataTypeSize() const =0;
  virtual void copyToBuffer(ConstArrayView<MatVarIndex> matvar_indexes,ByteArrayView bytes) const =0;
  virtual void copyFromBuffer(ConstArrayView<MatVarIndex> matvar_indexes,ByteConstArrayView bytes) =0;
  //@}

 public:

  /*!
   * \internal
   * Incrémente le compteur de référence.
   */
  void incrementReference();

 protected:

  ITraceMng* _traceMng() const;

 protected:

  MeshMaterialVariablePrivate* m_p;

 private:
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Caractéristiques pour une variable matériaux scalaire.
template<typename DataType>
class MaterialVariableScalarTraits
{
 public:

  typedef DataType ValueType;

  typedef DataType SubViewType;
  typedef DataType SubConstViewType;
  typedef DataType SubInputViewType;
  typedef ArrayView<DataType> ContainerViewType;
  typedef ConstArrayView<DataType> ContainerConstViewType;
  typedef VariableArrayT<DataType> PrivatePartType;
  typedef IArrayDataT<DataType> ValueDataType;
  typedef Array<DataType> ContainerType;
  typedef UniqueArray<DataType> UniqueContainerType;
  typedef VariableRefArrayT<DataType> VariableRefType;

 public:

  ARCANE_MATERIALS_EXPORT static void
  saveData(IMeshComponent* component,IData* data,Array<ContainerViewType>& cviews);
  ARCANE_MATERIALS_EXPORT static void
  copyTo(ConstArrayView<DataType> input,Int32ConstArrayView input_indexes,
         ArrayView<DataType> output,Int32ConstArrayView output_indexes);
  ARCANE_MATERIALS_EXPORT static void
  resizeAndFillWithDefault(ValueDataType* data,ContainerType& container,Integer dim1_size);

  static void setValue(DataType& view,const DataType& v)
  {
    view = v;
  }
  ARCANE_MATERIALS_EXPORT static void
  resizeWithReserve(PrivatePartType* var,Integer new_size);
  static Integer dimension() { return 0; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
//! Caractéristiques pour une variable matériaux tableau.
template<typename DataType>
class MaterialVariableArrayTraits
{
 public:

  typedef DataType ValueType;

  typedef ArrayView<DataType> SubViewType;
  typedef ConstArrayView<DataType> SubConstViewType;
  typedef Span<const DataType> SubInputViewType;
  typedef Array2View<DataType> ContainerViewType;
  typedef ConstArray2View<DataType> ContainerConstViewType;
  typedef Array2VariableT<DataType> PrivatePartType;
  typedef IArray2DataT<DataType> ValueDataType;
  typedef Array2<DataType> ContainerType;
  typedef UniqueArray2<DataType> UniqueContainerType;
  typedef VariableRefArray2T<DataType> VariableRefType;

 public:

  ARCANE_MATERIALS_EXPORT
  static void saveData(IMeshComponent* component,IData* data,
                       Array<ContainerViewType>& cviews);
  ARCANE_MATERIALS_EXPORT
  static void copyTo(ConstArray2View<DataType> input,Int32ConstArrayView input_indexes,
                     Array2View<DataType> output,Int32ConstArrayView output_indexes);
  ARCANE_MATERIALS_EXPORT
  static void resizeAndFillWithDefault(ValueDataType* data,ContainerType& container,
                                       Integer dim1_size);
  static void setValue(ArrayView<DataType> view,const DataType& v)
  {
    view.fill(v);
  }
  static void setValue(ArrayView<DataType> view,Span<const DataType> v)
  {
    view.copy(v);
  }
  static void resizeWithReserve(PrivatePartType* var,Integer new_size)
  {
    var->resize(new_size);
  }
  static Integer dimension() { return 0; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Classe de base des variables matériaux de avec les
 * caractéristiques spécifiées par \a Traits.
 */
template<typename Traits>
class ItemMaterialVariableBase
: public MeshMaterialVariable
{
 public:

  typedef Traits TraitsType;
  typedef typename Traits::ValueType ValueType;
  typedef typename Traits::ValueType DataType;
  typedef ItemMaterialVariableBase<Traits> ThatClass;

  typedef typename Traits::SubViewType SubViewType;
  typedef typename Traits::SubConstViewType SubConstViewType;
  typedef typename Traits::SubInputViewType SubInputViewType;
  typedef typename Traits::ContainerViewType ContainerViewType;
  typedef typename Traits::ContainerConstViewType ContainerConstViewType;
  typedef typename Traits::PrivatePartType PrivatePartType;
  typedef typename Traits::ValueDataType ValueDataType;
  typedef typename Traits::ContainerType ContainerType;
  typedef typename Traits::UniqueContainerType UniqueContainerType;
  typedef typename Traits::VariableRefType VariableRefType;

 public:

  ARCANE_MATERIALS_EXPORT
  ItemMaterialVariableBase(const MaterialVariableBuildInfo& v,
                           PrivatePartType* global_var,
                           VariableRef* global_var_ref,MatVarSpace mvs);
  ARCANE_MATERIALS_EXPORT ~ItemMaterialVariableBase();

 public:

  ARCANE_MATERIALS_EXPORT void syncReferences() override;
  ARCANE_MATERIALS_EXPORT IVariable* globalVariable() const override;
  ARCANE_MATERIALS_EXPORT void buildFromManager(bool is_continue) override;

  ARCANE_MATERIALS_EXPORT Ref<IData> _internalCreateSaveDataRef(Integer nb_value) override;
  ARCANE_MATERIALS_EXPORT void _saveData(IMeshComponent* env,IData* data) override;
  ARCANE_MATERIALS_EXPORT
  void _restoreData(IMeshComponent* component,IData* data,Integer data_index,
                    Int32ConstArrayView ids,bool allow_null_id) override;
  ARCANE_MATERIALS_EXPORT
  void _copyGlobalToPartial(Int32 var_index,Int32ConstArrayView local_ids,
                            Int32ConstArrayView indexes_in_multiple) override;
  ARCANE_MATERIALS_EXPORT
  void _copyPartialToGlobal(Int32 var_index,Int32ConstArrayView local_ids,
                            Int32ConstArrayView indexes_in_multiple) override;
  ARCANE_MATERIALS_EXPORT
  void _initializeNewItems(const ComponentItemListBuilder& list_builder) override;

  ARCANE_MATERIALS_EXPORT void fillPartialValuesWithGlobalValues() override;
  ARCANE_MATERIALS_EXPORT void
  fillPartialValuesWithSuperValues(Int32 level) override;

 public:

  void setValue(MatVarIndex mvi,SubInputViewType v)
  {
    Traits::setValue(m_views[mvi.arrayIndex()][mvi.valueIndex()],v);
  }

  void setFillValue(MatVarIndex mvi,const DataType& v)
  {
    Traits::setValue(m_views[mvi.arrayIndex()][mvi.valueIndex()],v);
  }

  SubConstViewType value(MatVarIndex mvi) const
  {
    return m_views[mvi.arrayIndex()][mvi.valueIndex()];
  }

 protected:

  void _syncFromGlobalVariable();
  PrivatePartType* _trueGlobalVariable()
  {
    return m_global_variable;
  }

  void _init(ArrayView<PrivatePartType*> vars);
  ARCANE_MATERIALS_EXPORT void
  _fillPartialValuesWithSuperValues(MeshComponentList components);

 protected:

  PrivatePartType* m_global_variable;
  VariableRef* m_global_variable_ref;
  //! Variables pour les différents matériaux.
  UniqueArray<PrivatePartType*> m_vars;
  UniqueArray<ContainerViewType> m_views;

 private:
  bool _isValidAndUsedAndGlobalUsed(PrivatePartType* partial_var);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename VariableTrueType>
class VariableReferenceGetter
{
 public:
  static ARCANE_MATERIALS_EXPORT IMeshMaterialVariable*
  getReference(const MaterialVariableBuildInfo& v,MatVarSpace mvs);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \ingroup ArcaneMaterials
 * \brief Variable scalaire sur un matériau du maillage.
 */
template<typename DataType>
class ItemMaterialVariableScalar
: public ItemMaterialVariableBase<MaterialVariableScalarTraits<DataType>>
{
 public:
	
  typedef ItemMaterialVariableBase<MaterialVariableScalarTraits<DataType>> BaseClass;
  typedef MaterialVariableScalarTraits<DataType> Traits;
  typedef ItemMaterialVariableScalar<DataType> ThatClass;

  typedef typename Traits::ContainerViewType ContainerViewType;
  typedef typename Traits::PrivatePartType PrivatePartType;
  typedef typename Traits::ValueDataType ValueDataType;
  typedef typename Traits::ContainerType ContainerType;
  typedef typename Traits::VariableRefType VariableRefType;

 protected:

  ARCANE_MATERIALS_EXPORT
  ItemMaterialVariableScalar(const MaterialVariableBuildInfo& v,
                             PrivatePartType* global_var,
                             VariableRef* global_var_ref,MatVarSpace mvs);

 public:

  ArrayView<DataType>* views() { return this->m_views.data(); }

 public:
  
  DataType operator[](MatVarIndex mvi) const
  {
    return this->m_views[mvi.arrayIndex()][mvi.valueIndex()];
  }

  using BaseClass::setValue;
  using BaseClass::value;

  ARCANE_MATERIALS_EXPORT void synchronize() override;
  ARCANE_MATERIALS_EXPORT void synchronize(MeshMaterialVariableSynchronizerList& sync_list) override;
  ARCANE_MATERIALS_EXPORT void dumpValues(std::ostream& ostr) override;
  ARCANE_MATERIALS_EXPORT void dumpValues(std::ostream& ostr,AllEnvCellVectorView view) override;
  ARCANE_MATERIALS_EXPORT void serialize(ISerializer* sbuffer,Int32ConstArrayView ids) override;

 public:

  ARCANE_MATERIALS_EXPORT
  void fillFromArray(IMeshMaterial* mat,ConstArrayView<DataType> values);
  ARCANE_MATERIALS_EXPORT
  void fillFromArray(IMeshMaterial* mat,ConstArrayView<DataType> values,
                     Int32ConstArrayView indexes);
  ARCANE_MATERIALS_EXPORT void fillToArray(IMeshMaterial* mat,ArrayView<DataType> values);
  ARCANE_MATERIALS_EXPORT void fillToArray(IMeshMaterial* mat,ArrayView<DataType> values,
                                           Int32ConstArrayView indexes);
  ARCANE_MATERIALS_EXPORT void fillPartialValues(const DataType& value);

  ARCANE_MATERIALS_EXPORT Int32 dataTypeSize() const override;
  ARCANE_MATERIALS_EXPORT
  void copyToBuffer(ConstArrayView<MatVarIndex> matvar_indexes,
                    ByteArrayView bytes) const override;
  ARCANE_MATERIALS_EXPORT
  void copyFromBuffer(ConstArrayView<MatVarIndex> matvar_indexes,
                      ByteConstArrayView bytes) override;

 protected:

  using BaseClass::m_p;

 private:
  
  using BaseClass::m_global_variable;
  using BaseClass::m_global_variable_ref;
  using BaseClass::m_vars;
  using BaseClass::m_views;

 private:

  void _synchronizeV1();
  void _synchronizeV2();
  void _synchronizeV3();
  void _synchronizeV4();
  void _synchronizeV5();
  Int64 _synchronize2();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Interface d'accès pour CellMaterialVariableScalarRef.
 */
template<typename ItemType,typename DataType>
class IMeshMaterialVariableScalar
{
 public:

  using VariableRefType = MeshVariableScalarRefT<ItemType,DataType>;

 public:

  virtual ~IMeshMaterialVariableScalar() = default;

 public:

  virtual ArrayView<DataType>* valuesView() = 0;
  virtual void fillFromArray(IMeshMaterial* mat,ConstArrayView<DataType> values) =0;
  virtual void fillFromArray(IMeshMaterial* mat,ConstArrayView<DataType> values,Int32ConstArrayView indexes) =0;
  virtual void fillToArray(IMeshMaterial* mat,ArrayView<DataType> values) =0;
  virtual void fillToArray(IMeshMaterial* mat,ArrayView<DataType> values,Int32ConstArrayView indexes) =0;
  virtual void fillPartialValues(const DataType& value) =0;
  virtual VariableRefType* globalVariableReference() const =0;
  virtual void incrementReference() =0;
  virtual IMeshMaterialVariable* toMeshMaterialVariable() =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \ingroup ArcaneMaterials
 * \brief Variable scalaire sur un matériau du maillage.
 */
template<typename ItemType,typename DataType>
class MeshMaterialVariableScalar
: public ItemMaterialVariableScalar<DataType>
, public IMeshMaterialVariableScalar<ItemType,DataType>
{
 public:
	
  using ThatClass = MeshMaterialVariableScalar<ItemType,DataType>;
  using ThatInterface = IMeshMaterialVariableScalar<ItemType,DataType>;
  using ItemTypeTemplate = ItemType;

  using BaseClass = ItemMaterialVariableScalar<DataType>;
  using VariableRefType = MeshVariableScalarRefT<ItemType,DataType>;
  using PrivatePartType = typename BaseClass::PrivatePartType;

  using ReferenceGetter = VariableReferenceGetter<ThatClass>;
  friend ReferenceGetter;

 protected:

  ARCANE_MATERIALS_EXPORT
  MeshMaterialVariableScalar(const MaterialVariableBuildInfo& v,
                             PrivatePartType* global_var,
                             VariableRefType* global_var_ref,MatVarSpace mvs);
  ARCANE_MATERIALS_EXPORT
  ~MeshMaterialVariableScalar();

 public:

  VariableRefType* globalVariableReference() const final { return m_true_global_variable_ref; }
  void incrementReference() final { BaseClass::incrementReference(); }
  ArrayView<DataType>* valuesView() final { return BaseClass::views(); }
  void fillFromArray(IMeshMaterial* mat,ConstArrayView<DataType> values) final
  {
    return BaseClass::fillFromArray(mat,values);
  }
  void fillFromArray(IMeshMaterial* mat,ConstArrayView<DataType> values,Int32ConstArrayView indexes) final
  {
    BaseClass::fillFromArray(mat,values,indexes);
  }
  void fillToArray(IMeshMaterial* mat,ArrayView<DataType> values) final
  {
    BaseClass::fillToArray(mat,values);
  }
  void fillToArray(IMeshMaterial* mat,ArrayView<DataType> values,Int32ConstArrayView indexes) final
  {
    BaseClass::fillToArray(mat,values,indexes);
  }
  void fillPartialValues(const DataType& value) final { BaseClass::fillPartialValues(value); }
  IMeshMaterialVariable* toMeshMaterialVariable() final { return this; }

 protected:
  
 private:
  
  VariableRefType* m_true_global_variable_ref;

 public:

  static ARCANE_MATERIALS_EXPORT MaterialVariableTypeInfo _buildVarTypeInfo(MatVarSpace space);

 private:

  static ARCANE_MATERIALS_EXPORT IMeshMaterialVariable* _autoCreate1(const MaterialVariableBuildInfo& vb);
  static ARCANE_MATERIALS_EXPORT IMeshMaterialVariable* _autoCreate2(const MaterialVariableBuildInfo& vb);
  static ARCANE_MATERIALS_EXPORT MeshMaterialVariableFactoryRegisterer m_auto_registerer1;
  static ARCANE_MATERIALS_EXPORT MeshMaterialVariableFactoryRegisterer m_auto_registerer2;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \ingroup ArcaneMaterials
 * \brief Variable tableau sur un matériau du maillage.
 */
template<typename DataType>
class ItemMaterialVariableArray
: public ItemMaterialVariableBase<MaterialVariableArrayTraits<DataType>>
{
 public:

  typedef ItemMaterialVariableBase<MaterialVariableArrayTraits<DataType>> BaseClass;
  typedef MaterialVariableArrayTraits<DataType> Traits;

  typedef ItemMaterialVariableArray<DataType> ThatClass;

  typedef typename Traits::ContainerViewType ContainerViewType;
  typedef typename Traits::PrivatePartType PrivatePartType;
  typedef typename Traits::ValueDataType ValueDataType;
  typedef typename Traits::ContainerType ContainerType;
  typedef typename Traits::VariableRefType VariableRefType;

 protected:

  ARCANE_MATERIALS_EXPORT
  ItemMaterialVariableArray(const MaterialVariableBuildInfo& v,
                            PrivatePartType* global_var,
                            VariableRef* global_var_ref,MatVarSpace mvs);
  ARCANE_MATERIALS_EXPORT ~ItemMaterialVariableArray() {}

 public:

 public:

  Array2View<DataType>* views() { return m_views.data(); }

 public:

  ARCANE_MATERIALS_EXPORT void synchronize() override;
  ARCANE_MATERIALS_EXPORT void synchronize(MeshMaterialVariableSynchronizerList& sync_list) override;
  ARCANE_MATERIALS_EXPORT void dumpValues(std::ostream& ostr) override;
  ARCANE_MATERIALS_EXPORT void dumpValues(std::ostream& ostr,AllEnvCellVectorView view) override;
  ARCANE_MATERIALS_EXPORT void serialize(ISerializer* sbuffer,Int32ConstArrayView ids) override;
  ARCANE_MATERIALS_EXPORT Int32 dataTypeSize() const override;

  ARCANE_MATERIALS_EXPORT
  void copyToBuffer(ConstArrayView<MatVarIndex> matvar_indexes,
                    ByteArrayView bytes) const override;

  ARCANE_MATERIALS_EXPORT
  void copyFromBuffer(ConstArrayView<MatVarIndex> matvar_indexes,
                      ByteConstArrayView bytes) override;

 public:

  ARCANE_MATERIALS_EXPORT void resize(Integer dim2_size);

 public:

  ConstArrayView<DataType> operator[](MatVarIndex mvi) const
  {
    return m_views[mvi.arrayIndex()][mvi.valueIndex()];
  }

  using BaseClass::setValue;
  using BaseClass::value;

 protected:

  using BaseClass::m_p;

 private:

  using BaseClass::m_global_variable;
  using BaseClass::m_global_variable_ref;
  using BaseClass::m_vars;
  using BaseClass::m_views;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Interface d'accès pour CellMaterialVariableArrayRef.
 */
template<typename ItemType,typename DataType>
class IMeshMaterialVariableArray
{
 public:

  using VariableRefType = MeshVariableArrayRefT<ItemType,DataType>;

 public:

  virtual ~IMeshMaterialVariableArray() = default;

 public:

  virtual Array2View<DataType>* valuesView() =0;
  virtual VariableRefType* globalVariableReference() const =0;
  virtual void incrementReference() =0;
  virtual IMeshMaterialVariable* toMeshMaterialVariable() =0;

  virtual void resize(Int32 dim2_size) =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \ingroup ArcaneMaterials
 * \brief Variable tableau sur un matériau du maillage.
 */
template<typename ItemType,typename DataType>
class MeshMaterialVariableArray
: public ItemMaterialVariableArray<DataType>
, public IMeshMaterialVariableArray<ItemType,DataType>
{
 public:

  using ThatClass = MeshMaterialVariableArray<ItemType,DataType>;
  using ThatInterface = IMeshMaterialVariableArray<ItemType,DataType>;
  typedef ItemType ItemTypeTemplate;

  using BaseClass = ItemMaterialVariableArray<DataType>;
  typedef MeshVariableArrayRefT<ItemType,DataType> VariableRefType;
  using PrivatePartType = typename BaseClass::PrivatePartType;

  typedef VariableReferenceGetter<ThatClass> ReferenceGetter;
  friend ReferenceGetter;

 protected:

  ARCANE_MATERIALS_EXPORT
  MeshMaterialVariableArray(const MaterialVariableBuildInfo& v,
                            PrivatePartType* global_var,
                            VariableRefType* global_var_ref,MatVarSpace mvs);
  ARCANE_MATERIALS_EXPORT
  ~MeshMaterialVariableArray(){}

 public:

  void incrementReference() final { BaseClass::incrementReference(); }
  Array2View<DataType>* valuesView() final { return BaseClass::views(); }
  void resize(Int32 dim2_size) final { BaseClass::resize(dim2_size); }
  VariableRefType* globalVariableReference() const final { return m_true_global_variable_ref; }
  IMeshMaterialVariable* toMeshMaterialVariable() final { return this; }

 private:

  VariableRefType* m_true_global_variable_ref;

 public:

  static ARCANE_MATERIALS_EXPORT MaterialVariableTypeInfo _buildVarTypeInfo(MatVarSpace space);

 private:

  static ARCANE_MATERIALS_EXPORT IMeshMaterialVariable* _autoCreate1(const MaterialVariableBuildInfo& vb);
  static ARCANE_MATERIALS_EXPORT IMeshMaterialVariable* _autoCreate2(const MaterialVariableBuildInfo& vb);
  static ARCANE_MATERIALS_EXPORT MeshMaterialVariableFactoryRegisterer m_auto_registerer1;
  static ARCANE_MATERIALS_EXPORT MeshMaterialVariableFactoryRegisterer m_auto_registerer2;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename VariableRefType> inline typename VariableRefType::PrivatePartType*
getVariableReference(VariableRefType*,const MaterialVariableBuildInfo& v,MatVarSpace mvs)
{
  using TruePrivatePartType = typename VariableRefType::TruePrivatePartType;
  using PrivatePartType = typename VariableRefType::PrivatePartType;

  MaterialVariableTypeInfo x = TruePrivatePartType::_buildVarTypeInfo(mvs);

  MeshHandle mesh_handle = v.meshHandle();
  if (mesh_handle.isNull())
    ARCANE_FATAL("No mesh handle for material variable");

  IMeshMaterialMng* mat_mng = v.materialMng();

  // TODO: regarder si verrou necessaire
  if (!mat_mng)
    mat_mng = IMeshMaterialMng::getReference(mesh_handle,true);

  IMeshMaterialVariableFactoryMng* vm = mat_mng->variableFactoryMng();
  IMeshMaterialVariable* var = vm->createVariable(x.fullName(),v);

  //IMeshMaterialVariable* var = ReferenceGetter::getReference(v,mvs);
  auto* true_var = dynamic_cast<PrivatePartType*>(var);
  ARCANE_CHECK_POINTER(true_var);
  return true_var;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

