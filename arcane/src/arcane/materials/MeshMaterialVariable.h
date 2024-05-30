// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshMaterialVariable.h                                      (C) 2000-2024 */
/*                                                                           */
/* Variable sur un matériau du maillage.                                     */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MATERIALS_MESHMATERIALVARIABLE_H
#define ARCANE_MATERIALS_MESHMATERIALVARIABLE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/NotImplementedException.h"
#include "arcane/utils/Array.h"
#include "arcane/utils/MemoryView.h"

#include "arcane/core/materials/IMeshMaterialVariable.h"
#include "arcane/core/materials/MatVarIndex.h"

#include "arcane/materials/MaterialsGlobal.h"

#include "arcane/materials/MeshMaterialVariableFactoryRegisterer.h"

#include "arcane/core/materials/IScalarMeshMaterialVariable.h"
#include "arcane/core/materials/IArrayMeshMaterialVariable.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class VariableInfo;
class VariableRef;
class VariableBuildInfo;
template <typename ItemType, typename DataTypeT> class MeshVariableScalarRefT;
} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class MaterialVariableBuildInfo;
class MeshMaterialVariablePrivate;
class MeshMaterialVariableSynchronizerList;
class CopyBetweenPartialAndGlobalArgs;

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
  friend MeshMaterialVariablePrivate;
  // Pour accès à 'copyToBuffer', 'copyFromBuffer'. A supprimer ensuite
  friend MeshMaterialVariableSynchronizerList;

 public:

  MeshMaterialVariable(const MaterialVariableBuildInfo& v,MatVarSpace mvs);
  ~MeshMaterialVariable() override;

 public:

  String name() const override;
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

  IMeshMaterialVariableInternal* _internalApi() override;

 public:

  //! @name Fonctions publiques mais réservées à Arcane pour gérer les synchronisations
  //@{
  virtual Int32 dataTypeSize() const =0;

 protected:

  // TODO: interface obsolète à supprimer
  virtual void copyToBuffer(ConstArrayView<MatVarIndex> matvar_indexes,ByteArrayView bytes) const =0;
  // TODO: interface obsolète à supprimer
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

  MeshMaterialVariablePrivate* m_p = nullptr;
  UniqueArray<Span<std::byte>> m_views_as_bytes;

 protected:

  void _copyToBuffer(SmallSpan<const MatVarIndex> matvar_indexes, Span<std::byte> bytes,RunQueue* queue) const;
  void _copyFromBuffer(SmallSpan<const MatVarIndex> matvar_indexes, Span<const std::byte> bytes,RunQueue* queue);

  virtual Ref<IData> _internalCreateSaveDataRef(Integer nb_value) =0;
  virtual void _saveData(IMeshComponent* component,IData* data) =0;
  virtual void _restoreData(IMeshComponent* component,IData* data,Integer data_index,
                            Int32ConstArrayView ids,bool allow_null_id) =0;
  virtual void _copyGlobalToPartial(const CopyBetweenPartialAndGlobalArgs& args) = 0;
  virtual void _copyPartialToGlobal(const CopyBetweenPartialAndGlobalArgs& args) = 0;
  virtual void _initializeNewItems(const ComponentItemListBuilder& list_builder, RunQueue& queue) = 0;
  virtual void _syncReferences(bool update_views) = 0;
  virtual void _resizeForIndexer(Int32 index, RunQueue& queue) = 0;

 private:

  static SmallSpan<const Int32> _toInt32Indexes(SmallSpan<const MatVarIndex> indexes);
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
  copyTo(SmallSpan<const DataType> input, SmallSpan<const Int32> input_indexes,
         SmallSpan<DataType> output, SmallSpan<const Int32> output_indexes,
         const RunQueue& queue);
  ARCANE_MATERIALS_EXPORT static void
  resizeAndFillWithDefault(ValueDataType* data,ContainerType& container,Integer dim1_size);

  static ARCCORE_HOST_DEVICE void setValue(DataType& view,const DataType& v)
  {
    view = v;
  }
  ARCANE_MATERIALS_EXPORT static void
  resizeWithReserve(PrivatePartType* var,Integer new_size);
  static Integer dimension() { return 0; }
  static Span<std::byte> toBytes(ArrayView<DataType> view)
  {
    Span<DataType> s(view);
    return asWritableBytes(s);
  }

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
  typedef SmallSpan<const DataType> SubInputViewType;
  typedef Array2View<DataType> ContainerViewType;
  typedef ConstArray2View<DataType> ContainerConstViewType;
  typedef Array2VariableT<DataType> PrivatePartType;
  typedef IArray2DataT<DataType> ValueDataType;
  typedef Array2<DataType> ContainerType;
  typedef UniqueArray2<DataType> UniqueContainerType;
  typedef VariableRefArray2T<DataType> VariableRefType;

 public:

  ARCANE_MATERIALS_EXPORT
  static void saveData(IMeshComponent* component, IData* data,
                       Array<ContainerViewType>& cviews);
  ARCANE_MATERIALS_EXPORT
  static void copyTo(SmallSpan2<const DataType> input, SmallSpan<const Int32> input_indexes,
                     SmallSpan2<DataType> output, SmallSpan<const Int32> output_indexes,
                     const RunQueue& queue);
  ARCANE_MATERIALS_EXPORT
  static void resizeAndFillWithDefault(ValueDataType* data, ContainerType& container,
                                       Integer dim1_size);
  static ARCCORE_HOST_DEVICE void setValue(SmallSpan<DataType> view, const DataType& v)
  {
    view.fill(v);
  }
  static ARCCORE_HOST_DEVICE void setValue(SmallSpan<DataType> view, SmallSpan<const DataType> v)
  {
    view.copy(v);
  }
  ARCANE_MATERIALS_EXPORT
  static void resizeWithReserve(PrivatePartType* var, Integer new_size);
  static Span<std::byte> toBytes(Array2View<DataType> view)
  {
    Span<DataType> s(view.data(),view.totalNbElement());
    return asWritableBytes(s);
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
  void _copyGlobalToPartial(const CopyBetweenPartialAndGlobalArgs& args) override;
  ARCANE_MATERIALS_EXPORT
  void _copyPartialToGlobal(const CopyBetweenPartialAndGlobalArgs& args) override;
  ARCANE_MATERIALS_EXPORT
  void _initializeNewItems(const ComponentItemListBuilder& list_builder, RunQueue& queue) override;

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
  ARCANE_MATERIALS_EXPORT void _syncReferences(bool check_resize) override;
  ARCANE_MATERIALS_EXPORT void _resizeForIndexer(Int32 index, RunQueue& queue) override;
  ARCANE_MATERIALS_EXPORT void _copyHostViewsToViews(RunQueue* queue);

 public:


 protected:

  PrivatePartType* m_global_variable = nullptr;
  VariableRef* m_global_variable_ref = nullptr;
  //! Variables pour les différents matériaux.
  UniqueArray<PrivatePartType*> m_vars;
  UniqueArray<ContainerViewType> m_views;
  //! Liste des vues visibles uniquement depuis l'ĥote
  UniqueArray<ContainerViewType> m_host_views;

 protected:

  void _setView(Int32 index)
  {
    ContainerViewType view;
    if (m_vars[index])
      view = m_vars[index]->valueView();
    m_host_views[index] = view;
    m_views_as_bytes[index] = TraitsType::toBytes(view);
  }

 private:

  bool _isValidAndUsedAndGlobalUsed(PrivatePartType* partial_var);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename TrueType>
class MeshMaterialVariableCommonStaticImpl
{
 public:
  static ARCANE_MATERIALS_EXPORT IMeshMaterialVariable*
  getReference(const MaterialVariableBuildInfo& v,MatVarSpace mvs);
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

 protected:

  ArrayView<ArrayView<DataType>> _containerView() { return this->m_views; }

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

 private:

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

 private:

  void _copyToBufferLegacy(SmallSpan<const MatVarIndex> matvar_indexes,
                           Span<std::byte> bytes) const;
  void _copyFromBufferLegacy(SmallSpan<const MatVarIndex> matvar_indexes,
                       Span<const std::byte> bytes);
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
, public IScalarMeshMaterialVariable<ItemType,DataType>
{
 public:
	
  using ThatClass = MeshMaterialVariableScalar<ItemType,DataType>;
  using ThatInterface = IScalarMeshMaterialVariable<ItemType,DataType>;
  using BuilderType = typename ThatInterface::BuilderType;
  using StaticImpl = MeshMaterialVariableCommonStaticImpl<ThatClass>;
  using ItemTypeTemplate = ItemType;

  using BaseClass = ItemMaterialVariableScalar<DataType>;
  using VariableRefType = MeshVariableScalarRefT<ItemType,DataType>;
  using PrivatePartType = typename BaseClass::PrivatePartType;

  friend StaticImpl;

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
  ArrayView<ArrayView<DataType>> _internalFullValuesView() final { return BaseClass::_containerView(); }
  void fillFromArray(IMeshMaterial* mat, ConstArrayView<DataType> values) final
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

 private:
  
  VariableRefType* m_true_global_variable_ref;
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

  ARCANE_DEPRECATED_REASON("Y2022: Do not use internal storage accessor")
  Array2View<DataType>* views() { return m_views.data(); }

 public:

  ARCANE_MATERIALS_EXPORT void synchronize() override;
  ARCANE_MATERIALS_EXPORT void synchronize(MeshMaterialVariableSynchronizerList& sync_list) override;
  ARCANE_MATERIALS_EXPORT void dumpValues(std::ostream& ostr) override;
  ARCANE_MATERIALS_EXPORT void dumpValues(std::ostream& ostr,AllEnvCellVectorView view) override;
  ARCANE_MATERIALS_EXPORT void serialize(ISerializer* sbuffer,Int32ConstArrayView ids) override;

 private:

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
  ArrayView<Array2View<DataType>> _containerView() { return m_views; }

 private:

  using BaseClass::m_global_variable;
  using BaseClass::m_global_variable_ref;
  using BaseClass::m_vars;
  using BaseClass::m_views;

  void _copyToBufferLegacy(SmallSpan<const MatVarIndex> matvar_indexes,
                           Span<std::byte> bytes) const;
  void _copyFromBufferLegacy(SmallSpan<const MatVarIndex> matvar_indexes,
                             Span<const std::byte> bytes);
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
, public IArrayMeshMaterialVariable<ItemType,DataType>
{
 public:

  using ThatClass = MeshMaterialVariableArray<ItemType,DataType>;
  using ThatInterface = IArrayMeshMaterialVariable<ItemType,DataType>;
  using BuilderType = typename ThatInterface::BuilderType;
  using StaticImpl = MeshMaterialVariableCommonStaticImpl<ThatClass>;
  typedef ItemType ItemTypeTemplate;

  using BaseClass = ItemMaterialVariableArray<DataType>;
  typedef MeshVariableArrayRefT<ItemType,DataType> VariableRefType;
  using PrivatePartType = typename BaseClass::PrivatePartType;

  friend StaticImpl;

 protected:

  ARCANE_MATERIALS_EXPORT
  MeshMaterialVariableArray(const MaterialVariableBuildInfo& v,
                            PrivatePartType* global_var,
                            VariableRefType* global_var_ref,MatVarSpace mvs);
  ARCANE_MATERIALS_EXPORT
  ~MeshMaterialVariableArray(){}

 public:

  void incrementReference() final { BaseClass::incrementReference(); }
  ArrayView<Array2View<DataType>> _internalFullValuesView() final { return BaseClass::_containerView(); }
  void resize(Int32 dim2_size) final { BaseClass::resize(dim2_size); }
  VariableRefType* globalVariableReference() const final { return m_true_global_variable_ref; }
  IMeshMaterialVariable* toMeshMaterialVariable() final { return this; }

 private:

  VariableRefType* m_true_global_variable_ref;
};


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
