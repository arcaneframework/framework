// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ComponentSimd.h                                             (C) 2000-2017 */
/*                                                                           */
/* Support de la vectorisation pour les matériaux et milieux.                */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MATERIALS_COMPONENTSIMD_H
#define ARCANE_MATERIALS_COMPONENTSIMD_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \file ComponentSimd.h
 *
 * Ce fichier contient les différents types pour gérer la
 * vectorisation sur les composants (matériaux et milieux).
 */

#include "arcane/ArcaneTypes.h"
#include "arcane/SimdItem.h"

#include "arcane/materials/MatItem.h"
#include "arcane/materials/MatItemEnumerator.h"
#include "arcane/materials/ComponentPartItemVectorView.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifdef __INTEL_COMPILER
#  define A_ALIGNED_64 __attribute__((align_value(64)))
#else
#  define A_ALIGNED_64
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE
MATERIALS_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Indexeur SIMD sur un composant.
 */
class ARCANE_MATERIALS_EXPORT ARCANE_ALIGNAS(64) SimdMatVarIndex
{
 public:
  typedef SimdEnumeratorBase::SimdIndexType SimdIndexType;
 public:

  SimdMatVarIndex(Int32 array_index,SimdIndexType value_index)
  : m_value_index(value_index), m_array_index(array_index)
  {
  }
  SimdMatVarIndex(){}

 public:

  //! Retourne l'indice du tableau de valeur dans la liste des variables.
  Int32 arrayIndex() const { return m_array_index; }

  //! Retourne l'indice dans le tableau de valeur
  const SimdIndexType& valueIndex() const { return m_value_index; }

 private:

  SimdIndexType m_value_index;
  Int32 m_array_index;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Enumérateur SIMD sur une sous-partie (pure ou partielle) d'un
 * sous-ensemble des mailles d'un composant (matériau ou milieu)
 */
class ARCANE_MATERIALS_EXPORT ComponentPartSimdCellEnumerator
: public SimdEnumeratorBase
{
 protected:
  ComponentPartSimdCellEnumerator(IMeshComponent* component,Int32 component_part_index,
                                  Int32ConstArrayView item_indexes)
  : SimdEnumeratorBase(item_indexes), m_component_part_index(component_part_index), m_component(component)
  {
  }
 public:
  static ComponentPartSimdCellEnumerator create(ComponentPartItemVectorView v)
  {
    return ComponentPartSimdCellEnumerator(v.component(),v.componentPartIndex(),v.valueIndexes());
  }
 public:

  SimdMatVarIndex _varIndex() const { return SimdMatVarIndex(m_component_part_index,*_currentSimdIndex()); }

  operator SimdMatVarIndex() const
  {
    return _varIndex();
  }

 protected:
  Integer m_component_part_index;
  IMeshComponent* m_component;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#define ENUMERATE_SIMD_COMPONENTCELL(iname,env) \
  A_ENUMERATE_COMPONENTCELL(ComponentPartSimdCellEnumerator,iname,env)

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename Lambda> void
simple_simd_env_loop(ComponentPartItemVectorView pure_items,
                     ComponentPartItemVectorView impure_items,
                     const Lambda& lambda)
{
  ENUMERATE_COMPONENTITEM(ComponentPartSimdCell,mvi,pure_items){
    lambda(mvi);
  }
  ENUMERATE_SIMD_COMPONENTCELL(mvi,impure_items){
    lambda(mvi);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_MATERIALS_EXPORT LoopFunctorEnvPartSimdCell
{
 public:
  typedef const SimdMatVarIndex& IterType;
 public:
  LoopFunctorEnvPartSimdCell(ComponentPartItemVectorView pure_items,
                         ComponentPartItemVectorView impure_items)
  : m_pure_items(pure_items), m_impure_items(impure_items){}
 public:
  static LoopFunctorEnvPartSimdCell create(const EnvCellVector& env);
  static LoopFunctorEnvPartSimdCell create(IMeshEnvironment* env);
 public:
  template<typename Lambda>
  void operator<<(Lambda&& lambda)
  {
    simple_simd_env_loop(m_pure_items,m_impure_items,lambda);
  }
 private:
  ComponentPartItemVectorView m_pure_items;
  ComponentPartItemVectorView m_impure_items;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Macro pour itérer sur les entités d'un composant via une
 * fonction lambda du C++11.
 *
 * Les arguments sont les mêmes que pour la macro ENUMERATE_COMPONENTITEM().
 *
 * Le code après la macro correspond au corps de la fonction lambda du C++11.
 * Il doit donc être compris entre deux accolades '{' '}' et se
 * terminer par un point-virgule ';'. Par exemple:
 *
 \code
 * ENUMERATE_COMPONENTITEM_LAMBDA(){
 * };
 \endcode
 *
 * \note Même si le code est similaire à celui d'une boucle, il s'agit d'une
 * fonction lambda du C++11 et donc il n'est pas possible d'utiliser des
 * mots clés comme 'break' ou 'continue'. Si on souhaite arrêtre une itération
 * il faut utiliser le mot clé 'return'.
 */
#define ENUMERATE_COMPONENTITEM_LAMBDA(iter_type,iter,container)\
  Arcane::Materials:: LoopFunctor ## iter_type :: create ( (container) ) << [=]( Arcane::Materials:: LoopFunctor ## iter_type :: IterType iter)

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe de base des vues sur les variables.
 */
class MatVariableViewBase
{
 public:
  MatVariableViewBase(IMeshMaterialVariable* var) : m_variable(var)
  {
  }
 public:
  IMeshMaterialVariable* variable() const { return m_variable; }
 private:
  IMeshMaterialVariable* m_variable;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vue en lecture sur une variable scalaire du maillage.
 */
template<typename ItemType,typename DataType>
class MatItemVariableScalarInViewT
: public MatVariableViewBase
{
 private:

  typedef MatVarIndex ItemIndexType;
  typedef A_ALIGNED_64 DataType* DataTypeAlignedPtr;

 public:

  MatItemVariableScalarInViewT(IMeshMaterialVariable* var,ArrayView<DataType>* v)
  : MatVariableViewBase(var), m_value(v), m_value0(v[0].unguardedBasePointer()){}

  //! Opérateur d'accès vectoriel avec indirection.
  typename SimdTypeTraits<DataType>::SimdType
  operator[](const SimdMatVarIndex& mvi) const
  {
    typedef typename SimdTypeTraits<DataType>::SimdType SimdType;
    return SimdType(m_value[mvi.arrayIndex()].data(),mvi.valueIndex());
  }

  //! Opérateur d'accès pour l'entité \a item
  DataType operator[](ItemIndexType mvi) const
  {
    return this->m_value[mvi.arrayIndex()][mvi.valueIndex()];
  }

  //! Opérateur d'accès pour l'entité \a item
  DataType operator[](ComponentItemLocalId lid) const
  {
    return this->m_value[lid.localId().arrayIndex()][lid.localId().valueIndex()];
  }

  //! Opérateur d'accès pour l'entité \a item
  DataType operator[](PureMatVarIndex pmvi) const
  {
    return this->m_value0[pmvi.valueIndex()];
  }

  //! Opérateur d'accès pour l'entité \a item
  DataType value(ItemIndexType mvi) const
  {
    return this->m_value[mvi.arrayIndex()][mvi.valueIndex()];
  }

  DataType value0(PureMatVarIndex idx) const
  {
    return this->m_value0[idx.valueIndex()];
  }

  //! Valeur partielle de la variable pour l'itérateur \a mc
  DataType operator[](CellComponentCellEnumerator mc) const
  {
    return this->operator[](mc._varIndex());
  }

  //! Valeur partielle de la variable pour l'itérateur \a mc
  DataType operator[](EnvCellEnumerator mc) const
  {
    return this->operator[](mc._varIndex());
  }

 private:
  ArrayView<DataType>* m_value;
  DataTypeAlignedPtr m_value0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vue en lecture sur une variable scalaire du maillage.
 */
template<typename ItemType,typename DataType>
class MatItemVariableScalarOutViewT
: public MatVariableViewBase
{
 private:

  typedef MatVarIndex ItemIndexType;
  typedef A_ALIGNED_64 DataType* DataTypeAlignedPtr;

 public:

  MatItemVariableScalarOutViewT(IMeshMaterialVariable* var,ArrayView<DataType>* v)
  : MatVariableViewBase(var), m_value(v), m_value0(v[0].unguardedBasePointer()){}

  //! Opérateur d'accès vectoriel avec indirection.
  SimdSetter<DataType> operator[](const SimdMatVarIndex& mvi) const
  {
    return SimdSetter<DataType>(m_value[mvi.arrayIndex()].data(),mvi.valueIndex());
  }

  //! Opérateur d'accès pour l'entité \a item
  DataType& operator[](ItemIndexType mvi) const
  {
    return this->m_value[mvi.arrayIndex()][mvi.valueIndex()];
  }

  //! Opérateur d'accès pour l'entité \a item
  DataType& operator[](ComponentItemLocalId lid) const
  {
    return this->m_value[lid.localId().arrayIndex()][lid.localId().valueIndex()];
  }

  DataType& operator[](PureMatVarIndex pmvi) const
  {
    return this->m_value0[pmvi.valueIndex()];
  }

  //! Opérateur d'accès pour l'entité \a item
  DataType& value(ItemIndexType mvi) const
  {
    return this->m_value[mvi.arrayIndex()][mvi.valueIndex()];
  }

  DataType& value0(PureMatVarIndex idx) const
  {
    return this->m_value0[idx.valueIndex()];
  }

  //! Valeur partielle de la variable pour l'itérateur \a mc
  DataType& operator[](CellComponentCellEnumerator mc) const
  {
    return this->operator[](mc._varIndex());
  }

  //! Valeur partielle de la variable pour l'itérateur \a mc
  DataType& operator[](EnvCellEnumerator mc) const
  {
    return this->operator[](mc._varIndex());
  }

 private:
  ArrayView<DataType>* m_value;
  DataTypeAlignedPtr m_value0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vue en lecture.
 */
template<typename DataType>
MatItemVariableScalarInViewT<Cell,DataType>
viewIn(const CellMaterialVariableScalarRef<DataType>& var)
{
  return MatItemVariableScalarInViewT<Cell,DataType>(var.materialVariable(),var._internalValue());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vue en écriture
 */
template<typename DataType>
MatItemVariableScalarOutViewT<Cell,DataType>
viewOut(CellMaterialVariableScalarRef<DataType>& var)
{
  return MatItemVariableScalarOutViewT<Cell,DataType>(var.materialVariable(),var._internalValue());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MATERIALS_END_NAMESPACE
ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
