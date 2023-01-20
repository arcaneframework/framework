// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MaterialVariableViews.h                                     (C) 2000-2022 */
/*                                                                           */
/* Gestion des vues sur les variables matériaux pour les accélérateurs.      */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ACCELERATOR_MATERIALVARIABLEVIEWS_H
#define ARCANE_ACCELERATOR_MATERIALVARIABLEVIEWS_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/ItemTypes.h"
#include "arcane/ItemLocalId.h"

#include "arcane/materials/IMeshMaterial.h"
#include "arcane/materials/MeshMaterialVariableRef.h"
#include "arcane/materials/EnvCellVector.h"
#include "arcane/materials/MatConcurrency.h"

#include "arcane/accelerator/AcceleratorGlobal.h"
#include "arcane/accelerator/ViewsCommon.h"
#include "arcane/accelerator/VariableViews.h"


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
using namespace Arcane;
using namespace Arcane::Materials;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Accelerator
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe de base des vues sur les variables matériaux.
 */
class MatVariableViewBase
{
 public:
  // Pour l'instant n'utilise pas encore \a command et \a var
  // mais il ne faut pas les supprimer
  // TODO: j'ai bestialement repris le fonctionnement des VariableViews de GG
  MatVariableViewBase(RunCommand& command,IMeshMaterialVariable* var)
  {
  }
 private:
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
  using ItemIndexType = MatVarIndex;
 
 public:

  MatItemVariableScalarInViewT(RunCommand& cmd, IMeshMaterialVariable* var, ArrayView<DataType>* v)
  : MatVariableViewBase(cmd, var), m_value(v), m_value0(v[0].unguardedBasePointer()){}

  //! Opérateur d'accès pour l'entité \a item
  ARCCORE_HOST_DEVICE const DataType& operator[](ItemIndexType mvi) const
  {
    return this->m_value[mvi.arrayIndex()][mvi.valueIndex()];
  }

  //! Opérateur d'accès pour l'entité \a item
  ARCCORE_HOST_DEVICE const DataType& operator[](ComponentItemLocalId lid) const
  {
    return this->m_value[lid.localId().arrayIndex()][lid.localId().valueIndex()];
  }

  //! Opérateur d'accès pour l'entité \a item
  ARCCORE_HOST_DEVICE const DataType& operator[](PureMatVarIndex pmvi) const
  {
    return this->m_value0[pmvi.valueIndex()];
  }

  //! Opérateur d'accès pour l'entité \a item
  ARCCORE_HOST_DEVICE const DataType& value(ItemIndexType mvi) const
  {
    return this->m_value[mvi.arrayIndex()][mvi.valueIndex()];
  }

  ARCCORE_HOST_DEVICE const DataType& value0(PureMatVarIndex idx) const
  {
    return this->m_value0[idx.valueIndex()];
  }

 private:
  ArrayView<DataType>* m_value;
  DataType* m_value0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vue en lecture sur une variable scalaire du maillage.
 */
template<typename ItemType,typename Accessor>
class MatItemVariableScalarOutViewT
: public MatVariableViewBase
{
 private:

  using DataType = typename Accessor::ValueType;
  using DataTypeReturnType = DataType&;
  using ItemIndexType = MatVarIndex;


// TODO: faut il rajouter des ARCANE_CHECK_AT(mvi.arrayIndex(), m_value.size()); ? il manquera tjrs le check sur l'autre dimension

 public:

  MatItemVariableScalarOutViewT(RunCommand& cmd,IMeshMaterialVariable* var,ArrayView<DataType>* v)
  : MatVariableViewBase(cmd, var), m_value(v), m_value0(v[0].unguardedBasePointer()){}

  //! Opérateur d'accès pour l'entité \a item
  ARCCORE_HOST_DEVICE Accessor operator[](ItemIndexType mvi) const
  {
    return Accessor(this->m_value[mvi.arrayIndex()].data()+mvi.valueIndex());
  }

  //! Opérateur d'accès pour l'entité \a item
  ARCCORE_HOST_DEVICE Accessor operator[](ComponentItemLocalId lid) const
  {
    return Accessor(this->m_value[lid.localId().arrayIndex()][lid.localId().valueIndex()]);
  }

  ARCCORE_HOST_DEVICE Accessor operator[](PureMatVarIndex pmvi) const
  {
    return Accessor(this->m_value0[pmvi.valueIndex()]);
  }

  //! Opérateur d'accès pour l'entité \a item
  ARCCORE_HOST_DEVICE Accessor value(ItemIndexType mvi) const
  {
    return Accessor(this->m_value[mvi.arrayIndex()][mvi.valueIndex()]);
  }

  // TODO: A été rajouté dans l'API pour faire comme dans les VariableViews... A garder ?
  //! Positionne la valeur pour l'entité \a item à \a v
  ARCCORE_HOST_DEVICE void setValue(ItemIndexType mvi,const DataType& v) const
  {
    this->m_value[mvi.arrayIndex()][mvi.valueIndex()] = v;
  }


  ARCCORE_HOST_DEVICE Accessor value0(PureMatVarIndex idx) const
  {
    return Accessor(this->m_value0[idx.valueIndex()]);
  }


// FIXME: Si on veut garder les 2 ci-dessous, il faudra
// redéfinir CellComponentCellEnumerator et EnvCellEnumerator en version accelerator
/*

  //! Valeur partielle de la variable pour l'itérateur \a mc
  ARCCORE_HOST_DEVICE Accessor operator[](CellComponentCellEnumerator mc) const
  {
    return Accessor(this->operator[](mc._varIndex()));
  }

  //! Valeur partielle de la variable pour l'itérateur \a mc
  ARCCORE_HOST_DEVICE Accessor operator[](EnvCellEnumerator mc) const
  {
    return Accessor(this->operator[](mc._varIndex()));
  }
*/

 private:
  ArrayView<DataType>* m_value;
  DataType* m_value0;
};


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vue en écriture pour les variables materiaux scalaire
 */
template<typename DataType> auto
viewOut(RunCommand& cmd, Materials::CellMaterialVariableScalarRef<DataType>& var)
{
  using Accessor = DataViewSetter<DataType>;
  return MatItemVariableScalarOutViewT<Cell,Accessor>(cmd, var.materialVariable(),var._internalValue());
}

/*!
 * \brief Vue en écriture pour les variables materiaux scalaire.
 * Spécialisation pour le Real2 pour éviter les mauvais usages
 * 
 * TODO: A faire plus tard ?
 * 
 *
template<> auto
viewOut(RunCommand& cmd, Materials::CellMaterialVariableScalarRef<Real2>&)
{
}
*/

/*!
 * \brief Vue en écriture pour les variables materiaux scalaire.
 * Spécialisation pour le Real3 pour éviter les mauvais usages
 * 
 * TODO: A faire plus tard ?
 * 
 *
template<> auto
viewOut(RunCommand& cmd, Materials::CellMaterialVariableScalarRef<Real3>& var)
{
}
*/

/*!
 * \brief Vue en écriture pour les variables materiaux tableau
 *
 * TODO: Array a faire plus tard ?
 *
template<typename DataType> auto
viewOut(RunCommand& cmd, Materials::CellMaterialVariableArrayRef<DataType>& var)
{
}
*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vue en lecture/écriture pour les variables materiaux scalaire
 */
template<typename DataType> auto
viewInOut(RunCommand& cmd, Materials::CellMaterialVariableScalarRef<DataType>& var)
{
  using Accessor = DataViewGetterSetter<DataType>;
  return MatItemVariableScalarOutViewT<Cell,Accessor>(cmd, var.materialVariable(),var._internalValue());
}

/*!
 * \brief Vue en lecture/écriture pour les variables materiaux scalaire.
 * Spécialisation pour le Real2 pour éviter les mauvais usages
 * 
 * TODO: A faire plus tard ?
 * 
 *
template<> auto
viewInOut(RunCommand& cmd, Materials::CellMaterialVariableScalarRef<Real2>& var)
{
}
*/

/*!
 * \brief Vue en lecture/écriture pour les variables materiaux scalaire.
 * Spécialisation pour le Real3 pour éviter les mauvais usages
 * 
 * TODO: A faire plus tard ?
 * 
 *
template<> auto
viewInOut(RunCommand& cmd, Materials::CellMaterialVariableScalarRef<Real3>& var)
{
}
*/

/*!
 * \brief Vue en lecture/écriture pour les variables materiaux tableau
 *
 *
 * TODO: Array => a faire plus tard ?
 * 
template<typename DataType> auto
viewInOut(RunCommand& cmd, Materials::CellMaterialVariableArrayRef<DataType>& var)
{
}
*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vue en lecture pour les variables materiaux scalaire
 */
template<typename DataType> auto
viewIn(RunCommand& cmd,const Materials::CellMaterialVariableScalarRef<DataType>& var)
{
  return MatItemVariableScalarInViewT<Cell,DataType>(cmd, var.materialVariable(),var._internalValue());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Si on garde cette classe il faudra la blinder, surtout sur le ctor component
class EnvCellAccessor {
 public:
  ARCCORE_HOST_DEVICE explicit EnvCellAccessor(EnvCell ec)
  : m_mvi(ec._varIndex()), m_cid(ec.globalCell().itemLocalId()) {}
  
  explicit EnvCellAccessor(ComponentItemInternal* cii)
  : m_mvi(cii->variableIndex()), m_cid(cii->globalItem()->localId()) {}

  ARCCORE_HOST_DEVICE explicit EnvCellAccessor(MatVarIndex mvi, CellLocalId cid)
  : m_mvi(MatVarIndex(mvi.arrayIndex(),mvi.valueIndex())), m_cid(cid) {}

  ARCCORE_HOST_DEVICE auto operator()() {
    return std::make_tuple(m_mvi, m_cid);
  }
  
  ARCCORE_HOST_DEVICE MatVarIndex varIndex() {return m_mvi;};
  
  ARCCORE_HOST_DEVICE CellLocalId globalCellId() {return m_cid;}
 
 private:
  MatVarIndex m_mvi;
  CellLocalId m_cid;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace impl
{

using ComponentItemInternalPtr = ComponentItemInternal*; 

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#if defined(ARCANE_COMPILING_CUDA) || defined(ARCANE_COMPILING_HIP)

template<typename ItemType,typename Lambda> __global__
void doIndirectGPULambda(SmallSpan<const MatVarIndex> mvis,SmallSpan<const Int32> cids,Lambda func)
{
  auto privatizer = privatize(func);
  auto& body = privatizer.privateCopy();

  Int32 i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i<mvis.size()){
    EnvCellAccessor lec(mvis[i], static_cast<CellLocalId>(cids[i]));

    //if (i<10)
    //printf("CUDA %d lid=%d\n",i,lid.localId());
    body(lec);
  }
}

#endif // ARCANE_COMPILING_CUDA

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename ItemType,typename Lambda>
void _doIndirectThreadLambda(const EnvCellVectorView& sub_items,Lambda func)
{
  auto privatizer = privatize(func);
  auto& body = privatizer.privateCopy();

// TODO: A valider avec GG si l'utilisation d'un for range est acceptable
  for (auto i : sub_items.itemsInternalView())
    body(EnvCellAccessor(i));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Applique l'enumération \a func sur la liste d'entité \a items.
 */
template<typename Lambda> void
_applyItems(RunCommand& command,const EnvCellVectorView& items,Lambda func)
{
  // TODO: fusionner la partie commune avec 'applyLoop'
  Int32 vsize = static_cast<Int32>(items.nbItem());
  if (vsize==0)
    return;
  impl::RunCommandLaunchInfo launch_info(command);
  const eExecutionPolicy exec_policy = launch_info.executionPolicy();
  switch(exec_policy){
  case eExecutionPolicy::CUDA:
#if defined(ARCANE_COMPILING_CUDA)
    {
      launch_info.beginExecute();
      SmallSpan<const MatVarIndex> mvis(items.component()->variableIndexer()->matvarIndexes());
      SmallSpan<const Int32> cids(items.component()->variableIndexer()->localIds());
      // TODO: vérifier que l'arcane assert n'est pas tout le temps fait
      ARCANE_ASSERT(mvis.size() == cids.size(), "MatVarIndex and CellLocalId arrays have different size");

      auto [b,t] = launch_info.computeThreadBlockInfo(vsize);
      cudaStream_t* s = reinterpret_cast<cudaStream_t*>(launch_info._internalStreamImpl());
      // TODO: le memadvise est mal placé ici, mais n'apporte pas plus dans l'indexer
      /*
      auto err1 = cudaMemAdvise(mvis.data(), mvis.sizeBytes(), cudaMemoryAdvise::cudaMemAdviseSetReadMostly, 0);
      auto err2 = cudaMemAdvise(cids.data(), cids.sizeBytes(), cudaMemoryAdvise::cudaMemAdviseSetReadMostly, 0);
      if ((err1!=0) || (err2!=0)) {
        ARCANE_FATAL("ERROR CUDA MemAdvise FAILED");
      }
      */
      // TODO: le prefetch fait chuter les perfs ... 
      /*
      auto err1 = cudaMemPrefetchAsync (mvis.data(), mvis.sizeBytes(), 0, *s);
      auto err2 = cudaMemPrefetchAsync (cids.data(), cids.sizeBytes(), 0, *s);
      if ((err1!=0) || (err2!=0)) {
        ARCANE_FATAL("ERROR de prefetch CUDA");
      }
      */

      // TODO: utiliser cudaLaunchKernel() à la place.
      impl::doIndirectGPULambda<EnvCell,Lambda> <<<b,t,0,*s>>>(mvis,cids,std::forward<Lambda>(func));
    }
#else
    ARCANE_FATAL("Requesting CUDA kernel execution but the kernel is not compiled with CUDA compiler");
#endif
    break;
/*
// TODO: NYI
  case eExecutionPolicy::HIP:
#if defined(ARCANE_COMPILING_HIP)
    {
      launch_info.beginExecute();
      SmallSpan<const Int32> local_ids = items.localIds();
      auto [b,t] = launch_info.computeThreadBlockInfo(vsize);
      hipStream_t* s = reinterpret_cast<hipStream_t*>(launch_info._internalStreamImpl());
      auto& loop_func = impl::doIndirectGPULambda<ItemType,Lambda>;
      hipLaunchKernelGGL(loop_func,b,t,0,*s, local_ids,std::forward<Lambda>(func));
    }
#else
    ARCANE_FATAL("Requesting HIP kernel execution but the kernel is not compiled with HIP compiler");
#endif
    break;
*/
  case eExecutionPolicy::Sequential:
    {
      launch_info.beginExecute();
      // TODO: A voir avec GG si un for range est acceptable
      for (auto i : items.itemsInternalView())
        func(EnvCellAccessor(i));

      // TODO: Faut il remplacer le code ci-dessus par celui ci-dessous pour avoir un comportement équivalent entre CPU et GPU ?
      /*
      SmallSpan<const MatVarIndex> mvis(items.component()->variableIndexer()->matvarIndexes());
      SmallSpan<const Int32> cids(items.component()->variableIndexer()->localIds());
      assert(mvis.size() == cids.size());
      for (int i(0); i<mvis.size(); ++i)
        func(EnvCellAccessor(mvis[i], (CellLocalId)cids[i]));
      */
    }
    break;
  case eExecutionPolicy::Thread:
    {
      launch_info.beginExecute();
      arcaneParallelForeach(items,
                            [&](EnvCellVectorView sub_items)
                            {
                              impl::_doIndirectThreadLambda<EnvCell,Lambda>(sub_items,func);
                            });
    }
    break;
  default:
    ARCANE_FATAL("Invalid execution policy '{0}'",exec_policy);
  }
  launch_info.endExecute();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace impl

template<typename Lambda> void
run(RunCommand& command,const EnvCellVectorView& items,Lambda func)
{
  impl::_applyItems(command,items,std::forward<Lambda>(func));
}

template<>
class ItemRunCommand<EnvCell>
{
 public:
  ItemRunCommand(RunCommand& command,const EnvCellVectorView& items)
  : m_command(command), m_items(items)
  {
  }
  RunCommand& m_command;
  EnvCellVectorView m_items;
};

ItemRunCommand<EnvCell>
operator<<(RunCommand& command,const EnvCellVectorView& items)
{
  return ItemRunCommand<EnvCell>(command,items);
}

// Surcharge pour avoir directement l'environnement en paramètre
ItemRunCommand<EnvCell>
operator<<(RunCommand& command,IMeshEnvironment* env)
{
  return ItemRunCommand<EnvCell>(command,env->envView());
}

template<typename Lambda>
void operator<<(ItemRunCommand<EnvCell>&& nr,Lambda f)
{
  run(nr.m_command,nr.m_items,std::forward<Lambda>(f));
}

} // End namespace Arcane::Accelerator

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// FIXME: Hack assez moche pour tester au plus vite, il faudra voir comment faire ça propre
namespace Arcane {
  namespace Materials {
    using EnvCellLocalId = Arcane::Accelerator::EnvCellAccessor;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
