// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
#ifndef MESHUTILS_H_
#define MESHUTILS_H_

#include "Utils/Utils.h"
#include <arcane/VariableTypes.h>
#include <arcane/ISubDomain.h>
#include <arcane/IVariable.h>
#include <arcane/IVariableAccessor.h>
#include <arcane/ItemGroup.h>
#include <arcane/IMesh.h>
#include <arcane/IItemFamily.h>

#include "Utils/MeshVarExpr.h"

BEGIN_ARCGEOSIM_NAMESPACE

BEGIN_MESH_NAMESPACE

//! define empty support
struct NullItem {} ;

template<typename Item>
Item dualToItem(ItemEnumeratorT<DualNode>& inode) ;

template<typename DataType,
         typename ItemType>
void assign(MeshVariableScalarRefT<ItemType,DataType>& var,
            const Real& value,
            const ItemGroupT<ItemType>& group)
{
  for( ItemEnumeratorT<ItemType> item((group).enumerator()); 
       item.hasNext(); ++item )
  {
    var[item] = value ;
  }
} ;

template<typename DataType,
         typename ItemType, 
         typename Expr>
void assign(MeshVariableScalarRefT<ItemType,DataType>& var,
            const Expr& expr,
            const ItemGroupT<ItemType>& group)
{
  for( ItemEnumeratorT<ItemType> item((group).enumerator()); 
       item.hasNext(); ++item )
  {
    var[item] = expr[item] ;
  }
} ;


template<typename DataType,
         typename ItemType, 
         typename Expr>
void assignFromDualExpr(MeshVariableScalarRefT<ItemType,DataType>& var,
                        const Expr& expr,
                        const ItemGroupT<DualNode>& group)
{
  ENUMERATE_DUALNODE(inode,group)
  {
    const ItemType& item = dualToItem<ItemType>(inode) ;
    var[item] = expr[inode] ;
  }
} ;

template<typename DataType,
         typename ItemType, 
         typename Expr>
void assignForDualGroup(MeshVariableScalarRefT<ItemType,DataType>& var,
                        const Expr& expr,
                        const ItemGroupT<DualNode>& group)
{
  ENUMERATE_DUALNODE(inode,group)
  {
    const ItemType& item = dualToItem<ItemType>(inode) ;
    var[item] = expr[item] ;
  }
} ;

END_MESH_NAMESPACE

END_ARCGEOSIM_NAMESPACE
#endif /*MESHUTILS_H_*/
