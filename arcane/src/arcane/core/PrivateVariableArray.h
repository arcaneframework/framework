// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* PrivateVariableArray.h                                      (C) 2000-2023 */
/*                                                                           */
/* Classe gérant une variable array sur une entité du maillage.              */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_PRIVATEVARIABLEARRAY_H
#define ARCANE_CORE_PRIVATEVARIABLEARRAY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/MeshVariableRef.h"

#include "arcane/core/Array2Variable.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup Variable
 * \brief Classe de factorisation des variables scalaires sur des entités du maillage.
 */
template<typename DataType>
class PrivateVariableArrayT
: public MeshVariableRef
{
 protected:
  
  typedef DataType& DataTypeReturnReference;
  typedef Array2VariableT<DataType> PrivatePartType;
  
 protected:
  
  ARCANE_CORE_EXPORT PrivateVariableArrayT(const VariableBuildInfo& vb, const VariableInfo& vi);
  ARCANE_CORE_EXPORT PrivateVariableArrayT(const PrivateVariableArrayT& rhs);
  ARCANE_CORE_EXPORT PrivateVariableArrayT(IVariable* var);
  
  ARCANE_CORE_EXPORT void operator=(const PrivateVariableArrayT& rhs);

 public:
  
  Array2View<DataType> asArray() { return m_view; }
  ConstArray2View<DataType> asArray() const { return m_view; }
  
  Integer totalNbElement() const { return m_view.totalNbElement(); }

  Integer arraySize() const { return m_view.dim2Size(); }
  
  bool isArrayVariable() const { return true; }
 
  ARCANE_CORE_EXPORT void updateFromInternal();

  ARCANE_CORE_EXPORT ItemGroup itemGroup() const;

  /*
   * \brief Redimensionne le nombre d'éléments du tableau.
   *
   * La première dimension reste toujours égale au nombre d'éléments du maillage.
   * Seule la deuxième composante est retaillée.
   * \warning le redimensionnement ne conserve pas les valeurs précédentes...
   */
  ARCANE_CORE_EXPORT void resize(Int32 dim2_size);
 
  /*
   * \brief Redimensionne le nombre d'éléments du tableau.
   *
   * \sa resize(Int32)
   */
  ARCANE_CORE_EXPORT void resizeAndReshape(const ArrayShape& shape);

 public:

  SmallSpan2<DataType> _internalSpan() { return m_view; }
  SmallSpan2<const DataType> _internalSpan() const { return m_view; }
  SmallSpan2<const DataType> _internalConstSpan() const { return m_view; }

 protected:
  
  void _internalInit() { MeshVariableRef::_internalInit(m_private_part); }
  
 protected:

  PrivatePartType* m_private_part;
    
  Array2View<DataType> m_view;  
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
