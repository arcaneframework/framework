// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshMaterialVariableRef.cc                                  (C) 2000-2024 */
/*                                                                           */
/* Référence à une variable sur un matériau du maillage.                     */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/NotImplementedException.h"
#include "arcane/utils/TraceInfo.h"
#include "arcane/utils/NumericTypes.h"

#include "arcane/core/MeshVariableScalarRef.h"
#include "arcane/core/VariableBuildInfo.h"
#include "arcane/core/ArcaneException.h"

#include "arcane/core/materials/IMeshMaterialMng.h"
#include "arcane/core/materials/IMeshMaterial.h"
#include "arcane/core/materials/MaterialVariableBuildInfo.h"
#include "arcane/core/materials/MeshMaterialVariableRef.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshMaterialVariableRef::
MeshMaterialVariableRef()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshMaterialVariableRef::
~MeshMaterialVariableRef()
{
  if (m_is_registered)
    unregisterVariable();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialVariableRef::
unregisterVariable()
{
  _checkValid();
  m_material_variable->removeVariableRef(this);
  m_is_registered = false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialVariableRef::
registerVariable()
{
  _checkValid();
  m_material_variable->addVariableRef(this);
  m_is_registered = true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialVariableRef::
_internalInit(IMeshMaterialVariable* mat_variable)
{
  m_material_variable = mat_variable;
  m_global_variable = mat_variable->globalVariable();
  registerVariable();
  updateFromInternal();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshMaterialVariableRef* MeshMaterialVariableRef::
previousReference()
{
  return m_previous_reference;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshMaterialVariableRef* MeshMaterialVariableRef::
nextReference()
{
  return m_next_reference;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialVariableRef::
setPreviousReference(MeshMaterialVariableRef* v)
{
  m_previous_reference = v;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialVariableRef::
setNextReference(MeshMaterialVariableRef* v)
{
  m_next_reference = v;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialVariableRef::
_throwInvalid() const
{
  ARCANE_THROW(InternalErrorException,"Trying to use uninitialized variable reference");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialVariableRef::
synchronize()
{
  _checkValid();
  m_material_variable->synchronize();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialVariableRef::
synchronize(MeshMaterialVariableSynchronizerList& sync_list)
{
  _checkValid();
  m_material_variable->synchronize(sync_list);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String MeshMaterialVariableRef::
name() const
{
  return m_global_variable->name();
}
void MeshMaterialVariableRef::
setUpToDate()
{
  m_global_variable->setUpToDate();
}
bool MeshMaterialVariableRef::
isUsed() const
{
  return m_global_variable->isUsed();
}
void MeshMaterialVariableRef::
update()
{
  m_global_variable->update();
}

void MeshMaterialVariableRef::
addDependCurrentTime(const VariableRef& var)
{
  m_global_variable->addDepend(var.variable(),IVariable::DPT_CurrentTime);
}
void MeshMaterialVariableRef::
addDependCurrentTime(const VariableRef& var,const TraceInfo& tinfo)
{
  m_global_variable->addDepend(var.variable(),IVariable::DPT_CurrentTime,tinfo);
}

void MeshMaterialVariableRef::
addDependCurrentTime(const MeshMaterialVariableRef& var)
{
  m_global_variable->addDepend(var.m_global_variable,IVariable::DPT_CurrentTime);
}

void MeshMaterialVariableRef::
addDependPreviousTime(const MeshMaterialVariableRef& var)
{
  m_global_variable->addDepend(var.m_global_variable,IVariable::DPT_PreviousTime);
}

void MeshMaterialVariableRef::
removeDepend(const MeshMaterialVariableRef& var)
{
  m_global_variable->removeDepend(var.m_global_variable);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialVariableRef::
setUpToDate(IMeshMaterial* mat)
{
  m_material_variable->setUpToDate(mat);
}

void MeshMaterialVariableRef::
update(IMeshMaterial* mat)
{
  m_material_variable->update(mat);
}

void MeshMaterialVariableRef::
addMaterialDepend(const VariableRef& var)
{
  m_material_variable->addDepend(var.variable());
}

void MeshMaterialVariableRef::
addMaterialDepend(const VariableRef& var,const TraceInfo& tinfo)
{
  m_material_variable->addDepend(var.variable(),tinfo);
}

void MeshMaterialVariableRef::
addMaterialDepend(const MeshMaterialVariableRef& var)
{
  m_material_variable->addDepend(var.materialVariable());
}

void MeshMaterialVariableRef::
addMaterialDepend(const MeshMaterialVariableRef& var,const TraceInfo& tinfo)
{
  m_material_variable->addDepend(var.materialVariable(),tinfo);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> CellMaterialVariableScalarRef<DataType>::
CellMaterialVariableScalarRef(const VariableBuildInfo& vb)
: CellMaterialVariableScalarRef(MaterialVariableBuildInfo(nullptr,vb))
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> CellMaterialVariableScalarRef<DataType>::
CellMaterialVariableScalarRef(const MaterialVariableBuildInfo& vb)
: m_private_part(PrivatePartType::BuilderType::getVariableReference(vb,MatVarSpace::MaterialAndEnvironment))
{
  _init();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType>  CellMaterialVariableScalarRef<DataType>::
CellMaterialVariableScalarRef(const CellMaterialVariableScalarRef<DataType>& rhs)
: m_private_part(rhs.m_private_part)
{
  // Il faut incrémenter manuellement le compteur de référence car normalement
  // cela est fait dans getReference() mais ici on ne l'appelle pas
  if (m_private_part)
    m_private_part->incrementReference();

  _init();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType>  CellMaterialVariableScalarRef<DataType>::
CellMaterialVariableScalarRef(IMeshMaterialVariable* var)
: m_private_part(PrivatePartType::BuilderType::getVariableReference(var))
{
  // Il faut incrémenter manuellement le compteur de référence car normalement
  // cela est fait dans getReference() mais ici on ne l'appelle pas
  m_private_part->incrementReference();

  _init();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> void
CellMaterialVariableScalarRef<DataType>::
_init()
{
  if (m_private_part){
    _setContainerView();
    _internalInit(m_private_part->toMeshMaterialVariable());
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> void
CellMaterialVariableScalarRef<DataType>::
refersTo(const CellMaterialVariableScalarRef<DataType>& rhs)
{
  if (rhs.m_private_part==m_private_part)
    return;
  if (_isRegistered())
    unregisterVariable();
  m_private_part = rhs.m_private_part;
  m_container_value = {};
  m_value = nullptr;

  // Il faut incrémenter manuellement le compteur de référence car normalement
  // cela est fait dans getReference() mais ici on ne l'appelle pas
  if (m_private_part)
    m_private_part->incrementReference();
  _init();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> void
CellMaterialVariableScalarRef<DataType>::
updateFromInternal()
{
  _setContainerView();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> DataType
CellMaterialVariableScalarRef<DataType>::
matValue(AllEnvCell c,Int32 mat_id) const
{
  ENUMERATE_CELL_ENVCELL(ienvcell,c){
    ENUMERATE_CELL_MATCELL(imatcell,(*ienvcell)){
      MatCell mc = *imatcell;
      Int32 mid = mc.materialId();
      if (mid==mat_id)
        return this->operator[](imatcell);
    }
  }
  return DataType();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> DataType
CellMaterialVariableScalarRef<DataType>::
envValue(AllEnvCell c,Int32 env_id) const
{
  ENUMERATE_CELL_ENVCELL(ienvcell,c){
    EnvCell ec = *ienvcell;
    Int32 eid = ec.environmentId();
    if (eid==env_id)
      return this->operator[](ienvcell);
  }
  return DataType();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Remplit les valeurs de la variable pour un matériau.
 *
 * Cette méthode effectue l'opération suivante:
 \code
 * Integer index=0;
 * ENUMERATE_MATCELL(imatcell,mat){
 *  matvar[imatcell] = values[index];
 *  ++index;
 * }
 \endcode
*/
template<typename DataType> void
CellMaterialVariableScalarRef<DataType>::
fillFromArray(IMeshMaterial* mat,ConstArrayView<DataType> values)
{
  m_private_part->fillFromArray(mat,values);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Remplit les valeurs de la variable pour un matériau.
 *
 * Cette méthode effectue l'opération suivante:
 \code
 * Integer index=0;
 * ENUMERATE_MATCELL(imatcell,mat){
 *  matvar[imatcell] = values[index];
 *  ++index;
 * }
 \endcode
*/
template<typename DataType> void
CellMaterialVariableScalarRef<DataType>::
fillFromArray(IMeshMaterial* mat,ConstArrayView<DataType> values,Int32ConstArrayView indexes)
{
  m_private_part->fillFromArray(mat,values,indexes);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Remplit un tableau à partir des valeurs de la variable pour un matériau.
 *
 * Cette méthode effectue l'opération suivante:
 \code
 * Integer index=0;
 * ENUMERATE_MATCELL(imatcell,mat){
 *  values[index] = matvar[imatcell];
 *  ++index;
 * }
 \endcode
*/
template<typename DataType> void
CellMaterialVariableScalarRef<DataType>::
fillToArray(IMeshMaterial* mat,ArrayView<DataType> values)
{
  m_private_part->fillToArray(mat,values);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Remplit un tableau à partir des valeurs de la variable pour un matériau.
 *
 * Cette méthode effectue l'opération suivante:
 \code
 * Integer index=0;
 * ENUMERATE_MATCELL(imatcell,mat){
 *  values[index] = matvar[imatcell];
 *  ++index;
 * }
 \endcode
*/
template<typename DataType> void
CellMaterialVariableScalarRef<DataType>::
fillToArray(IMeshMaterial* mat,ArrayView<DataType> values,Int32ConstArrayView indexes)
{
  m_private_part->fillToArray(mat,values,indexes);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Remplit un tableau à partir des valeurs de la variable pour un matériau.
 *
 * Le tableau \a values est redimensionné si besoin.
 */
template<typename DataType> void
CellMaterialVariableScalarRef<DataType>::
fillToArray(IMeshMaterial* mat,Array<DataType>& values)
{
  values.resize(mat->cells().size());
  fillToArray(mat,values.view());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Remplit un tableau à partir des valeurs de la variable pour un matériau.
 *
 * Le tableau \a values est redimensionné si besoin.
 */
template<typename DataType> void
CellMaterialVariableScalarRef<DataType>::
fillToArray(IMeshMaterial* mat,Array<DataType>& values,Int32ConstArrayView indexes)
{
  values.resize(mat->cells().size());
  fillToArray(mat,values.view(),indexes);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Remplit les valeurs partielles et globales de la variable avec la valeur \a value
 */
template<typename DataType> void
CellMaterialVariableScalarRef<DataType>::
fill(const DataType& value)
{
  globalVariable().fill(value);
  fillPartialValues(value);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Remplit les valeurs partielles de la variable avec la valeur \a value
 */
template<typename DataType> void
CellMaterialVariableScalarRef<DataType>::
fillPartialValues(const DataType& value)
{
  m_private_part->fillPartialValues(value);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> MeshVariableScalarRefT<Cell,DataType>&
CellMaterialVariableScalarRef<DataType>::
globalVariable()
{
  GlobalVariableRefType* rt = m_private_part->globalVariableReference();
  if (!rt)
    ARCANE_FATAL("null global variable");
  return *rt;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> const MeshVariableScalarRefT<Cell,DataType>&
CellMaterialVariableScalarRef<DataType>::
globalVariable() const
{
  GlobalVariableRefType* rt = m_private_part->globalVariableReference();
  if (!rt)
    ARCANE_FATAL("null global variable");
  return *rt;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> void
CellMaterialVariableScalarRef<DataType>::
_setContainerView()
{
  if (m_private_part){
    m_container_value = m_private_part->_internalFullValuesView();
    m_value = m_container_value.data();
  }
  else{
    m_container_value = {};
    m_value = nullptr;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// TODO: fusionner avec la version scalaire
template<typename DataType> CellMaterialVariableArrayRef<DataType>::
CellMaterialVariableArrayRef(const VariableBuildInfo& vb)
: CellMaterialVariableArrayRef(MaterialVariableBuildInfo(nullptr,vb))
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// TODO: fusionner avec la version scalaire
template<typename DataType> CellMaterialVariableArrayRef<DataType>::
CellMaterialVariableArrayRef(const MaterialVariableBuildInfo& vb)
: m_private_part(PrivatePartType::BuilderType::getVariableReference(vb,MatVarSpace::MaterialAndEnvironment))
{
  _init();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> CellMaterialVariableArrayRef<DataType>::
CellMaterialVariableArrayRef(IMeshMaterialVariable* var)
: m_private_part(PrivatePartType::BuilderType::getVariableReference(var))
{
  // Il faut incrémenter manuellement le compteur de référence car normalement
  // cela est fait dans getReference() mais ici on ne l'appelle pas
  m_private_part->incrementReference();

  _init();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// TODO: fusionner avec la version scalaire
template<typename DataType>  CellMaterialVariableArrayRef<DataType>::
CellMaterialVariableArrayRef(const CellMaterialVariableArrayRef<DataType>& rhs)
: m_private_part(rhs.m_private_part)
, m_value(nullptr)
{
  // Il faut incrémenter manuellement le compteur de référence car normalement
  // cela est fait dans getReference() mais ici on ne l'appelle pas
  if (m_private_part)
    m_private_part->incrementReference();

  _init();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// TODO: fusionner avec la version scalaire
template<typename DataType> void
CellMaterialVariableArrayRef<DataType>::
_init()
{
  if (m_private_part){
    _setContainerView();
    _internalInit(m_private_part->toMeshMaterialVariable());
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// TODO: fusionner avec la version scalaire
template<typename DataType> void
CellMaterialVariableArrayRef<DataType>::
refersTo(const CellMaterialVariableArrayRef<DataType>& rhs)
{
  if (rhs.m_private_part==m_private_part)
    return;
  if (_isRegistered())
    unregisterVariable();

  m_private_part = rhs.m_private_part;
  m_value = nullptr;
  m_container_value = {};

  // Il faut incrémenter manuellement le compteur de référence car normalement
  // cela est fait dans getReference() mais ici on ne l'appelle pas
  if (m_private_part)
    m_private_part->incrementReference();
  _init();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// TODO: fusionner avec la version scalaire
template<typename DataType> void
CellMaterialVariableArrayRef<DataType>::
updateFromInternal()
{
  _setContainerView();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> MeshVariableArrayRefT<Cell,DataType>&
CellMaterialVariableArrayRef<DataType>::
globalVariable()
{
  GlobalVariableRefType* rt = m_private_part->globalVariableReference();
  if (!rt)
    ARCANE_FATAL("null global variable");
  return *rt;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> const MeshVariableArrayRefT<Cell,DataType>&
CellMaterialVariableArrayRef<DataType>::
globalVariable() const
{
  GlobalVariableRefType* rt = m_private_part->globalVariableReference();
  if (!rt)
    ARCANE_FATAL("null global variable");
  return *rt;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> void
CellMaterialVariableArrayRef<DataType>::
resize(Integer dim2_size)
{
  m_private_part->resize(dim2_size);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType> void
CellMaterialVariableArrayRef<DataType>::
_setContainerView()
{
  if (m_private_part){
    m_container_value = m_private_part->_internalFullValuesView();
    m_value = m_container_value.data();
  }
  else{
    m_container_value = {};
    m_value = nullptr;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#define ARCANE_INSTANTIATE_MAT(type) \
  template class ARCANE_TEMPLATE_EXPORT CellMaterialVariableScalarRef<type>;\
  template class ARCANE_TEMPLATE_EXPORT CellMaterialVariableArrayRef<type>

ARCANE_INSTANTIATE_MAT(Byte);
ARCANE_INSTANTIATE_MAT(Int8);
ARCANE_INSTANTIATE_MAT(Int16);
ARCANE_INSTANTIATE_MAT(Int32);
ARCANE_INSTANTIATE_MAT(Int64);
ARCANE_INSTANTIATE_MAT(BFloat16);
ARCANE_INSTANTIATE_MAT(Float16);
ARCANE_INSTANTIATE_MAT(Float32);
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
