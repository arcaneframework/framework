// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshMaterialVariableRef.h                                   (C) 2000-2025 */
/*                                                                           */
/* Référence à une variable sur un matériau du maillage.                     */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MATERIALS_MESHMATERIALVARIABLEREF_H
#define ARCANE_MATERIALS_MESHMATERIALVARIABLEREF_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \file MeshMaterialVariableRef.h
 *
 * Ce fichier contient les différents types gérant les références
 * sur les variables matériaux.
 */
#include "arcane/utils/NotImplementedException.h"
#include "arcane/utils/Array2View.h"

#include "arcane/core/Item.h"
#include "arcane/core/VariableRef.h"

#include "arcane/core/materials/IMeshMaterialVariable.h"
#include "arcane/core/materials/MatItemEnumerator.h"
#include "arcane/core/materials/MeshMaterialVariableComputeFunction.h"
#include "arcane/core/materials/IScalarMeshMaterialVariable.h"
#include "arcane/core/materials/IArrayMeshMaterialVariable.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup ArcaneMaterials
 * \brief Classe de base des références aux variables matériaux.
 */
class ARCANE_CORE_EXPORT MeshMaterialVariableRef
{
 public:
  class Enumerator
  {
   public:

    explicit Enumerator(const IMeshMaterialVariable* vp)
    : m_vref(vp->firstReference())
    {
    }
    void operator++()
    {
      m_vref = m_vref->nextReference();
    }
    MeshMaterialVariableRef* operator*() const
    {
      return m_vref;
    }
    bool hasNext() const
    {
      return m_vref;
    }

   private:

    MeshMaterialVariableRef* m_vref = nullptr;
  };

 public:

  MeshMaterialVariableRef();
  virtual ~MeshMaterialVariableRef();

 public:

  //! Référence précédente (ou null) sur variable()
  MeshMaterialVariableRef* previousReference();

  //! Référence suivante (ou null) sur variable()
  MeshMaterialVariableRef* nextReference();

  /*!
   * \internal
   * \brief Positionne la référence précédente.
   *
   * For internal use only.
   */
  void setPreviousReference(MeshMaterialVariableRef* v);

  /*!
   * \internal
   * \brief Positionne la référence suivante.
   *
   * For internal use only.
   */
  void setNextReference(MeshMaterialVariableRef* v);

  //! Enregistre la variable (interne)
  void registerVariable();

  //! Supprime l'enregistrement de la variable (interne)
  void unregisterVariable();

  virtual void updateFromInternal() =0;
  
  //! Variable matériau associée.
  IMeshMaterialVariable* materialVariable() const { return m_material_variable; }

  //! Synchronise les valeurs entre les sous-domaines
  void synchronize();

  //! Ajoute cette variable à la liste des synchronisations \a sync_list.
  void synchronize(MeshMaterialVariableSynchronizerList& sync_list);

  //! Espace de définition de la variable (matériau+milieu ou milieu uniquement)
  MatVarSpace space() const { return m_material_variable->space(); }

  /*!
   * \brief Remplit les valeurs partielles avec la valeur de la maille du dessus.
   * Si \a level vaut LEVEL_MATERIAL, copie les valeurs matériaux avec celle du milieu.
   * Si \a level vaut LEVEL_ENVIRONNEMENT, copie les valeurs des milieux avec
   * celui de la maille globale.
   * Si \a level vaut LEVEL_ALLENVIRONMENT, remplit toutes les valeurs partielles
   * avec celle de la maille globale (cela rend cette méthode équivalente à
   * fillGlobalValuesWithGlobalValues().
   */
  void fillPartialValuesWithSuperValues(Int32 level)
  {
    m_material_variable->fillPartialValuesWithSuperValues(level);
  }

 public:

  // Fonctions issues de VariableRef. A terme, il faudra que la variable
  // materiau dérive de la variable classique.
  //@{ Fonctions issues de VariablesRef. Ces fonctions s'appliquent sur la variable globale associée.
  String name() const;
	void setUpToDate();
	bool isUsed() const;
	void update();
	void addDependCurrentTime(const VariableRef& var);
	void addDependCurrentTime(const VariableRef& var,const TraceInfo& tinfo);
  void addDependCurrentTime(const MeshMaterialVariableRef& var);
  void addDependPreviousTime(const MeshMaterialVariableRef& var);
  void removeDepend(const MeshMaterialVariableRef& var);
	template<typename ClassType> void
	setComputeFunction(ClassType* instance,void (ClassType::*func)())
	{ m_global_variable->setComputeFunction(new VariableComputeFunction(instance,func)); }
  //@}

  //! Fonctions pour gérer les dépendances sur la partie matériau de la variable.
  //@{
	void setUpToDate(IMeshMaterial*);
	void update(IMeshMaterial*);
	void addMaterialDepend(const VariableRef& var);
	void addMaterialDepend(const VariableRef& var,const TraceInfo& tinfo);
	void addMaterialDepend(const MeshMaterialVariableRef& var);
	void addMaterialDepend(const MeshMaterialVariableRef& var,const TraceInfo& tinfo);
	template<typename ClassType> void
	setMaterialComputeFunction(ClassType* instance,void (ClassType::*func)(IMeshMaterial*))
	{ m_material_variable->setComputeFunction(new MeshMaterialVariableComputeFunction(instance,func)); }
  //@}

 protected:
  
  void _internalInit(IMeshMaterialVariable* mat_variable);
  bool _isRegistered() const { return m_is_registered; }

 private:

  //! Variable associée
  IMeshMaterialVariable* m_material_variable;

  //! Référence précédente sur \a m_variable
  MeshMaterialVariableRef* m_previous_reference;

  //! Référence suivante sur \a m_variable
  MeshMaterialVariableRef* m_next_reference;

  //! Variable globale associée
  IVariable* m_global_variable;

  bool m_is_registered;

 private:
  void _checkValid() const
  {
#ifdef ARCANE_CHECK
    if (!m_material_variable)
      _throwInvalid();
#endif
  }
  void _throwInvalid() const;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup ArcaneMaterials
 * \brief Variable scalaire sur les mailles d'un matériau du maillage.
 * Pour l'instant, cette classe n'est instanciée que pour les mailles
 */
template<typename DataType_>
class CellMaterialVariableScalarRef
: public MeshMaterialVariableRef
{
 public:

  using DataType = DataType_;
  using PrivatePartType = IScalarMeshMaterialVariable<Cell, DataType>;
  using ItemType = Cell;
  using GlobalVariableRefType = MeshVariableScalarRefT<ItemType, DataType>;
  using ThatClass = CellMaterialVariableScalarRef<DataType>;

 public:

  explicit ARCANE_CORE_EXPORT CellMaterialVariableScalarRef(const VariableBuildInfo& vb);
  //! Construit une référence à la variable spécifiée dans \a vb
  explicit ARCANE_CORE_EXPORT CellMaterialVariableScalarRef(const MaterialVariableBuildInfo& vb);
  ARCANE_CORE_EXPORT CellMaterialVariableScalarRef(const ThatClass& rhs);

 public:

  //! Opérateur de recopie (interdit)
  ARCANE_CORE_EXPORT ThatClass& operator=(const ThatClass& rhs) = delete;
  //! Constructeur vide (interdit)
  CellMaterialVariableScalarRef() = delete;

 public:

  //! Positionne la référence de l'instance à la variable \a rhs.
  ARCANE_CORE_EXPORT virtual void refersTo(const ThatClass& rhs);

  /*!
   * \internal
   */
  ARCANE_CORE_EXPORT void updateFromInternal() override;

 protected:

  DataType operator[](MatVarIndex mvi) const
  {
    return m_value[mvi.arrayIndex()][mvi.valueIndex()];
  }
  DataType& operator[](MatVarIndex mvi)
  {
    return m_value[mvi.arrayIndex()][mvi.valueIndex()];
  }

 public:

  //! Valeur partielle de la variable pour la maille matériau \a mc
  DataType operator[](ComponentItemLocalId mc) const
  {
    return this->operator[](mc.localId());
  }

  //! Valeur partielle de la variable pour la maille matériau \a mc
  DataType& operator[](ComponentItemLocalId mc)
  {
    return this->operator[](mc.localId());
  }

  //! Valeur partielle de la variable pour l'itérateur \a mc
  DataType operator[](CellComponentCellEnumerator mc) const
  {
    return this->operator[](mc._varIndex());
  }

  //! Valeur partielle de la variable pour l'itérateur \a mc
  DataType& operator[](CellComponentCellEnumerator mc)
  {
    return this->operator[](mc._varIndex());
  }

  //! Valeur globale de la variable pour la maille \a c
  DataType operator[](CellLocalId c) const
  {
    return m_value[0][c.localId()];
  }

  //! Valeur globale de la variable pour la maille \a c
  DataType& operator[](CellLocalId c)
  {
    return m_value[0][c.localId()];
  }

  //! Valeur de la variable pour la maille matériau \a mvi
  DataType operator[](PureMatVarIndex mvi) const
  {
    return m_value[0][mvi.valueIndex()];
  }

  //! Valeur de la variable pour la maille matériau \a mvi
  DataType& operator[](PureMatVarIndex mvi)
  {
    return m_value[0][mvi.valueIndex()];
  }

  /*!
   * \brief Valeur de la variable pour le matériau d'index \a mat_id de
   * la maille \a ou 0 si absent de la maille.
   */
  ARCANE_CORE_EXPORT DataType matValue(AllEnvCell c,Int32 mat_id) const;

  /*!
   * \brief Valeur de la variable pour le milieu d'index \a env_id de
   * la maille \a ou 0 si absent de la maille.
   */
  ARCANE_CORE_EXPORT DataType envValue(AllEnvCell c,Int32 env_id) const;

 public:
  
  ARCANE_CORE_EXPORT void fillFromArray(IMeshMaterial* mat,ConstArrayView<DataType> values);
  ARCANE_CORE_EXPORT void fillFromArray(IMeshMaterial* mat,ConstArrayView<DataType> values,Int32ConstArrayView indexes);
  ARCANE_CORE_EXPORT void fillToArray(IMeshMaterial* mat,ArrayView<DataType> values);
  ARCANE_CORE_EXPORT void fillToArray(IMeshMaterial* mat,ArrayView<DataType> values,Int32ConstArrayView indexes);
  ARCANE_CORE_EXPORT void fillToArray(IMeshMaterial* mat,Array<DataType>& values);
  ARCANE_CORE_EXPORT void fillToArray(IMeshMaterial* mat,Array<DataType>& values,Int32ConstArrayView indexes);
  ARCANE_CORE_EXPORT void fill(const DataType& value);
  ARCANE_CORE_EXPORT void fillPartialValues(const DataType& value);

 public:

  //! Variable globale associée à cette variable matériau
  ARCANE_CORE_EXPORT GlobalVariableRefType& globalVariable();
  //! Variable globale associée à cette variable matériau
  ARCANE_CORE_EXPORT const GlobalVariableRefType& globalVariable() const;

 private:

  PrivatePartType* m_private_part = nullptr;
  ArrayView<DataType>* m_value = nullptr;
  ArrayView<ArrayView<DataType>> m_container_value;

 private:

  void _init();
  void _setContainerView();

 public:

  // TODO: Temporaire. a supprimer.
  ArrayView<DataType>* _internalValue() const { return m_value; }

 public:

#ifdef ARCANE_DOTNET
  // Uniquement pour le wrapper C#
  // TODO: a terme utiliser 'm_container_view' à la place
  void* _internalValueAsPointerOfPointer() { return reinterpret_cast<void*>(&m_value); }
#endif
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup ArcaneMaterials
 * \brief Variable tableau sur les mailles d'un matériau du maillage.
 * Pour l'instant, cette classe n'est instanciée que pour les mailles
 */
template<typename DataType_>
class CellMaterialVariableArrayRef
: public MeshMaterialVariableRef
{
 public:

  using DataType = DataType_;
  using PrivatePartType = IArrayMeshMaterialVariable<Cell, DataType>;
  using ItemType = Cell;
  using GlobalVariableRefType = MeshVariableArrayRefT<ItemType, DataType>;
  using ThatClass = CellMaterialVariableArrayRef<DataType>;

 public:

  explicit ARCANE_CORE_EXPORT CellMaterialVariableArrayRef(const VariableBuildInfo& vb);
  //! Construit une référence à la variable spécifiée dans \a vb
  explicit ARCANE_CORE_EXPORT CellMaterialVariableArrayRef(const MaterialVariableBuildInfo& vb);
  ARCANE_CORE_EXPORT CellMaterialVariableArrayRef(const ThatClass& rhs);

 public:

  //! Opérateur de recopie (interdit)
  ARCANE_CORE_EXPORT ThatClass& operator=(const ThatClass& rhs) = delete;
  //! Constructeur vide (interdit)
  CellMaterialVariableArrayRef() = delete;

 public:

  //! Positionne la référence de l'instance à la variable \a rhs.
  ARCANE_CORE_EXPORT virtual void refersTo(const ThatClass& rhs);

  /*!
   * \internal
   */
  ARCANE_CORE_EXPORT void updateFromInternal() override;

 public:

  //! Variable globale associée à cette variable matériau
  ARCANE_CORE_EXPORT GlobalVariableRefType& globalVariable();
  //! Variable globale associée à cette variable matériau
  ARCANE_CORE_EXPORT const GlobalVariableRefType& globalVariable() const;

 public:

  /*!
   * \brief Redimensionne le nombre d'éléments du tableau.
   *
   * La première dimension reste toujours égale au nombre d'éléments du maillage.
   * Seule la deuxième composante est retaillée.
   */
  ARCANE_CORE_EXPORT void resize(Integer dim2_size);

 protected:

  ConstArrayView<DataType> operator[](MatVarIndex mvi) const
  {
    return m_value[mvi.arrayIndex()][mvi.valueIndex()];
  }
  ArrayView<DataType> operator[](MatVarIndex mvi)
  {
    return m_value[mvi.arrayIndex()][mvi.valueIndex()];
  }

 public:

  //! Valeur partielle de la variable pour la maille matériau \a mc
  ConstArrayView<DataType> operator[](ComponentItemLocalId mc) const
  {
    return this->operator[](mc.localId());
  }

  //! Valeur partielle de la variable pour la maille matériau \a mc
  ArrayView<DataType> operator[](ComponentItemLocalId mc)
  {
    return this->operator[](mc.localId());
  }

  //! Valeur globale de la variable pour la maille \a c
  ConstArrayView<DataType> operator[](CellLocalId c) const
  {
    return m_value[0][c.localId()];
  }

  //! Valeur globale de la variable pour la maille \a c
  ArrayView<DataType> operator[](CellLocalId c)
  {
    return m_value[0][c.localId()];
  }

  //! Valeur de la variable pour la maille matériau \a mvi
  ConstArrayView<DataType> operator[](PureMatVarIndex mvi) const
  {
    return m_value[0][mvi.valueIndex()];
  }

  //! Valeur de la variable pour la maille matériau \a mvi
  ArrayView<DataType> operator[](PureMatVarIndex mvi)
  {
    return m_value[0][mvi.valueIndex()];
  }

 private:

  PrivatePartType* m_private_part = nullptr;
  Array2View<DataType>* m_value = nullptr;
  ArrayView<Array2View<DataType>> m_container_value;

 private:

  void _init();
  void _setContainerView();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! %Variable matériau de type \a #Byte
typedef CellMaterialVariableScalarRef<Byte> MaterialVariableCellByte;
//! %Variable matériau de type \a #Real
typedef CellMaterialVariableScalarRef<Real> MaterialVariableCellReal;
//! %Variable matériau de type \a #Int16
typedef CellMaterialVariableScalarRef<Int16> MaterialVariableCellInt16;
//! %Variable matériau de type \a #Int32
typedef CellMaterialVariableScalarRef<Int32> MaterialVariableCellInt32;
//! %Variable matériau de type \a #Int64
typedef CellMaterialVariableScalarRef<Int64> MaterialVariableCellInt64;
//! %Variable matériau de type \a Real2
typedef CellMaterialVariableScalarRef<Real2> MaterialVariableCellReal2;
//! %Variable matériau de type \a Real3
typedef CellMaterialVariableScalarRef<Real3> MaterialVariableCellReal3;
//! %Variable matériau de type \a Real2x2
typedef CellMaterialVariableScalarRef<Real2x2> MaterialVariableCellReal2x2;
//! %Variable matériau de type \a Real3x3
typedef CellMaterialVariableScalarRef<Real3x3> MaterialVariableCellReal3x3;

#ifdef ARCANE_64BIT
//! %Variable matériau de type \a #Integer
typedef MaterialVariableCellInt64 MaterialVariableCellInteger;
#else
//! %Variable matériau de type \a #Integer
typedef MaterialVariableCellInt32 MaterialVariableCellInteger;
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! %Variable matériau de type tableau de \a #Byte
typedef CellMaterialVariableArrayRef<Byte> MaterialVariableCellArrayByte;
//! %Variable matériau de type tableau de \a #Real
typedef CellMaterialVariableArrayRef<Real> MaterialVariableCellArrayReal;
//! %Variable matériau de type tableau de \a #Int16
typedef CellMaterialVariableArrayRef<Int16> MaterialVariableCellArrayInt16;
//! %Variable matériau de type tableau de \a #Int32
typedef CellMaterialVariableArrayRef<Int32> MaterialVariableCellArrayInt32;
//! %Variable matériau de type tableau de \a #Int64
typedef CellMaterialVariableArrayRef<Int64> MaterialVariableCellArrayInt64;
//! %Variable matériau de type tableau de \a Real2
typedef CellMaterialVariableArrayRef<Real2> MaterialVariableCellArrayReal2;
//! %Variable matériau de type tableau de \a Real3
typedef CellMaterialVariableArrayRef<Real3> MaterialVariableCellArrayReal3;
//! %Variable matériau de type tableau de \a Real2x2
typedef CellMaterialVariableArrayRef<Real2x2> MaterialVariableCellArrayReal2x2;
//! %Variable matériau de type tableau de \a Real3x3
typedef CellMaterialVariableArrayRef<Real3x3> MaterialVariableCellArrayReal3x3;

#ifdef ARCANE_64BIT
//! %Variable matériau de type tableau de \a #Integer
typedef MaterialVariableCellArrayInt64 MaterialVariableCellArrayInteger;
#else
//! %Variable matériau de type tableau de \a #Integer
typedef MaterialVariableCellArrayInt32 MaterialVariableCellArrayInteger;
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

