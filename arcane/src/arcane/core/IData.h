// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IData.h                                                     (C) 2000-2023 */
/*                                                                           */
/* Interface d'une donnée.                                                   */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IDATA_H
#define ARCANE_IDATA_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface d'une donnée.
 *
 * Cette classe gère la mémoire associée à une variable.
 */
class ARCANE_CORE_EXPORT IData
{
  ARCCORE_DECLARE_REFERENCE_COUNTED_INCLASS_METHODS();

 public:

  virtual ~IData() = default;

 public:

  //! Type de la donnée
  virtual eDataType dataType() const = 0;

  //! Dimension. 0 pour un scalaire, 1 pour un tableau mono-dim, 2 pour un tableau bi-dim.
  virtual Integer dimension() const = 0;

  //! Tag multiple. 0 si non multiple, 1 si multiple, 2 si multiple pour les variable MultiArray (obsolète)
  virtual Integer multiTag() const = 0;

  //! Clone la donnée. L'instance créée doit être détruite par l'opérateur 'delete'
  ARCCORE_DEPRECATED_2020("Use cloneRef() instead")
  virtual IData* clone() = 0;

  //! Clone la donnée mais sans éléments. L'instance créée doit être détruite par l'opérateur 'delete'
  ARCCORE_DEPRECATED_2020("Use cloneEmptyRef() instead")
  virtual IData* cloneEmpty() = 0;

  //! Clone la donnée
  virtual Ref<IData> cloneRef() = 0;

  //! Clone la donnée mais sans éléments.
  virtual Ref<IData> cloneEmptyRef() = 0;

  //! Informations sur le type de conteneur de la donnée
  virtual DataStorageTypeInfo storageTypeInfo() const = 0;

  //! Sérialise la donnée en appliquant l'opération \a operation
  virtual void serialize(ISerializer* sbuf, IDataOperation* operation) = 0;

  /*!
   * \brief Redimensionne la donnée.
   *
   * Cette opération n'a de sens que pour les données de dimension 1 ou plus.
   * Si le nouveau nombre d'éléments est supérieur à l'ancien, les valeurs ajoutées à
   * la donnée ne sont pas initialisées.
   */
  virtual void resize(Integer new_size) = 0;

  /*!
   * \brief Serialise la donnée pour les indices \a ids.
   *
   * Cette opération n'a de sens que pour les données de dimension 1 ou plus.
   */
  virtual void serialize(ISerializer* sbuf, Int32ConstArrayView ids, IDataOperation* operation) = 0;

  //! Remplit la donnée avec sa valeur par défaut.
  virtual void fillDefault() = 0;

  //! Positionne le nom de la donnée (interne)
  virtual void setName(const String& name) = 0;

  /*!
   * \brief Sérialise la donnée.
   *
   * Pour des raisons de performances, l'instance retournée peut faire
   * directement référence à la zone mémoire de cette donnée. Par
   * conséquent, elle n'est valide que tant que cette donnée n'est
   * pas modifiée. Si on souhaite modifier cette instance, il faut
   * d'abord la cloner (via IData::cloneRef()) puis sérialiser la donnée clonée.
   *
   * Si \a use_basic_type est vrai, la donnée est sérialisée pour un type
   * de base, à savoir #DT_Byte, #DT_Int16, #DT_Int32, #DT_Int64 ou #DT_Real. Sinon,
   * le type peut être un POD, à savoir #DT_Byte, #DT_Int16, #DT_Int32, #DT_Int64,
   * #DT_Real, #DT_Real2, #DT_Real3, #DT_Real2x2, #DT_Real3x3.
   */
  virtual Ref<ISerializedData> createSerializedDataRef(bool use_basic_type) const = 0;

  /*!
   * \brief Assigne à la donnée les valeurs sérialisées \a sdata.
   *
   * Le tampon contenant les valeurs de sérialisation doit avoir
   * être alloué par appel à allocateBufferForSerializedData().
   */
  virtual void assignSerializedData(const ISerializedData* sdata) = 0;

  /*!
   * \brief Alloue la mémoire pour lire les valeurs sérialisées \a sdata.
   *
   * Cette méthode positionne sdata->setBuffer() qui contiendra la
   * mémoire nécessaire pour lire les données sérialisées.
   */
  virtual void allocateBufferForSerializedData(ISerializedData* sdata) = 0;

  /*!
   * \brief Copie la donnée \a data dans l'instance courante.
   *
   * La donnée \a data doit être du même type que l'instance.
   */
  virtual void copy(const IData* data) = 0;

  /*!
   * \brief Échange les valeurs de \a data avec celles de l'instance.
   *
   * La donnée \a IData doit être du même type que l'instance. Seules
   * les valeurs sont échangés et les autres propriétés éventuelles
   * (telles que le nom par exemple) ne sont pas modifiées.
   */
  virtual void swapValues(IData* data) = 0;

  /*!
   * \brief Calcul une clé de hashage sur cette donnée.
   *
   * La clé est ajoutée dans \a output. La longueur de la clé dépend
   * de l'algorithme utilisé.
   */
  virtual void computeHash(IHashAlgorithm* algo, ByteArray& output) const = 0;

  /*!
   * \brief Forme du tableau pour une donnée 1D ou 2D.
   *
   * La forme n'est prise en compte que pour les dimensions supérieures à 1.
   * Pour une donnée 1D, la forme est donc par défaut {1}. Pour un tableau 2D,
   * la forme vaut par défaut {dim2_size}. Il est possible de changer le rang
   * de la forme et ses valeurs tant que shape().totalNbElement()==dim2_size.
   * Par exemple si le nombre de valeurs dim2_size vaut 12, alors il est
   * possible d'avoir { 12 }, { 6, 2 } ou { 3, 2, 2 } comme forme.
   *
   * Les valeurs ne sont pas conservés lors d'une reprise et il faut donc
   * repositionner la forme dans ce cas. C'est à l'utilisateur de s'assurer
   * que la forme est homogène entre les sous-domaines.
   */
  virtual ArrayShape shape() const = 0;

  //! Positionne la forme du tableau.
  virtual void setShape(const ArrayShape& new_shape) = 0;

 public:

  //! Positionne les informations sur l'allocation
  virtual void setAllocationInfo(const DataAllocationInfo& v) = 0;

  //! Informations sur l'allocation
  virtual DataAllocationInfo allocationInfo() const = 0;

 public:

  //! Applique le visiteur à la donnée
  virtual void visit(IDataVisitor* visitor) = 0;

  /*!
   * \brief Applique le visiteur à la donnée.
   *
   * Si la donnée n'est pas scalaire, une exception
   * NotSupportedException est lancée.
   */
  virtual void visitScalar(IScalarDataVisitor* visitor) = 0;

  /*!
   * \brief Applique le visiteur à la donnée.
   *
   * Si la donnée n'est pas un tableau 1D, une exception 
   * NotSupportedException est lancée.
   */
  virtual void visitArray(IArrayDataVisitor* visitor) = 0;

  /*!
   * \brief Applique le visiteur à la donnée.
   *
   * Si la donnée n'est pas un tableau 2D, une exception 
   * NotSupportedException est lancée.
   */
  virtual void visitArray2(IArray2DataVisitor* visitor) = 0;

  /*!
   * \brief Applique le visiteur à la donnée.
   *
   * Si la donnée n'est pas un tableau 2D, une exception 
   * NotSupportedException est lancée.
   *
   * \deprecated Ce visiteur est obsolète car il n'y a pas plus
   * d'implémentation de IMultiArray2.
   */
  virtual void visitMultiArray2(IMultiArray2DataVisitor* visitor);

  //! \internal
  virtual IDataInternal* _commonInternal() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface d'une donnée scalaire.
 */
class IScalarData
: public IData
{
 public:
  virtual void visit(IDataVisitor* visitor) = 0;
  //! Applique le visiteur à la donnée.
  virtual void visit(IScalarDataVisitor* visitor) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Interface d'une donnée scalaire d'un type \a T
 */
template <class DataType>
class IScalarDataT
: public IScalarData
{
 public:
  typedef IScalarDataT<DataType> ThatClass;

 public:
  //! Valeur de la donnée
  virtual DataType& value() = 0;

  //! Valeur de la donnée
  virtual const DataType& value() const = 0;

  //! Clone la donnée
  ARCCORE_DEPRECATED_2020("Use cloneTrueRef() instead")
  virtual ThatClass* cloneTrue() = 0;

  //! Clone la donnée mais sans éléments.
  ARCCORE_DEPRECATED_2020("Use cloneTrueEmpty() instead")
  virtual ThatClass* cloneTrueEmpty() = 0;

  //! Clone la donnée
  virtual Ref<ThatClass> cloneTrueRef() = 0;

  //! Clone la donnée mais sans éléments.
  virtual Ref<ThatClass> cloneTrueEmptyRef() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface d'une donnée tableau 1D.
 */
class IArrayData
: public IData
{
 public:
  virtual void visit(IDataVisitor* visitor) = 0;
  //! Applique le visiteur à la donnée.
  virtual void visit(IArrayDataVisitor* visitor) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Interface d'une donnée tableau d'un type \a T
 */
template <class DataType>
class IArrayDataT
: public IArrayData
{
 public:

  typedef IArrayDataT<DataType> ThatClass;

 public:

  //! Valeur de la donnée
  ARCCORE_DEPRECATED_2021("Use view() instead.")
  virtual Array<DataType>& value() = 0;

  //! Valeur constante de la donnée
  ARCCORE_DEPRECATED_2021("Use view() instead.")
  virtual const Array<DataType>& value() const = 0;

 public:

  //! Vue constante sur la donnée
  virtual ConstArrayView<DataType> view() const = 0;

  //! Vue sur la donnée
  virtual ArrayView<DataType> view() = 0;

  //! Clone la donnée
  ARCCORE_DEPRECATED_2020("Use cloneTrueRef() instead")
  virtual ThatClass* cloneTrue() = 0;

  //! Clone la donnée mais sans éléments.
  ARCCORE_DEPRECATED_2020("Use cloneTrueEmptyRef() instead")
  virtual ThatClass* cloneTrueEmpty() = 0;

  //! Clone la donnée
  virtual Ref<ThatClass> cloneTrueRef() = 0;

  //! Clone la donnée mais sans éléments.
  virtual Ref<ThatClass> cloneTrueEmptyRef() = 0;

  //! \internal
  virtual IArrayDataInternalT<DataType>* _internal() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface d'une donnée tableau 2D.
 */
class IArray2Data
: public IData
{
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface d'une donnée tableau multi 2D.
 * \deprecated Cette interface n'est plus utilisée.
 */
class IMultiArray2Data
: public IData
{
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Interface d'une donnée tableau bi-dimensionnel d'un type \a T
 */
template <class DataType>
class IArray2DataT
: public IArray2Data
{
 public:

  typedef IArray2DataT<DataType> ThatClass;

  //! Valeur de la donnée
  ARCCORE_DEPRECATED_2021("Use view() instead.")
  virtual Array2<DataType>& value() = 0;

  //! Valeur de la donnée
  ARCCORE_DEPRECATED_2021("Use view() instead.")
  virtual const Array2<DataType>& value() const = 0;

 public:

  //! Vue constante sur la donnée
  virtual ConstArray2View<DataType> view() const = 0;

  //! Vue sur la donnée
  virtual Array2View<DataType> view() = 0;

  //! Clone la donnée
  ARCCORE_DEPRECATED_2020("Use cloneTrueRef() instead")
  virtual ThatClass* cloneTrue() = 0;

  //! Clone la donnée mais sans éléments.
  ARCCORE_DEPRECATED_2020("Use cloneTrueEmptyRef() instead")
  virtual ThatClass* cloneTrueEmpty() = 0;

  //! Clone la donnée
  virtual Ref<ThatClass> cloneTrueRef() = 0;

  //! Clone la donnée mais sans éléments.
  virtual Ref<ThatClass> cloneTrueEmptyRef() = 0;

  //! \internal
  virtual IArray2DataInternalT<DataType>* _internal() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Interface d'une donnée tableau 2D à taille multiple d'un type \a T
 * \deprecated Cette interface n'est plus utilisée.
 */
template <class DataType>
class IMultiArray2DataT
: public IMultiArray2Data
{
 public:
  typedef IMultiArray2DataT<DataType> ThatClass;


  //! Valeur de la donnée
  virtual MultiArray2<DataType>& value() = 0;

  //! Valeur de la donnée
  virtual const MultiArray2<DataType>& value() const = 0;

  //! Clone la donnée
  virtual ThatClass* cloneTrue() = 0;

  //! Clone la donnée mais sans éléments.
  virtual ThatClass* cloneTrueEmpty() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
