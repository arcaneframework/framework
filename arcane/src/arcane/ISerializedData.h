// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ISerializedData.h                                           (C) 2000-2021 */
/*                                                                           */
/* Interface d'une donnée sérialisée.                                        */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ISERIALIZEDDATA_H
#define ARCANE_ISERIALIZEDDATA_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Interface d'une donnée sérialisée.
 *
 * Une donnée (IData) est sérialisée en une instance de cette classe.
 *
 * Quel que soit le type de la donnée, le type sérialisé est obligatoirement
 * un type de base parmi les suivants: DT_Byte, DT_Int32, DT_Int64, DT_Real.
 *
 * Une instance de cette classe n'est valable que tant que la donnée
 * de référence n'est pas modifiée.
 * 
 * Pour sérialiser une donnée \a data en écriture:
 * \code
 * IData* data = ...;
 * ISerializedData* sdata = data->createSerializedData();
 * // sdata->buffer() contient la donnée sérialisée.
 * // Écrire via sdata->buffer().
 * ...
 * \endcode
 *
 * Pour sérialiser une donnée \a data en lecture:
 * \code
 * // Récupére le IDataFactory à partir d'un \a IApplication:
 * IApplication* app = ...
 * IDataFactory* df = app->dataFactory();
 * // Créé une instance d'un ISerializedData.
 * ISerializedData* sdata = sf->createSerializedData(...);
 * data->allocateBufferForSerializedData(sdata);
 * // remplir le sdata->buffer(0 à partir de votre source
 * ...
 * // assigner la valeur à \a data
 * data->assignSerializedData(sd);
 * \endcode
 */
class ARCANE_CORE_EXPORT ISerializedData
{
  ARCCORE_DECLARE_REFERENCE_COUNTED_INCLASS_METHODS();

 public:
  
  //! Libère les ressources
  virtual ~ISerializedData() = default;

 public:

  //! Type de la donnée
  virtual eDataType baseDataType() const =0;
  
  //! Dimension. 0 pour un scalaire, 1 pour un tableau mono-dim, ...
  virtual Integer nbDimension() const =0;

  //! Nombre d'éléments
  virtual Int64 nbElement() const =0;

  //! Nombre d'éléments du type de base
  virtual Int64 nbBaseElement() const =0;

  //! Indique s'il s'agit d'un tableau multi-taille. (pertinent uniquement si nbDimension()>1)
  virtual bool isMultiSize() const =0;

  //! Indique le nombre d'octets qu'il faut allouer pour stocker ou lire les données
  virtual Int64 memorySize() const =0;

  //! Tableau contenant le nombre d'éléments pour chaque dimension
  virtual Int64ConstArrayView extents() const =0;

  /*!
   * \brief Valeurs sérialisées.
   * \deprecated Utiliser bytes() à la place.
   */
  ARCANE_DEPRECATED_2018_R("Use method 'bytes() const' instead")
  virtual ByteConstArrayView buffer() const =0;

  /*!
   * \brief Valeurs sérialisées.
   * \deprecated Utiliser bytes() à la place.
   */
  ARCANE_DEPRECATED_2018_R("Use method 'bytes()' instead")
  virtual ByteArrayView buffer() =0;

  //! Valeurs sérialisées
  virtual Span<const Byte> bytes() const =0;

  //! Valeurs sérialisées
  virtual Span<Byte> bytes() =0;

  /*!
   * \brief Positionne les valeurs de sérialisation.
   *
   * Le tableau \a buffer ne doit pas être modifié
   * tant que cette instance est utilisée.
   * \deprecated Utiliser setBytes() à la place.
   */
  virtual void setBuffer(ByteArrayView buffer) =0;

  /*!
   * \brief Positionne les valeurs de sérialisation.
   *
   * Le tableau \a buffer ne doit pas être modifié
   * tant que cette instance est utilisée.
   * \deprecated Utiliser setBytes() à la place.
   */
  virtual void setBuffer(ByteConstArrayView buffer) =0;

  /*!
   * \brief Positionne les valeurs de sérialisation.
   *
   * Le tableau \a bytes ne doit pas être modifié
   * tant que cette instance est utilisée.
   */
  virtual void setBytes(Span<Byte> bytes) =0;

  /*!
   * \brief Positionne les valeurs de sérialisation.
   *
   * Le tableau \a bytes ne doit pas être modifié
   * tant que cette instance est utilisée.
   */
  virtual void setBytes(Span<const Byte> bytes) =0;

  /*!
   * \brief Alloue un tableaux pour contenir les éléments sérialisés.
   *
   */
  virtual void allocateMemory(Int64 size) =0;

 public:

  /*!
   * \brief Serialize en lecture ou écriture la donnée
   */
  virtual void serialize(ISerializer* buffer) =0;

  /*!
   * \brief Serialize en lecture la donnée
   */
  virtual void serialize(ISerializer* buffer) const =0;

 public:

  /*!
   * \brief Calcul une clé de hashage sur cette donnée.
   *
   * La clé est ajoutée dans \a output. La longueur de la clé dépend
   * de l'algorithme utilisé.
   */
  virtual void computeHash(IHashAlgorithm* algo,ByteArray& output) const =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

