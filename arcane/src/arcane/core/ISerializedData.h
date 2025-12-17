// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ISerializedData.h                                           (C) 2000-2025 */
/*                                                                           */
/* Interface d'une donnée sérialisée.                                        */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ISERIALIZEDDATA_H
#define ARCANE_CORE_ISERIALIZEDDATA_H
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
 * \internal
 * \brief Interface d'une donnée sérialisée.
 *
 * Une donnée (IData) est sérialisée en une instance de cette classe.
 *
 * Quel que soit le type de la donnée, le type sérialisé est obligatoirement
 * un type de base parmi les suivants: DT_Byte, DT_Int16, DT_Int32, DT_Int64, DT_Real.
 *
 * Une instance de cette classe n'est valable que tant que la donnée
 * de référence n'est pas modifiée.
 * 
 * Pour sérialiser une donnée \a data en écriture:
 * \code
 * IData* data = ...;
 * ISerializedData* sdata = data->createSerializedData();
 * // sdata->constBytes() contient la donnée sérialisée.
 * Span<const Byte> buf(sdata->constBytes());
 * std::cout.write(reinterpret_cast<const char*>(buf.data()),buf.size());
 * \endcode
 *
 * Pour sérialiser une donnée \a data en lecture:
 * \code
 * IData* data = ...
 * // Créé une instance d'un ISerializedData.
 * Ref<ISerializedData> sdata = arcaneCreateSerializedDataRef(...);
 * data->allocateBufferForSerializedData(sdata);
 * // Remplit sdata->writableBytes() à partir de votre source
 * Span<Byte> buf(sdata->writableBytes());
 * std::cin.read(reinterpret_cast<char*>(buf.data()),buf.size());
 * // Assigne la valeur à \a data
 * data->assignSerializedData(sdata);
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
  virtual eDataType baseDataType() const = 0;

  //! Dimension. 0 pour un scalaire, 1 pour un tableau mono-dim, ...
  virtual Integer nbDimension() const = 0;

  //! Nombre d'éléments
  virtual Int64 nbElement() const = 0;

  //! Nombre d'éléments du type de base
  virtual Int64 nbBaseElement() const = 0;

  //! Indique s'il s'agit d'un tableau multi-taille. (pertinent uniquement si nbDimension()>1)
  virtual bool isMultiSize() const = 0;

  //! Indique le nombre d'octets qu'il faut allouer pour stocker ou lire les données
  virtual Int64 memorySize() const = 0;

  //! Tableau contenant le nombre d'éléments pour chaque dimension
  virtual Int64ConstArrayView extents() const = 0;

  //! Forme du tableau associé aux données
  virtual ArrayShape shape() const = 0;

  //! Valeurs sérialisées.
  virtual Span<const Byte> constBytes() const = 0;

  /*!
   * \brief Vue sur les valeurs sérialisées
   *
   * \warning Cette méthode renvoie une vue non vide uniquement si on
   * a appelé allocateMemory() ou setWritableBytes(Span<Byte>) avant.
   */
  virtual Span<Byte> writableBytes() = 0;

  /*!
   * \brief Positionne les valeurs de sérialisation.
   *
   * La vue \a bytes doit rester valide tant que cette instance est utilisée.
   */
  virtual void setWritableBytes(Span<Byte> bytes) = 0;

  /*!
   * \brief Positionne les valeurs de sérialisation pour la lecture
   *
   * La vue \a bytes doit rester valide tant que cette instance est utilisée.
   */
  virtual void setConstBytes(Span<const Byte> bytes) = 0;

  /*!
   * \brief Alloue un tableaux pour contenir les éléments sérialisés.
   *
   * Après appel à cette méthode, il est possible de récupérer une
   * vue sur les valeurs sérialisées via writableBytes() ou constBytes().
   */
  virtual void allocateMemory(Int64 size) = 0;

 public:

  /*!
   * \brief Serialize en lecture ou écriture la donnée
   */
  virtual void serialize(ISerializer* buffer) = 0;

  /*!
   * \brief Serialize en lecture la donnée
   */
  virtual void serialize(ISerializer* buffer) const = 0;

 public:

  /*!
   * \brief Calcul une clé de hashage sur cette donnée.
   *
   * La clé est ajoutée dans \a output. La longueur de la clé dépend
   * de l'algorithme utilisé.
   */
  virtual void computeHash(IHashAlgorithm* algo, ByteArray& output) const = 0;

 public:

  /*!
   * \brief Valeurs sérialisées.
   * \deprecated Utiliser bytes() à la place.
   */
  ARCANE_DEPRECATED_2018_R("Use method 'writableBytes()' or 'constBytes()' instead")
  virtual ByteConstArrayView buffer() const = 0;

  /*!
   * \brief Valeurs sérialisées.
   * \deprecated Utiliser bytes() à la place.
   */
  ARCANE_DEPRECATED_2018_R("Use method 'writableBytes()' or 'constBytes()' instead")
  virtual ByteArrayView buffer() = 0;

  //! Valeurs sérialisées.
  ARCCORE_DEPRECATED_2021("Use method 'writableBytes()' or 'constBytes()' instead")
  virtual Span<const Byte> bytes() const = 0;

  /*!
   * \brief Positionne les valeurs de sérialisation.
   *
   * Le tableau \a buffer ne doit pas être modifié
   * tant que cette instance est utilisée.
   * \deprecated Utiliser setBytes() à la place.
   */
  ARCCORE_DEPRECATED_2021("Use method 'setWritableBytes()' instead")
  virtual void setBuffer(ByteArrayView buffer) = 0;

  /*!
   * \brief Positionne les valeurs de sérialisation.
   *
   * Le tableau \a buffer ne doit pas être modifié
   * tant que cette instance est utilisée.
   * \deprecated Utiliser setBytes() à la place.
   */
  ARCCORE_DEPRECATED_2021("Use method 'setConstBytes()' instead")
  virtual void setBuffer(ByteConstArrayView buffer) = 0;

  /*!
   * \brief Positionne les valeurs de sérialisation.
   *
   * Le tableau \a bytes ne doit pas être modifié
   * tant que cette instance est utilisée.
   */
  ARCCORE_DEPRECATED_2021("Use method 'setWritableBytes()' instead")
  virtual void setBytes(Span<Byte> bytes) = 0;

  /*!
   * \brief Positionne les valeurs de sérialisation.
   *
   * Le tableau \a bytes ne doit pas être modifié
   * tant que cette instance est utilisée.
   */
  ARCCORE_DEPRECATED_2021("Use method 'setConstBytes()' instead")
  virtual void setBytes(Span<const Byte> bytes) = 0;

  /*!
   * \brief Valeurs sérialisées
   *
   * \warning Cette méthode renvoie une vue non vide uniquement si on
   * a appelé setBytes(Span<Byte>) ou allocateMemory().
   */
  ARCCORE_DEPRECATED_2021("Use method 'writableBytes()' or 'constBytes()' instead")
  virtual Span<Byte> bytes() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Créé des données sérialisées.
 *
 * les tableaux \a dimensions et \a values ne sont pas dupliqués et ne doivent
 * pas être modifiés tant que l'objet sérialisé est utilisé.
 *
 * Le type \a data_type doit être un type parmi \a DT_Byte, \a DT_Int16, \a DT_Int32,
 * \a DT_Int64 ou DT_Real.
 */
extern "C++" ARCANE_CORE_EXPORT
Ref<ISerializedData>
arcaneCreateSerializedDataRef(eDataType data_type, Int64 memory_size,
                              Integer nb_dim, Int64 nb_element, Int64 nb_base_element,
                              bool is_multi_size, Int64ConstArrayView dimensions);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Créé des données sérialisées.
 *
 * les tableaux \a dimensions et \a values ne sont pas dupliqués et ne doivent
 * pas être modifiés tant que l'objet sérialisé est utilisé.
 *
 * Le type \a data_type doit être un type parmi \a DT_Byte, \a DT_Int16, \a DT_Int32,
 * \a DT_Int64 ou DT_Real.
 */
extern "C++" ARCANE_CORE_EXPORT
Ref<ISerializedData>
arcaneCreateSerializedDataRef(eDataType data_type, Int64 memory_size,
                              Integer nb_dim, Int64 nb_element, Int64 nb_base_element,
                              bool is_multi_size, Int64ConstArrayView dimensions,
                              const ArrayShape& shape);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Créé des données sérialisées.
 *
 * la donnée sérialisée est vide. Elle ne pourra être utilisée qu'après un
 * appel à ISerializedData::serialize() en mode ISerializer::ModePut.
 */
extern "C++" ARCANE_CORE_EXPORT
Ref<ISerializedData>
arcaneCreateEmptySerializedDataRef();

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

