// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IDataFactoryMng.h                                           (C) 2000-2021 */
/*                                                                           */
/* Interface du gestionnaire de fabrique de données.                         */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IDATAFACTORYMNG_H
#define ARCANE_IDATAFACTORYMNG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"
#include "arcane/core/Parallel.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class IDataFactory;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Interface du gestionnaire de fabrique d'une donnée.
 *
 * \note Cette interface remplace 'IDataFactory'
 *
 * Cette interface permet d'enregistrer des fabriques pour créér des 'IData'
 * (via registerDataStorageFactory()) et de construire la bonne instance de
 * IData via l'appel à createSimpleDataRef().
 */
class IDataFactoryMng
{
 public:
  
  virtual ~IDataFactoryMng() = default;

 public:

  //! Construit l'instance
  virtual void build() =0;

  //! Gestionnaire de trace associé
  virtual ITraceMng* traceMng() const =0;

  //! Enregistre la fabrique \a factory.
  virtual void registerDataStorageFactory(Ref<IDataStorageFactory> factory) =0;

  /*
   * \brief Créé une donnée.
   *
   * La fabrique utilisée pour créér la donnée est issu d'une instance de DataStorageTypeInfo via
   * l'utilisation de la méthode DataStorageTypeInfo::fullName().
   */
  virtual Ref<IData>
  createSimpleDataRef(const String& storage_type,const DataStorageBuildInfo& build_info) =0;

  /*!
   * \brief Créé une opération effectuant une réduction de type \a rt.
   */
  virtual IDataOperation* createDataOperation(Parallel::eReduceType rt) =0;

  /*!
   * \brief Créé des données sérialisées.
   *
   * les tableaux \a dimensions et \a values ne sont pas dupliqués et ne doivent
   * pas être modifiés tant que l'objet sérialisé est utilisé.
   *
   * Le type \a data_type doit être un type parmi \a DT_Byte, \a DT_Int16, \a DT_Int32,
   * \a DT_Int64 ou DT_Real.
   *
   * \deprecated Utiliser arcaneCreateSerializedDataRef() à la place
   */
  ARCCORE_DEPRECATED_2021("Use global method arcaneCreateSerializedDataRef() instead")
  virtual Ref<ISerializedData>
  createSerializedDataRef(eDataType data_type,Int64 memory_size,
                          Integer nb_dim,Int64 nb_element,
                          Int64 nb_base_element,bool is_multi_size,
                          Int64ConstArrayView dimensions) =0;

  /*!
   * \brief Créé des données sérialisées.
   *
   * la donnée sérialisée est vide. Elle ne pourra être utilisée qu'après un
   * appel à ISerializedData::serialize() en mode ISerializer::ModePut.
   *
   * \deprecated Utiliser arcaneCreateEmptySerializedDataRef() à la place.
   */
  ARCCORE_DEPRECATED_2021("Use global method arcaneCreateEmptySerializedDataRef() instead")
  virtual Ref<ISerializedData> createEmptySerializedDataRef() =0;

  //! Récupère l'ancienne fabrique (obsolète)
  ARCCORE_DEPRECATED_2021("Do not use deprecated interface 'IDataFactory'")
  virtual IDataFactory* deprecatedOldFactory() const =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
