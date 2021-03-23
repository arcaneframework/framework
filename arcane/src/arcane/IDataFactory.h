// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IDataFactory.h                                              (C) 2000-2020 */
/*                                                                           */
/* Interface d'une fabrique de donnée.                                       */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IDATAFACTORY_H
#define ARCANE_IDATAFACTORY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/ArcaneTypes.h"
#include "arcane/Parallel.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Interface d'une fabrique d'une donnée.
 * \todo Renommer en 'IDataFactoryMng'.
 * \warning Cette classe est obsolète et ne doit plus être utilisée en
 * dehors de Arcane.
 */
class IDataFactory
{
 public:
  
  virtual ~IDataFactory() = default;

 public:

  //! Construit l'instance
  virtual void build() =0;

  //! Application
  virtual IApplication* application() =0;

  /*!
   * \brief Enregistre dans la fabrique la donnée \a data.
   * La donnée \a data enregistrée servira à créer des données
   * de même type (par appel à IData::clone()) lors de l'appel
   * à la méthode createSimpleData().
   * Si une donnée de même type est déjà enregistrée, la
   * nouvelle remplace l'ancienne qui est retournée.
   * Les données enregistrées deviennent la propriété de cette
   * instance et sont libérées lors de la desctruction par l'appel
   * à l'opérateur delete.
   * \return l'ancienne donnée correspondante ou 0 si aucune.
   */
  virtual IData* registerData(IData* data) =0;

  /*
   * \brief Créé une donnée d'un type simple.
   * \param data_type Type de la donnée à créer
   * \param dimension Dimension de la donnée
   * \return la donnée ou zéro si aucune donnée de ce type ne peut
   * être fabriquée.
   */
  ARCCORE_DEPRECATED_2020("Use createSimpleDataRef() instead")
  virtual IData* createSimpleData(eDataType data_type,Integer dimension,Integer multi_tag) =0;

  /*
   * \brief Créé une donnée d'un type simple.
   * \param data_type Type de la donnée à créer
   * \param dimension Dimension de la donnée
   * \return la donnée ou zéro si aucune donnée de ce type ne peut
   * être fabriquée.
   */
  virtual Ref<IData> createSimpleDataRef(eDataType data_type,Integer dimension,Integer multi_tag) =0;

  /*!
   * \brief Créé une opération effectuant une réduction de type \a rt.
   * \todo mettre dans une autre interface.
   */
  virtual IDataOperation* createDataOperation(Parallel::eReduceType rt) =0;

  /*!
   * \brief Créé des données sérialisées.
   *
   * les tableaux \a dimensions et \a values ne sont pas dupliqués et ne doivent
   * pas être modifiés tant que l'objet sérialisé est utilisé.
   *
   * Le type \a data_type doit être un type parmi \a DT_Byte, \a DT_Int32,
   * \a DT_Int64 ou DT_Real.
   *
   * \deprecated Utiliser la version 64 bits de createSerializedData().
   */
  ARCCORE_DEPRECATED_2020("Use IDataFactoryMng::createSimpleDataRef() instead")
  virtual ISerializedData*
  createSerializedData(eDataType data_type,Integer memory_size,
                       Integer nb_dim,Integer nb_element,
                       Integer nb_base_element,bool is_multi_size,
                       IntegerConstArrayView dimensions) =0;

  /*!
   * \brief Créé des données sérialisées.
   *
   * les tableaux \a dimensions et \a values ne sont pas dupliqués et ne doivent
   * pas être modifiés tant que l'objet sérialisé est utilisé.
   *
   * Le type \a data_type doit être un type parmi \a DT_Byte, \a DT_Int32,
   * \a DT_Int64 ou DT_Real.
   */
   ARCCORE_DEPRECATED_2020("Use IDataFactoryMng::createSimpleDataRef() instead")
   virtual ISerializedData* createSerializedData(eDataType data_type,Int64 memory_size,
                                                 Integer nb_dim,Int64 nb_element,
                                                 Int64 nb_base_element,bool is_multi_size,
                                                 Int64ConstArrayView dimensions) =0;

  /*!
   * \brief Créé des données sérialisées.
   *
   * la donnée sérialisée est vide. Elle ne pourra être utilisée qu'après un
   * appel à ISerializedData::serialize() en mode ISerializer::ModePut.
   */
  ARCCORE_DEPRECATED_2020("Use IDataFactoryMng::createEmptySimpleDataRef() instead")
  virtual ISerializedData* createSerializedData() =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
