// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DataFactory.h                                               (C) 2000-2020 */
/*                                                                           */
/* Fabrique de donnée.                                                       */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IMPL_DATAFACTORY_H
#define ARCANE_IMPL_DATAFACTORY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/List.h"
#include "arcane/IDataFactory.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface d'une fabrique d'une donnée.
 */
class DataFactory
: public IDataFactory
{
  typedef List<IData*> DataArray;

 public:
  
  DataFactory(IApplication* sm);
  virtual ~DataFactory();

 public:

  void build() override;

  IApplication* application() override { return m_application; }
  /*!
   * \brief Enregistre dans la fabrique la donnée \a data.
   * \warning L'implémentation actuelle ne permet d'enregistrer
   * que les données simples.
   */
  IData* registerData(IData* data) override;

  IData* createSimpleData(eDataType data_type,Integer dimension,Integer multi_tag) override;
  
  Ref<IData> createSimpleDataRef(eDataType data_type,Integer dimension,Integer multi_tag) override;

  IDataOperation* createDataOperation(Parallel::eReduceType rt) override;

  ARCCORE_DEPRECATED_2019("Use overload with extents")
  ISerializedData* createSerializedData(eDataType data_type,Integer memory_size,
                                        Integer nb_dim,Integer nb_element,
                                        Integer nb_base_element,bool is_multi_size,
                                        IntegerConstArrayView dimensions) override;

  ISerializedData* createSerializedData(eDataType data_type,Int64 memory_size,
                                        Integer nb_dim,Int64 nb_element,
                                        Int64 nb_base_element,bool is_multi_size,
                                        Int64ConstArrayView extents) override;

  ISerializedData* createSerializedData() override;

 private:

  IApplication* m_application;
  DataArray m_data;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

