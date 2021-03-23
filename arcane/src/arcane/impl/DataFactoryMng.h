// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DataFactoryMng.h                                            (C) 2000-2020 */
/*                                                                           */
/* Gestionnaire de fabriques de données.                                     */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IMPL_DATAFACTORYMNG_H
#define ARCANE_IMPL_DATAFACTORYMNG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/IDataFactoryMng.h"

#include <map>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

class DataFactory;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Gestionnaire de fabrique d'une donnée.
 */
class DataFactoryMng
: public IDataFactoryMng
{
 public:
  
  DataFactoryMng(IApplication* sm);
  ~DataFactoryMng() override;

 public:

  void build() override;
  IApplication* application() override { return m_application; }
  void registerDataStorageFactory(Ref<IDataStorageFactory> factory) override;
  Ref<IData> createSimpleDataRef(const String& storage_type,const DataStorageBuildInfo& build_info) override;
  Ref<ISerializedData>
  createSerializedDataRef(eDataType data_type,Int64 memory_size,
                          Integer nb_dim,Int64 nb_element,
                          Int64 nb_base_element,bool is_multi_size,
                          Int64ConstArrayView extents) override;

  Ref<ISerializedData> createEmptySerializedDataRef() override;
  IDataFactory* deprecatedOldFactory() const override { return m_old_factory; }

 private:

  IApplication* m_application;
  IDataFactory* m_old_factory;
  std::map<String,Ref<IDataStorageFactory>> m_factories;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

