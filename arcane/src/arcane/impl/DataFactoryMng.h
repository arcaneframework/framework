// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DataFactoryMng.h                                            (C) 2000-2021 */
/*                                                                           */
/* Gestionnaire de fabriques de données.                                     */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IMPL_DATAFACTORYMNG_H
#define ARCANE_IMPL_DATAFACTORYMNG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"

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
 * \internal
 * \brief Gestionnaire de fabrique d'une donnée.
 *
 * Les fabriques s'enregistrent via la méthode registerDataStorageFactory().
 *
 * TODO: supprimer l'utilisation de 'IApplication' lorqu'on n'aura plus
 * besoin de 'm_old_factory'.
 */
class DataFactoryMng
: public TraceAccessor
, public IDataFactoryMng
{
 public:
  
  DataFactoryMng(IApplication* sm);
  ~DataFactoryMng() override;

 public:

  void build() override;
  IApplication* application() { return m_application; }
  ITraceMng* traceMng() const override;
  void registerDataStorageFactory(Ref<IDataStorageFactory> factory) override;
  Ref<IData> createSimpleDataRef(const String& storage_type,const DataStorageBuildInfo& build_info) override;
  IDataOperation* createDataOperation(Parallel::eReduceType rt) override;
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

