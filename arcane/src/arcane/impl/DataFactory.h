// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DataFactory.h                                               (C) 2000-2021 */
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

  Ref<IData> createSimpleDataRef(eDataType data_type,Integer dimension,Integer multi_tag) override;

  IDataOperation* createDataOperation(Parallel::eReduceType rt) override;

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

