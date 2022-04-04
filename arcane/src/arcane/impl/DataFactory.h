// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
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
 public:
  
  DataFactory(IApplication* sm);
  virtual ~DataFactory();

 public:

  void build() override;

  IApplication* application() override { return m_application; }

  IDataOperation* createDataOperation(Parallel::eReduceType rt) override;

 private:

  IApplication* m_application;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

