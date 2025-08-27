// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IDataFactory.h                                              (C) 2000-2021 */
/*                                                                           */
/* Interface d'une fabrique de donnée.                                       */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IDATAFACTORY_H
#define ARCANE_IDATAFACTORY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"
#include "arcane/core/Parallel.h"

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
   * \brief Créé une opération effectuant une réduction de type \a rt.
   * \todo mettre dans une autre interface.
   */
  ARCCORE_DEPRECATED_2021("Do not use deprecated interface 'IDataFactory'. Use 'IDataFactoryMng' instead")
  virtual IDataOperation* createDataOperation(Parallel::eReduceType rt) =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
