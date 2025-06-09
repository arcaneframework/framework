// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IFactoryService.h                                           (C) 2000-2025 */
/*                                                                           */
/* Interface d'un service de fabrique.                                       */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IFACTORYSERVICE_H
#define ARCANE_CORE_IFACTORYSERVICE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/IService.h"
#include "arcane/core/ServiceBuildInfo.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Interface d'une fabrique sur une classe implémentant \a InterfaceType.
 */
template <typename InterfaceType>
class IFactoryServiceT
: public IService
{
 protected:

  IFactoryServiceT() = default;

 public:

  /*!
   * \brief Créé une instance.
   *
   * L'objet retourné est garanti ne pas être nul.
   */
  virtual InterfaceType* createInstance() = 0;

  /*!
   * \brief Créé une instance pour le maillage \a mesh.
   *
   * Seuls les service de sous-domaine supporte ce type de création.
   *
   * L'objet retourné est garanti ne pas être nul.
   */
  virtual InterfaceType* createInstance(IMesh* mesh) = 0;

  /*! \brief Créé une instance singleton.
   *
   * L'objet retourné est garanti ne pas être nul. L'instance retournée
   * est toujours la même.
   */
  virtual InterfaceType* singletonInstance() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

