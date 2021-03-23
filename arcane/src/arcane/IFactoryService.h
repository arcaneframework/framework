// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IFactoryService.h                                           (C) 2000-2006 */
/*                                                                           */
/* Interface d'un service de fabrique.                                       */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IFACTORYSERVICE_H
#define ARCANE_IFACTORYSERVICE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/IService.h"
#include "arcane/ServiceBuildInfo.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IService;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Interface d'une fabrique sur une classe implémentant \a InterfaceType.
 */
template<typename InterfaceType>
class IFactoryServiceT
: public IService
{
 protected:
  IFactoryServiceT() {}
 public:

  //! Libère les ressources
  virtual ~IFactoryServiceT() {}

 public:

  /*! \brief Créé une instance.
   *
   * L'objet retourné est garanti ne pas être nul.
   */  
  virtual InterfaceType* createInstance() =0;

  /*! \brief Créé une instance pour le maillage \a mesh.
   *
   * Seuls les service de sous-domaine supporte ce type de création.
   *
   * L'objet retourné est garanti ne pas être nul.
   */  
  virtual InterfaceType* createInstance(IMesh* mesh) =0;

  /*! \brief Créé une instance singleton.
   *
   * L'objet retourné est garanti ne pas être nul. L'instance retournée
   * est toujours la même.
   */  
  virtual InterfaceType* singletonInstance() =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

