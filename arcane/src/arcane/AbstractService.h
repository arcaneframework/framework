// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AbstractService.h                                           (C) 2000-2017 */
/*                                                                           */
/* Classe de base d'un service.                                              */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ABSTRACTSERVICE_H
#define ARCANE_ABSTRACTSERVICE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"
#include "arcane/IService.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ServiceBuildInfo;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe de base d'un service.
 *
 * Cette classe est LA classe d'implémentation bas niveau de l'interface \a IService.
 *
 * \ingroup Service
 */
class ARCANE_CORE_EXPORT AbstractService
: public TraceAccessor
, public IService
{
 protected:

  //! Constructeur à partir d'un \a ServiceBuildInfo
  AbstractService(const ServiceBuildInfo&);

 public:
	
  //! Destructeur
  virtual ~AbstractService();

 public:

  //! Construction de niveau \a build du service
  /*! L'appel à \a build est effectué au moment de sa construction,
   *  usuellement au niveau de sa lecture dans les options en phase1.
   */
  virtual void build() {}

 public:

  //! Accès aux informations du service
  /*! Voir \a IServiceInfo pour les détails */
  virtual IServiceInfo* serviceInfo() const { return m_service_info; }
  
  //! Accès à l'interface de base des principaux objets Arcane
  virtual IBase* serviceParent() const { return m_parent; }

  //! Retourne l'interface bas niveau \a IService du service
  virtual IService* serviceInterface() { return this; }

 private:

  IServiceInfo* m_service_info;
  IBase* m_parent;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

