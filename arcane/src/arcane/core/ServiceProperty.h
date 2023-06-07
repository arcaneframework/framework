// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ServiceProperty.h                                           (C) 2000-2022 */
/*                                                                           */
/* Propriétés d'un service.                                                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_SERVICEPROPERTY_H
#define ARCANE_SERVICEPROPERTY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcaneGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Propriétés pour les fabriques des service.
 *
 * Il s'agit de drapeaux qui s'utilisent avec l'opérateur ou binaire (|)
 */
enum eServiceFactoryProperties
{
  //! Aucune propriété particulière
  SFP_None = 0,
  //! Indique que le service est singleton
  SFP_Singleton = 1,
  //! Indique que le service se charge automatiquement.
  SFP_Autoload = 2
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Type du service.
 *
 * Cette énumération permet de connaitre à quel endroit peut
 * être créé un service.
 *
 * Il s'agit de drapeaux qui s'utilisent avec l'opérateur ou binaire (|).
 * Un service peut donc être disponible à plusieurs endroits. Par exemple,
 * il peut être présent comme option du jeu de données (#ST_CaseOption) et aussi
 * au niveau du sous-domaine (#ST_SubDomain). Dans ce dernier cas,
 * il peut être créé via la classe ServiceBuilder.
 *
 * \note Ce type doit correspondre avec le type C# correspondant
 */
enum eServiceType
{
  ST_None = 0,
  //! Le service s'utilise au niveau de l'application
  ST_Application = 1,
  //! Le service s'utilise au niveau de la session
  ST_Session = 2,
  //! Le service s'utilise au niveau du sous-domaine
  ST_SubDomain = 4,
  //! Le service s'utilise au niveau du jeu de données.
  ST_CaseOption = 8,
  // NOTE: Cette valeur n'est pas encore utilisée.
  //! Le service s'utilise avec un maillage spécifié explicitement.
  ST_Mesh = 16
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Propriétés de création d'un service.
 *
 * Cette classe est utilisée dans les macros d'enregistrement des services
 * et peut donc être instantiée en tant que variable globale avant d'entrer
 * dans le main() du code. Elle ne doit donc contenir que des champs de type
 * Plain Object Data (POD).
 *
 * En général, les instances de cette classe sont utilisés lors
 * de l'enregistrement d'un service via la macro ARCANE_REGISTER_SERVICE().
 *
 * Dans le constructeur, les paramètres \a type et \a properties
 * peuvent utiliser une combinaison de valeur énumérées. Par exemple,
 * pour spécifier un service pouvant être utilisé à la fois dans le
 * jeu de données et au niveau du sous-domaine, on peut faire comme suit:
 *
 * \code
 * ServiceProperty("ServiceName",ST_SubDomain|ST_CaseOption);
 * \endcode
 */
class ARCANE_CORE_EXPORT ServiceProperty
{
 public:

  /*!
   * \brief Construit une instance pour un service de nom \a aname  et de type \a atype
   * avec les propriétés \a properties.
   */
  ServiceProperty(const char* aname, int atype, eServiceFactoryProperties aproperties) ARCANE_NOEXCEPT
  : m_name(aname)
  , m_type(atype)
  , m_properties(aproperties)
  {
  }

  //! Construit une instance pour un service de nom \a aname et de type \a atype
  ServiceProperty(const char* aname, int atype) ARCANE_NOEXCEPT
  : m_name(aname)
  , m_type(atype)
  , m_properties(SFP_None)
  {
  }

  //! Construit une instance pour un service de nom \a aname et de type \a atype
  ServiceProperty(const char* aname, eServiceType atype) ARCANE_NOEXCEPT
  : m_name(aname)
  , m_type((int)atype)
  , m_properties(SFP_None)
  {
  }

 public:

  //! Nom du service.
  const char* name() const { return m_name; }

  //! Type du service (combinaison de eServiceType)
  int type() const { return m_type; }

  //! Propriétés du service (combinaison de eServiceFactoryProperties)
  eServiceFactoryProperties properties() const { return m_properties; }

 private:

  const char* m_name;
  int m_type;
  eServiceFactoryProperties m_properties;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
