// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ConfigurationReader.h                                       (C) 2000-2020 */
/*                                                                           */
/* Lecteurs de fichiers de configuration.                                    */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_IMPL_CONFIGURATIONREADER_H
#define ARCANE_IMPL_CONFIGURATIONREADER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class XmlNode;
class IConfiguration;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Lecteurs de fichiers de configuration.
 */
class ARCANE_IMPL_EXPORT ConfigurationReader
: public TraceAccessor
{
 public:

  enum Priority
  {
    P_CaseDocument = 10,
    P_TimeLoop = 50,
    P_Global = 100,
    P_GlobalRuntime = 110
  };

 public:

  ConfigurationReader(ITraceMng* tm,IConfiguration* config)
  : TraceAccessor(tm), m_configuration(config) {}
  /*!
   * \brief Ajoute des valeurs à la configuration.
   *
   * Ajoute à la configuration les valeurs contenus dans les
   * les éléments fils de \a element. Les éléments pris en compte
   * sont les éléments de la forme suivante:
   * - <add name="ConfigName" value="ConfigValue" />
   *
   * La méthode IConfiguration::addValue() est appelé pour chaque valeur
   * avec la priorité \a priority.
   */
  void addValuesFromXmlNode(const XmlNode& element,Integer priority);

  /*!
   * \brief Ajoute des valeurs à la configuration.
   *
   * Ajoute à la configuration les valeurs contenus dans les
   * champs fils de \a jv.
   *
   * Les éléments tableaux ne sont pas pris en compte. L'appel
   * est récursif et les enfants de \a jv sont pris en compte. Cela
   * est équivalent à faire une sous-section dans la configuration.
   *
   * La méthode IConfiguration::addValue() est appelé pour chaque valeur
   * avec la priorité \a priority.
   */
  void addValuesFromJSON(const JSONValue& jv,Integer priority);

 private:

  IConfiguration* m_configuration;

 private:

  void _addValuesFromJSON(const JSONValue& jv,Integer priority,const String& base_name);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
