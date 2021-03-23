// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Configuration.h                                             (C) 2000-2020 */
/*                                                                           */
/* Gestion des options de configuration de l'exécution.                      */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CONFIGURATION_H
#define ARCANE_CONFIGURATION_H
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
 * \brief Interface d'une section de configuration.
 *
 * Cette interface permet de récupérer les valeurs d'une option
 * de configuration.
 */
class ARCANE_CORE_EXPORT IConfigurationSection
{
 public:

  virtual ~IConfigurationSection() {} //!< Libère les ressources

 public:

  virtual Int32 value(const String& name,Int32 default_value) const =0;
  virtual Int64 value(const String& name,Int64 default_value) const =0;
  virtual Real value(const String& name,Real default_value) const =0;
  virtual bool value(const String& name,bool default_value) const =0;
  virtual String value(const String& name,const String& default_value) const =0;
  virtual String value(const String& name,const char* default_value) const =0;

  virtual Integer valueAsInteger(const String& name,Integer default_value) const =0;
  virtual Int32 valueAsInt32(const String& name,Int32 default_value) const =0;
  virtual Int64 valueAsInt64(const String& name,Int64 default_value) const =0;
  virtual Real valueAsReal(const String& name,Real default_value) const =0;
  virtual bool valueAsBool(const String& name,bool default_value) const =0;
  virtual String valueAsString(const String& name,const String& default_value) const =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface d'une configuration.
 */
class ARCANE_CORE_EXPORT IConfiguration
{
 public:

  virtual ~IConfiguration() {} //!< Libère les ressources

 public:

  /*!
   * \brief Créé une section de configuration.
   *
   * L'instance retournée doit être détruire par l'opérateur delete.
   */
  virtual IConfigurationSection* createSection(const String& name) const =0;

  /*!
   * \brief Section principale.
   *
   * L'instance retournée reste la propriété de cette instance
   * et ne doit pas être détruite.
   */
  virtual IConfigurationSection* mainSection() const =0;

  /*!
   * \brief Ajout une valeur à la configuration.
   *
   * Ajoute à la configuration la valeur \a value pour le
   * nom \a name. La nouvelle valeur aura comme priority \a priority. Si
   * une valeur de nom \a name existe déjà, elle est remplacée par
   * \a value si \a priority est inférieure à la priorité actuelle.
   */
  virtual void addValue(const String& name,const String& value,Integer priority) =0;
  
  /*!
   * \brief Clone cette configuration.
   */
  virtual IConfiguration* clone() const =0;

  /*!
   * \brief Fusionne cette configuration avec la configuration \a c.
   *
   * Si une option existe à la fois dans cette configuration et dans \a c,
   * c'est celle qui a la priorité la plus faible qui est conservée.
   */
  virtual void merge(const IConfiguration* c) =0;

  //! Affiche les valeurs des paramètres de configuration via le traceMng()
  virtual void dump() const =0;

  //! Affiche les valeurs des paramètres de configuration sur le flot o
  virtual void dump(std::ostream& ostr) const =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Gestionnaire de configuration.
 */
class ARCANE_CORE_EXPORT IConfigurationMng
{
 public:

  virtual ~IConfigurationMng() {} //!< Libère les ressources

 public:

  //! Configuration par défaut.
  virtual IConfiguration* defaultConfiguration() const =0;

  /*!
   * \brief Créé une nouvelle configuration.
   *
   * L'instance retournée doit être détruite par l'opérateur delete.
   */
  virtual IConfiguration* createConfiguration() =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
