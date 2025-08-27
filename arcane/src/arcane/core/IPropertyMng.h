// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IPropertyMng.h                                              (C) 2000-2025 */
/*                                                                           */
/* Interface du gestionnaire des propriétés.                                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IPROPERTYMNG_H
#define ARCANE_CORE_IPROPERTYMNG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class Properties;
class PropertiesImpl;
class IObservable;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface du gestionnaire des propriétés.
 */
class IPropertyMng
{
 public:

  virtual ~IPropertyMng() {} //!< Libère les ressources.

 public:

  virtual ITraceMng* traceMng() const =0;

 public:

  /*!
   * \internal
   * \brief Récupère la liste de propriétés de nom complet \a full_name.
   *
   * Cette méthode ne doit être appelée que par la classe Properties.
   * Pour récupérer une instance, il faut utiliser le constructeur de Properties.
   */
  virtual PropertiesImpl* getPropertiesImpl(const String& full_name) =0;

  /*!
   * \internal
   * \brief Enregister les propriétés référencées par \a p.
   */
  virtual void registerProperties(const Properties& p) =0;

  //! Supprime les propriétés référencées par \a p
  virtual void destroyProperties(const Properties& p) =0;

  //! Effectue la sérialisation
  virtual void serialize(ISerializer* serializer) =0;

  //! Sérialise les informations de propriété dans \a bytes.
  virtual void writeTo(ByteArray& bytes) =0;

  /*!
   * \brief Relit les informations sérialisées contenues dans \a bytes.
   *
   * Le tableau \a bytes doit avoir été créé par un appel à writeTo().
   */
  virtual void readFrom(Span<const Byte> bytes) =0;

  //! Affiche les propriétés et leurs valeurs sur le flot \a o
  virtual void print(std::ostream& o) const =0;

  /*!
   * \brief Observable pour l'écriture.
   *
   * Les observateurs enregistrés dans cet observable sont appelés
   * au début de writeTo().
   */
  virtual IObservable* writeObservable() =0;

  /*!
   * \brief Observable pour la lecture.
   *
   * Les observateurs enregistrés dans cet observable sont appelés
   * à la fin de readFrom().
   */
  virtual IObservable* readObservable() =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

