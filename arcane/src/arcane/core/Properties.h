// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Properties.h                                                (C) 2000-2022 */
/*                                                                           */
/* Liste de propriétés.                                                      */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_PROPERTIES_H
#define ARCANE_PROPERTIES_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/UtilsTypes.h"
#include "arcane/utils/AutoRef.h"
#include "arcane/core/SharedReference.h"
#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class PropertiesImpl;
class IPropertyMng;

class PropertiesImplBase
: public SharedReference
{
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Liste de propriétés.
 *
 * Cette classe gère une liste de propriétés. Une propriété est
 * caractérisée par un nom et une valeur d'un type donnée.
 * Le nom ne doit pas contenir le caractère '.' qui sert de délimiteur
 * pour les hiérarchies de propriétés.
 *
 * Les fonctions set*() permettent de positionner une propriété. Les
 * fonctions get*() permettent de récupérer la valeur.
 *
 * Pour les propriétés scalaires, il existe trois manière de récupérer une
 * valeur. Ces trois méthodes sont équivalentes sauf si la propriété n'a pas été positionnée.
 * - via une surcharge de la méthode get(). Dans le cas où la propriété n'est pas positionnée,
 * la valeur passée en argument est inchangée et la méthode retourne false.
 * - via un appel explicite (par exemple getBool()). Dans le cas où la propriété n'est pas positionnée,
 * c'est la valeur obtenue avec le constructeur par défaut pour le type concerné qui est utilisée.
 * - via un appel explicite avec valeur par défaut possible (par exemple getBoolWithDefault()).
 * Dans le cas où la propriété n'est pas positionnée, c'est la valeur par défaut passée en argument qui
 * est utilisée.
 *
 */
class ARCANE_CORE_EXPORT Properties
{
 public:
	
  //! Créé ou récupère une liste de propriétés de nom \a name
  Properties(IPropertyMng* pm,const String& name);

  //! Créé ou récupère une liste de propriétés de nom \a name et fille de \a parent_property 
  Properties(const Properties& parent_property,const String& name);

  //! Constructeur par recopie
  Properties(const Properties& rhs);
  //! Opérateur de recopie
  const Properties& operator=(const Properties& rhs);
  //! Détruit la référence à cette propriété
  virtual ~Properties();

 public:

  //! Positionne une propriété de type bool de nom \a name et de valeur \a value.
  void setBool(const String& name,bool value);

  //! Positionne une propriété de type bool de nom \a name et de valeur \a value.
  void set(const String& name,bool value);
  
  //! Valeur de la propriété de nom \a name.
  bool getBool(const String& name) const;

  //! Valeur de la propriété de nom \a name.
  bool getBoolWithDefault(const String& name,bool default_value) const;

  //! Valeur de la propriété de nom \a name.
  bool get(const String& name,bool& value) const;

  //! Positionne une propriété de type Int32 de nom \a name et de valeur \a value.
  void setInt32(const String& name,Int32 value);

  //! Positionne une propriété de type Int32 de nom \a name et de valeur \a value.
  void set(const String& name,Int32 value);
  
  //! Valeur de la propriété de nom \a name.
  Int32 getInt32(const String& name) const;

  //! Valeur de la propriété de nom \a name.
  Int32 getInt32WithDefault(const String& name,Int32 default_value) const;

  //! Valeur de la propriété de nom \a name.
  bool get(const String& name,Int32& value) const;

  //! Positionne une propriété de type Int64 de nom \a name et de valeur \a value.
  void setInt64(const String& name,Int64 value);
  
  //! Positionne une propriété de type Int64 de nom \a name et de valeur \a value.
  void set(const String& name,Int64 value);

  //! Valeur de la propriété de nom \a name.
  Int64 getInt64(const String& name) const;

  //! Valeur de la propriété de nom \a name.
  Int64 getInt64WithDefault(const String& name,Int64 default_value) const;

  //! Valeur de la propriété de nom \a name.
  bool get(const String& name,Int64& value) const;

  //! Positionne une propriété de type Integer de nom \a name et de valeur \a value.
  void setInteger(const String& name,Integer value);
  
  //! Valeur de la propriété de nom \a name.
  Integer getInteger(const String& name) const;

  //! Valeur de la propriété de nom \a name.
  Integer getIntegerWithDefault(const String& name,Integer default_value) const;

  //! Positionne une propriété de type Real de nom \a name et de valeur \a value.
  void setReal(const String& name,Real value);

  //! Positionne une propriété de type Real de nom \a name et de valeur \a value.
  void set(const String& name,Real value);
  
  //! Valeur de la propriété de nom \a name.
  Real getReal(const String& name) const;

  //! Valeur de la propriété de nom \a name.
  Real getRealWithDefault(const String& name,Real default_value) const;

  //! Valeur de la propriété de nom \a name.
  bool get(const String& name,Real& value) const;

  //! Positionne une propriété de type String de nom \a name et de valeur \a value.
  void setString(const String& name,const String& value);

  //! Positionne une propriété de type String de nom \a name et de valeur \a value.
  void set(const String& name,const String& value);
  
  //! Valeur de la propriété de nom \a name.
  String getString(const String& name) const;

  //! Valeur de la propriété de nom \a name.
  String getStringWithDefault(const String& name,const String& default_value) const;

  //! Valeur de la propriété de nom \a name.
  bool get(const String& name,String& value) const;

  //! Positionne une propriété de type BoolUniqueArray de nom \a name et de valeur \a value.
  void set(const String& name,BoolConstArrayView value);
  
  //! Valeur de la propriété de nom \a name.
  void get(const String& name,BoolArray& value) const;

  //! Positionne une propriété de type Int32UniqueArray de nom \a name et de valeur \a value.
  void set(const String& name,Int32ConstArrayView value);
  
  //! Valeur de la propriété de nom \a name.
  void get(const String& name,Int32Array& value) const;

  //! Positionne une propriété de type Int64UniqueArray de nom \a name et de valeur \a value.
  void set(const String& name,Int64ConstArrayView value);
  
  //! Valeur de la propriété de nom \a name.
  void get(const String& name,Int64Array& value) const;

  //! Positionne une propriété de type RealUniqueArray de nom \a name et de valeur \a value.
  void set(const String& name,RealConstArrayView value);
  
  //! Valeur de la propriété de nom \a name.
  void get(const String& name,RealArray& value) const;

  //! Positionne une propriété de type StringUniqueArray de nom \a name et de valeur \a value.
  void set(const String& name,StringConstArrayView value);
  
  //! Valeur de la propriété de nom \a name.
  void get(const String& name,StringArray& value) const;

 public:

  //! Sort les propriétés et leurs valeurs sur le flot \a o
  void print(std::ostream& o) const;

  //! Effectue la sérialisation des propriétés
  void serialize(ISerializer* serializer);

  //! Nom de la propriété.
  const String& name() const;

  //! Nom complet de la propriété.
  const String& fullName() const;

  IPropertyMng* propertyMng() const;

  /*!
   * \brief Supprime les valeurs associées des propriétés associées à cette référence.
   */
  void destroy();

  //! \internal
  PropertiesImpl* impl() const { return m_p; }

  //! \internal
  PropertiesImplBase* baseImpl() const { return m_ref.get(); }

 private:

  PropertiesImpl* m_p;
  AutoRefT<PropertiesImplBase> m_ref;

 private:

  Properties(PropertiesImpl* p);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

