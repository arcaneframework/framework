// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* JSONPropertyReader.h                                        (C) 2000-2025 */
/*                                                                           */
/* Lecture de propriétés au format JSON.                                     */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_INTERNAL_JSONPROPERTYREADER_H
#define ARCANE_UTILS_INTERNAL_JSONPROPERTYREADER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*
 * NOTE: Les classes de ce fichier sont en cours de mise au point.
 * NOTE: L'API peut changer à tout moment. Ne pas utiliser en dehors de Arcane.
 */

#include "arccore/common/JSONReader.h"
#include "arccore/common/internal/Property.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::properties
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename T>
class JSONPropertyReader
: public PropertyVisitor<T>
{
 public:
  JSONPropertyReader(JSONValue jv,T& instance)
  : m_jv(jv), m_instance(instance){}
 private:
  JSONValue m_jv;
  T& m_instance;
 public:
  void visit(const PropertySettingBase<T>& s) override
  {
    JSONValue child_value = m_jv.child(s.setting()->name());
    if (child_value.null())
      return;
    s.setFromJSON(child_value,m_instance);
    s.print(std::cout,m_instance);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Remplit les valeurs de \a instance à partir de l'élément JSON \a jv.
 *
 * Les valeurs de la propriété doivent être dans un élément fils de \a jv
 * dont le nom est celui de la classe \a T.
 */
template<typename T, typename PropertyType = T> inline void
readFromJSON(JSONValue jv,T& instance)
{
  const char* instance_property_name = PropertyType :: propertyClassName();
  JSONValue child_value = jv.child(instance_property_name);
  if (child_value.null())
    return;
  JSONPropertyReader reader(child_value,instance);
  PropertyType :: applyPropertyVisitor(reader);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::properties

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
