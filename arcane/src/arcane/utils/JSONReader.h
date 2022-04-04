﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* JSONReader.h                                                (C) 2000-2020 */
/*                                                                           */
/* Lecteur au format JSON.                                                   */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_JSONREADER_H
#define ARCANE_UTILS_JSONREADER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/UtilsTypes.h"
#include "arcane/utils/Array.h"

#include <vector>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class JSONWrapperUtils;
class JSONKeyValue;
class JSONKeyValueList;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Représente une valeur JSON.
 *
 * Les instances de cette classe ne sont valides que tant que le document
 * associé existe.
 *
 * \warning API Interne. Ne pas utiliser en dehors de Arcane.
 */
class ARCANE_UTILS_EXPORT JSONValue
{
  class Impl;
  friend JSONWrapperUtils;
  friend JSONKeyValue;
 private:
  explicit JSONValue(Impl* p) : m_p(p){}
 public:
  JSONValue() : m_p(nullptr){}
 public:
  //! Vrai si le noeud est nul
  bool null() const { return !m_p; }
  bool operator!() const { return null(); }
 public:
  StringView valueAsString() const;
  Real valueAsReal() const;
  Int64 valueAsInt64() const;
  Int32 valueAsInt32() const;
  JSONValueList valueAsArray() const;
 public:
  JSONKeyValue keyValueChild(StringView name) const;
  //! Valeur fille de nom \a name. Retourne une valeur nulle si non trouvé.
  JSONValue child(StringView name) const;
  //! Valeur fille de nom \a name. Lance une exception si non trouvé.
  JSONValue expectedChild(StringView name) const;
  // Liste des objects fils de cet objet. L'instance doit être un objet
  JSONValueList children() const;
  JSONKeyValueList keyValueChildren() const;
 public:
  bool isArray() const;
  bool isObject() const;
 private:
  Impl* m_p;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Représente une paire (clé,valeur) de JSON.
 *
 * Les instances de cette classe ne sont valides que tant que le document
 * associé existe.
 *
 * \warning API Interne. Ne pas utiliser en dehors de Arcane.
 */
class ARCANE_UTILS_EXPORT JSONKeyValue
{
  class Impl;
  friend JSONWrapperUtils;
 private:
  explicit JSONKeyValue(Impl* p) : m_p(p){}
 public:
  JSONKeyValue() : m_p(nullptr){}
 public:
  //! Vrai si le noeud est nul
  bool null() const { return !m_p; }
  bool operator!() const { return null(); }
 public:
  StringView name() const;
  JSONValue value() const;
 private:
  Impl* m_p;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Liste de (clé,valeur) d'un document JSON.
 *
 * Les instances de cette classe ne sont valides que tant que le document
 * associé existe.
 *
 * \warning API Interne. Ne pas utiliser en dehors de Arcane.
 */
class ARCANE_UTILS_EXPORT JSONKeyValueList
{
  typedef std::vector<JSONKeyValue> ContainerType;
 public:
  typedef ContainerType::const_iterator const_iterator;
  typedef ContainerType::iterator iterator;
 public:
  void add(JSONKeyValue v)
  {
    m_values.push_back(v);
  }
  const_iterator begin() const { return m_values.begin(); }
  const_iterator end() const { return m_values.end(); }
 private:
  std::vector<JSONKeyValue> m_values;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Liste de valeurs d'un document JSON.
 *
 * Les instances de cette classe ne sont valides que tant que le document
 * associé existe.
 *
 * \warning API Interne. Ne pas utiliser en dehors de Arcane.
 */
class ARCANE_UTILS_EXPORT JSONValueList
{
  typedef std::vector<JSONValue> ContainerType;
 public:
  typedef ContainerType::const_iterator const_iterator;
  typedef ContainerType::iterator iterator;
 public:
  void add(JSONValue v)
  {
    m_values.push_back(v);
  }
  const_iterator begin() const { return m_values.begin(); }
  const_iterator end() const { return m_values.end(); }
 private:
  std::vector<JSONValue> m_values;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Gestion d'un document JSON.
 *
 * \warning API Interne. Ne pas utiliser en dehors de Arcane.
 */
class ARCANE_UTILS_EXPORT JSONDocument
{
  class Impl;
 public:
  JSONDocument();
  ~JSONDocument();
 public:
  //! Lit le fichier au format UTF-8.
  void parse(Span<const Byte> bytes);
  //! Lit le fichier au format UTF-8.
  void parse(Span<const std::byte> bytes);
  //! Lit le fichier au format UTF-8.
  void parse(Span<const Byte> bytes,StringView file_name);
  //! Lit le fichier au format UTF-8.
  void parse(Span<const std::byte> bytes,StringView file_name);
  //! Elément racine
  JSONValue root() const;
 private:
  Impl* m_p;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

