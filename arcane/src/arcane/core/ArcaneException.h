// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ArcaneException.h                                           (C) 2000-2024 */
/*                                                                           */
/* Exceptions lancées par Arcane.                                            */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ARCANEEXCEPTION_H
#define ARCANE_ARCANEEXCEPTION_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Exception.h"
#include "arcane/utils/String.h"

#include "arcane/core/ArcaneTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class IVariable;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Exception sur un identifiant non valide.
 *
 * Cette exception est envoyée à chaque fois qu'un identifiant non valide
 * est utilisé dans l'architecture.

 Les règles suivantes doivent être respectées pour qu'un identifiant soit
 valide:
 
 \arg il doit comporter au moins un caractère.
 \arg il doit commencer par un caractère alphabétique (a-zA-Z),
 \arg il doit se poursuivre par une suite de caractère alphabétique, de chiffre
 ou le caractère souligné '_'.
 */
class ARCANE_CORE_EXPORT BadIDException
: public Exception
{
 public:

  /*!
   * Construit une exception liée au gestionnaire \a m, issue de la fonction
   * \a where et avec le nom invalide \a invalid_name.
   */
  BadIDException(const String& where,const String& invalid_name);
  ~BadIDException() ARCANE_NOEXCEPT override {}

 public:

  void explain(std::ostream& m) const override;

 private:

  String m_invalid_name; //!< Identifiant invalide.
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Exception sur un numéro d'entité non valide.
 *
 Cette exception est envoyée à chaque fois qu'un numéro d'entité (qu'il soit
 local ou global) est non valide.
 */
class ARCANE_CORE_EXPORT BadItemIdException
: public Exception
{
 public:

  /*!
   * \brief Construit une exception.
   
   Construit une exception liée au gestionnaire de message \a m,
   issue de la fonction \a where et avec le numéro invalide \a id.
   */
  BadItemIdException(const String& where,Integer bad_id);
  ~BadItemIdException() ARCANE_NOEXCEPT override {}

 public:

  void explain(std::ostream& m) const override;

 private:

  Integer m_bad_id; //!< Numéro invalide.
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Exception lorsqu'une erreur interne survient.
 */
class ARCANE_CORE_EXPORT InternalErrorException
: public Exception
{
 public:
	
  InternalErrorException(const String& where,const String& why);
  InternalErrorException(const TraceInfo& where,const String& why);
  InternalErrorException(const InternalErrorException& ex) ARCANE_NOEXCEPT;
  ~InternalErrorException() ARCANE_NOEXCEPT override {}

 public:
	
  void explain(std::ostream& m) const override;

 private:

  String m_why;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Exception sur un genre/type de variable non valide.
 *
 * Cette exception est envoyée lorsqu'on essaye de référencer une variable
 * qui existe déjà dans un autre module avec un genre ou un type
 * différent.
 */
class ARCANE_CORE_EXPORT BadVariableKindTypeException
: public Exception
{
 public:
	
  BadVariableKindTypeException(const TraceInfo& where,IVariable* valid_var,
                               eItemKind kind,eDataType datatype,int dimension);
  ~BadVariableKindTypeException() ARCANE_NOEXCEPT override {}
  
 public:

  void explain(std::ostream& m) const override;

 private:

  IVariable *m_valid_var;
  eItemKind m_item_kind;
  eDataType m_data_type;
  int m_dimension;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Exception sur un nom de groupe d'items de variable partielle non valide.
 *
 * Cette exception est envoyée lorsqu'on essaye de référencer une variable partielle
 * qui existe déjà dans un autre module avec un nom de groupe d'items
 * différent.
 */
class ARCANE_CORE_EXPORT BadPartialVariableItemGroupNameException
: public Exception
{
 public:

  BadPartialVariableItemGroupNameException(const TraceInfo& where,IVariable* valid_var,
                                           const String& item_group_name);
  ~BadPartialVariableItemGroupNameException() ARCANE_NOEXCEPT override {}

 public:

  void explain(std::ostream& m) const override;

 private:

  IVariable *m_valid_var;
  String m_item_group_name;
};


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Exception lorsqu'une entité du maillage n'est pas d'un type
 * connu.
 */
class ARCANE_CORE_EXPORT UnknownItemTypeException
: public Exception
{
 public:
	
  UnknownItemTypeException(const String& where,Integer nb_node,Integer item_id);
  UnknownItemTypeException(const UnknownItemTypeException& ex) ARCANE_NOEXCEPT;
  ~UnknownItemTypeException() ARCANE_NOEXCEPT override {}

 public:
	
  void explain(std::ostream& m) const override;

 private:

  Integer m_nb_node;
  Integer m_item_id;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Exception lorsqu'on essaie de déréférencer un pointer nul.
 */
class ARCANE_CORE_EXPORT BadReferenceException
: public Exception
{
 public:
	
  explicit BadReferenceException(const String& where);
  ~BadReferenceException() ARCANE_NOEXCEPT override {}

 public:
	
  void explain(std::ostream& m) const override;

 private:
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Exception dans un lecteur ou écrivain.
 */
class ARCANE_CORE_EXPORT ReaderWriterException
: public Exception
{
 public:
	
  ReaderWriterException(const String& where,const String& message);
  ReaderWriterException(const TraceInfo& where,const String& message);
  ReaderWriterException(const ReaderWriterException& ex) ARCANE_NOEXCEPT;
  ~ReaderWriterException() ARCANE_NOEXCEPT override {}

 public:
	
  void explain(std::ostream& m) const override;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Exception dans une assertion.
 */
class ARCANE_CORE_EXPORT AssertionException
: public Exception
{
 public:
  /*!
   * Construit une exception issue de la fonction \a where.
   */
  explicit AssertionException(const TraceInfo& where);

  /*!
   * Construit une exception issue de la fonction \a where.
   * La valeur attendue dans l'assertion était \a expected, le résultat obtenu \a actual.
   */
  AssertionException(const TraceInfo& where, const String& expected, const String& actual);

 public:

  void explain(std::ostream& m) const override;
  //! Fichier de l'exception
  const char* file() const { return m_file; }
  //! Ligne de l'exception
  int line() const { return m_line; }

 public:

  using Exception::where;
  using Exception::message;

 private:

  const char* m_file;
  int m_line;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
