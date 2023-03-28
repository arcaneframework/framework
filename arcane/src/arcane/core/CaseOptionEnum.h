// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CaseOptionEnum.h                                            (C) 2000-2023 */
/*                                                                           */
/* Option du jeu de données de type énuméré.                                 */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CASEOPTIONENUM_H
#define ARCANE_CASEOPTIONENUM_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/CaseOptionSimple.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Nom d'une option du jeu de données.
 * Cette classe permet de stocker le nom d'une option dans plusieurs
 * langues.
 */
class ARCANE_CORE_EXPORT CaseOptionName
{
 public:

  //! Construit une option de nom \a true_name
  CaseOptionName(const String& true_name);
  //! Constructeur de recopie
  CaseOptionName(const CaseOptionName& rhs);
  //! Libère les ressources
  virtual ~CaseOptionName();

 public:

  /*! \brief retourne le nom de l'option dans le langage \a lang.
   * Si aucune traduction n'est disponible dans le langage \a lang,
   * c'est trueName() qui est retourné.
   */
  String name(const String& lang) const;
  //! Retourne le vrai nom (non traduit) de l'option
  String trueName() const { return m_true_name; }
  /*!
    \brief Ajoute une traduction pour le nom de l'option.
    Ajoute le nom \a tname correspondant au langage \a lang.
    Si une traduction existe déjà pour ce langage, elle est remplacée par
    celle-ci.
    \param tname traduction du nom
    \param lang langue de la traduction
  */
  void addAlternativeNodeName(const String& lang, const String& tname);

 private:

  String m_true_name; //!< Nom de l'option
  StringDictionary* m_translations; //!< Traductions.
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Nom et valeur d'une énumération du jeu de données.
 */
class ARCANE_CORE_EXPORT CaseOptionEnumValue
: public CaseOptionName
{
 public:

  CaseOptionEnumValue(const String& name, int value);
  //! Constructeur de recopie
  CaseOptionEnumValue(const CaseOptionEnumValue& rhs);
  ~CaseOptionEnumValue();

 public:

  int value() const { return m_value; }

 private:

  int m_value;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Ensemble des valeurs d'une énumération.
 */
class ARCANE_CORE_EXPORT CaseOptionEnumValues
{
 public:

  //! Type de la liste des valeurs
  typedef UniqueArray<CaseOptionEnumValue*> EnumValueList;

 public:

  //! Contruit l'instance
  CaseOptionEnumValues();
  ~CaseOptionEnumValues(); //!< Libère les ressources

 public:

  /*! \brief Ajoute la valeur d'énumération \a value.
   * L'instance devient propriétaire de \a value qui est détruite
   * lorsqu'elle n'est plus utilisée.
   * Cette fonction ne doit être appelée qu'à l'initialisation.
   * Si \a do_clone est vrai, c'est une copie de \a value qui est utilisée
   */
  void addEnumValue(CaseOptionEnumValue* value, bool do_clone);

  //! Retourne le nombre de valeurs de l'énumération
  Integer nbEnumValue() const;

  //! Retourne la ième valeur
  CaseOptionEnumValue* enumValue(Integer index) const;

  /*! \brief Retourne la valeur de l'énumération ayant le nom \a name
   *
   * La valeur est retournée dans \a index.
   * \param name nom de l'énumération
   * \param lang est le langage du jeu de données
   * \param value est la valeur de l'énumération (en retour)
   * \retval true en cas d'erreur,
   * \retval false en cas de succès.
   */
  bool valueOfName(const String& name, const String& lang, int& value) const;

  //! Retourne le nom de correspondant à la valeur \a value pour le langage \a lang
  String nameOfValue(int value, const String& lang) const;

  /*!
   * \brief Remplit \a names avec les noms valides pour la langue \a lang.
   */
  void getValidNames(const String& lang, StringArray& names) const;

 private:

  EnumValueList* m_enum_values; //!< Valeurs de l'énumération
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Option du jeu de donnée de type énumération.
 * \ingroup CaseOption
 */
class ARCANE_CORE_EXPORT CaseOptionEnum
: public CaseOptionSimple
{
 public:

  CaseOptionEnum(const CaseOptionBuildInfo& cob, const String& type_name);
  ~CaseOptionEnum();

 public:

  virtual void print(const String& lang, std::ostream& o) const;
  virtual void updateFromFunction(Real current_time, Integer current_iteration)
  {
    _updateFromFunction(current_time, current_iteration);
  }

  void addEnumValue(CaseOptionEnumValue* value, bool do_clone)
  {
    m_enum_values->addEnumValue(value, do_clone);
  }
  CaseOptionEnumValues* enumValues() const { return m_enum_values; }

  virtual void visit(ICaseDocumentVisitor* visitor) const;

  int enumValueAsInt() const { return _optionValue(); }

 public:
 protected:

  virtual void _search(bool is_phase1);
  virtual bool _allowPhysicalUnit() { return false; }

  //! Positionne à \a v la valeur de l'option
  virtual void _setOptionValue(int v) = 0;
  //! Retourne la valeur de l'option
  virtual int _optionValue() const = 0;

 protected:

  void _setEnumDefaultValue(int def_value);

 private:

  String m_type_name; //!< Nom de l'énumération
  CaseOptionEnumValues* m_enum_values;
  void _updateFromFunction(Real current_time, Integer current_iteration);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup CaseOption
 * \brief Option du jeu de données de type énuméré.
 *
 * \a T est le type informatique de l'énumération.
 */
template <class EnumType>
class CaseOptionEnumT
: public CaseOptionEnum
{
 public:

  CaseOptionEnumT(const CaseOptionBuildInfo& cob, const String& type_name)
  : CaseOptionEnum(cob, type_name)
  , m_value(EnumType())
  {}

 public:

  //! Valeur de l'option
  EnumType value() const
  {
    ARCANE_CASEOPTION_CHECK_IS_INITIALIZED;
    return m_value;
  }

  //! Valeur de l'option
  operator EnumType() const { return value(); }

  //! Valeur de l'option
  EnumType operator()() const { return value(); }

  /*!
   * \brief Positionne la valeur par défaut de l'option.
   *
   * Si l'option n'est pas pas présente dans le jeu de données, alors sa valeur sera
   * celle spécifiée par l'argument \a def_value, sinon l'appel de cette méthode est sans effet.
   */
  void setDefaultValue(EnumType def_value)
  {
    _setEnumDefaultValue(static_cast<int>(def_value));
  }

  //! Retourne la valeur de l'option si isPresent()==true ou sinon \a arg_value
  EnumType valueIfPresentOrArgument(EnumType arg_value)
  {
    ARCANE_CASEOPTION_CHECK_IS_INITIALIZED;
    return isPresent() ? m_value : arg_value;
  }

 private:

  EnumType m_value; //!< Valeur de l'option

 public:
 protected:

  virtual void _setOptionValue(int i)
  {
    m_value = static_cast<EnumType>(i);
  }
  virtual int _optionValue() const
  {
    return static_cast<int>(m_value);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Option du jeu de données de type liste d'énumération.
 * \ingroup CaseOption
 */
class ARCANE_CORE_EXPORT CaseOptionMultiEnum
: public CaseOptionBase
{
 public:
 public:

  CaseOptionMultiEnum(const CaseOptionBuildInfo& cob, const String& type_name);
  ~CaseOptionMultiEnum();

 public:

  virtual void print(const String& lang, std::ostream& o) const;
  virtual ICaseFunction* function() const { return 0; }
  virtual void updateFromFunction(Real /*current_time*/, Integer /*current_iteration*/) {}

  void addEnumValue(CaseOptionEnumValue* value, bool do_clone)
  {
    m_enum_values->addEnumValue(value, do_clone);
  }

  CaseOptionEnumValues* enumValues() const { return m_enum_values; }

  virtual void visit(ICaseDocumentVisitor* visitor) const;

 protected:

  virtual void _search(bool is_phase1);
  virtual bool _allowPhysicalUnit() { return false; }

  //! Alloue un tableau pour \a size éléments
  virtual void _allocate(Integer size) = 0;
  //! Retourne le nombre d'éléments du tableau.
  virtual Integer _nbElem() const = 0;
  /*! Positionne la valeur de l'option à la valeur \a v.
   * \a v est directement convertie en la valeur de l'énumération.
   */
  virtual void _setOptionValue(Integer index, int v) = 0;
  //! Retourne la valeur de l'énumération pour l'indice \a index.
  virtual int _optionValue(Integer index) const = 0;

 private:

  String m_type_name; //!< Nom de l'énumération
  CaseOptionEnumValues* m_enum_values;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Option du jeu de données de type liste de types énumérés.
 * \ingroup CaseOption
 */
template <class T>
class CaseOptionMultiEnumT
: public CaseOptionMultiEnum
, public ArrayView<T>
{
 public:

  typedef T Type; //!< Type de l'option.

 public:

  CaseOptionMultiEnumT(const CaseOptionBuildInfo& cob, const String type_name)
  : CaseOptionMultiEnum(cob, type_name)
  {}

 protected:

  virtual void _allocate(Integer size)
  {
    m_values.resize(size);
    ArrayView<T>* view = this;
    *view = m_values.view();
  }
  virtual Integer _nbElem() const
  {
    return this->size();
  }
  virtual void _setOptionValue(Integer index, int v)
  {
    (*this)[index] = static_cast<T>(v);
  }
  virtual int _optionValue(Integer index) const
  {
    return static_cast<int>((*this)[index]);
  }

 private:

  UniqueArray<T> m_values;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
