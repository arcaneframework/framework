// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Property.h                                                  (C) 2000-2025 */
/*                                                                           */
/* Gestion des propriétés.                                                   */
/*---------------------------------------------------------------------------*/
#ifndef ARCCORE_COMMON_INTERNAL_PROPERTY_H
#define ARCCORE_COMMON_INTERNAL_PROPERTY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/String.h"
#include "arccore/base/Ref.h"
#include "arccore/common/internal/PropertyDeclarations.h"

#include <iosfwd>
#include <functional>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::properties
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename T, typename DataType>
class PropertySetting;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename T>
class PropertySettingTraits
{};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <>
class ARCCORE_COMMON_EXPORT PropertySettingTraits<String>
{
 public:

  typedef StringView InputType;
  typedef String OutputType;
  static InputType fromJSON(const JSONValue& jv);
  static InputType fromString(const String& v);
  static void print(std::ostream& o, InputType v);
  static const char* typeName() { return "String"; }
};

template <>
class ARCCORE_COMMON_EXPORT PropertySettingTraits<StringList>
{
 public:

  typedef StringList InputType;
  typedef Collection<String> OutputType;
  static InputType fromJSON(const JSONValue& jv);
  static InputType fromString(const String& v);
  static void print(std::ostream& o, Collection<String> v);
  static const char* typeName() { return "StringList"; }
};

template <>
class ARCCORE_COMMON_EXPORT PropertySettingTraits<bool>
{
 public:

  typedef bool InputType;
  typedef bool OutputType;
  static InputType fromJSON(const JSONValue& jv);
  static InputType fromString(const String& v);
  static void print(std::ostream& o, InputType v);
  static const char* typeName() { return "Bool"; }
};

template <>
class ARCCORE_COMMON_EXPORT PropertySettingTraits<Int32>
{
 public:

  typedef Int32 InputType;
  typedef Int32 OutputType;
  static InputType fromJSON(const JSONValue& jv);
  static InputType fromString(const String& v);
  static void print(std::ostream& o, InputType v);
  static const char* typeName() { return "Int32"; }
};

template <>
class ARCCORE_COMMON_EXPORT PropertySettingTraits<Int64>
{
 public:

  typedef Int64 InputType;
  typedef Int64 OutputType;
  static InputType fromJSON(const JSONValue& jv);
  static InputType fromString(const String& v);
  static void print(std::ostream& o, InputType v);
  static const char* typeName() { return "Int64"; }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface d'un paramètre de propriété.
 */
class ARCCORE_COMMON_EXPORT IPropertySetting
{
 public:

  virtual ~IPropertySetting() = default;

 public:

  //! Nom de la propriété
  virtual String name() const = 0;
  //! Type de la propriété
  virtual String typeName() const = 0;
  //! Nom de l'argument de la ligne de commande (nul si aucun)
  virtual String commandLineArgument() const = 0;
  //! Description de la propriété
  virtual String description() const = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe de base d'une proriété typée par une classe.
 */
template <typename T>
class PropertySettingBase
{
 public:

  virtual IPropertySetting* setting() = 0;
  virtual const IPropertySetting* setting() const = 0;
  virtual void setFromJSON(const JSONValue& v, T& instance) const = 0;
  virtual void setFromString(const String& v, T& instance) const = 0;
  virtual void print(std::ostream& o, const T& instance) const = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename T>
class PropertySettingBuilder
{
 public:

  PropertySetting<T, String> addString(StringView name)
  {
    return PropertySetting<T, String>(name);
  }
  PropertySetting<T, StringList> addStringList(StringView name)
  {
    return PropertySetting<T, StringList>(name);
  }
  PropertySetting<T, bool> addBool(StringView name)
  {
    return PropertySetting<T, bool>(name);
  }
  PropertySetting<T, Int64> addInt64(StringView name)
  {
    return PropertySetting<T, Int64>(name);
  }
  PropertySetting<T, Int32> addInt32(StringView name)
  {
    return PropertySetting<T, Int32>(name);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Interface d'un visiteur sur une propriété.
 */
class ARCCORE_COMMON_EXPORT IPropertyVisitor
{
 public:

  virtual ~IPropertyVisitor() = default;
  virtual void visit(const IPropertySetting* ps) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe de base d'un visiteur typé sur une propriété.
 */
template <typename T>
class PropertyVisitor
{
 public:

  virtual void visit(const PropertySettingBase<T>&) = 0;

 public:

  PropertySettingBuilder<T> builder()
  {
    return PropertySettingBuilder<T>();
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename T>
class GenericPropertyVisitorWrapper
: public PropertyVisitor<T>
{
 public:

  GenericPropertyVisitorWrapper(IPropertyVisitor* pv)
  : m_visitor(pv)
  {}

 public:

  void visit(const PropertySettingBase<T>& p) override
  {
    m_visitor->visit(p.setting());
  }

 private:

  IPropertyVisitor* m_visitor;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename T, typename DataType>
class PropertySetting
: public PropertySettingBase<T>
, public IPropertySetting
{
 public:

  typedef PropertySetting<T, DataType> ThatClass;
  typedef PropertySettingTraits<DataType> SettingsTraits;
  typedef typename SettingsTraits::InputType InputType;
  typedef typename SettingsTraits::OutputType OutputType;

 public:

  class SetterArg
  {
   public:

    SetterArg(T& ax, InputType av)
    : x(ax)
    , v(av)
    {}
    T& x;
    InputType v;
  };
  class GetterArg
  {
   public:

    GetterArg(const T& ax)
    : x(ax)
    {}
    const T& x;
  };

 public:

  typedef std::function<void(SetterArg a)> SetterType;
  typedef std::function<OutputType(GetterArg a)> GetterType;

 public:

  PropertySetting(StringView name, GetterType getter, SetterType setter)
  : m_name(name)
  , m_getter(getter)
  , m_setter(setter)
  {}
  PropertySetting(StringView name)
  : m_name(name)
  , m_getter(nullptr)
  , m_setter(nullptr)
  {}

 public:

  IPropertySetting* setting() final
  {
    return this;
  }
  const IPropertySetting* setting() const final
  {
    return this;
  }
  String name() const final
  {
    return m_name;
  }
  String commandLineArgument() const final
  {
    return m_command_line_argument;
  }
  String description() const final
  {
    return m_description;
  }
  String typeName() const final
  {
    return SettingsTraits::typeName();
  }
  ThatClass& addSetter(const SetterType& setter)
  {
    m_setter = setter;
    return (*this);
  }
  ThatClass& addGetter(const GetterType& getter)
  {
    m_getter = getter;
    return (*this);
  }
  ThatClass& addCommandLineArgument(const String& arg)
  {
    m_command_line_argument = arg;
    return (*this);
  }
  ThatClass& addDescription(const String& arg)
  {
    m_description = arg;
    return (*this);
  }
  void setInstanceValue(T& instance, InputType value) const
  {
    m_setter(SetterArg(instance, value));
  }
  void setFromJSON(const JSONValue& v, T& instance) const override
  {
    InputType x1 = SettingsTraits::fromJSON(v);
    setInstanceValue(instance, x1);
  }
  void setFromString(const String& v, T& instance) const override
  {
    InputType x1 = SettingsTraits::fromString(v);
    setInstanceValue(instance, x1);
  }
  void print(std::ostream& o, const T& instance) const override
  {
    o << "PROP: name=" << m_name << " V=";
    if (m_getter)
      SettingsTraits::print(o, m_getter(GetterArg(instance)));
    else
      o << "?";
    o << "\n";
  }
  friend PropertyVisitor<T>& operator<<(PropertyVisitor<T>& o, const ThatClass& me)
  {
    o.visit(me);
    return o;
  }

 private:

  String m_name;
  GetterType m_getter;
  SetterType m_setter;
  String m_command_line_argument;
  String m_description;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCCORE_COMMON_EXPORT PropertySettingsBuildInfo
{
};

class ARCCORE_COMMON_EXPORT IPropertySettingsInfo
{
 public:

  virtual ~IPropertySettingsInfo() = default;

 public:

  virtual void applyVisitor(IPropertyVisitor* v) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename T> class PropertySettingsInfo
: public IPropertySettingsInfo
{
 public:

  explicit PropertySettingsInfo([[maybe_unused]] const PropertySettingsBuildInfo& sbi)
  {
  }

 public:

  static PropertySettingsInfo*
  create(const PropertySettingsBuildInfo& sbi, [[maybe_unused]] const char* filename,
         [[maybe_unused]] int line)
  {
    auto x = new PropertySettingsInfo<T>(sbi);
    return x;
  }

 public:

  void applyVisitor(IPropertyVisitor* pv) override
  {
    T ::applyPropertyVisitor(pv);
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Enregistreur de paramètres de propriétés
 */
class ARCCORE_COMMON_EXPORT PropertySettingsRegisterer
{
 public:

  typedef IPropertySettingsInfo* (*CreateFunc)(const PropertySettingsBuildInfo& sbi);
  typedef PropertySettingsBuildInfo (*CreateBuildInfoFunc)();

 public:

  PropertySettingsRegisterer(CreateFunc func, CreateBuildInfoFunc build_info_func,
                             const char* name) ARCCORE_NOEXCEPT;

 public:

  //! Accès au premier élément de la chaine d'enregistreur
  static PropertySettingsRegisterer* firstRegisterer();

  //! Nombre d'enregisteur dans la chaîne
  static Integer nbRegisterer();

  //! Enregistreur précédent (nullptr si le premier)
  PropertySettingsRegisterer* previousRegisterer() const { return m_previous; }

  //! Enregistreur suivant (nullptr si le dernier)
  PropertySettingsRegisterer* nextRegisterer() const { return m_next; }

 public:

  //! Nom de classe associée
  const char* name() const { return m_name; }

  Ref<IPropertySettingsInfo> createSettingsInfoRef() const;

 private:

  //! Positionne l'enregistreur précédent
  void _setPreviousRegisterer(PropertySettingsRegisterer* s) { m_previous = s; }

  //! Positionne l'enregistreur suivant
  void _setNextRegisterer(PropertySettingsRegisterer* s) { m_next = s; }

 private:

  //! Enregistreur précédent
  PropertySettingsRegisterer* m_previous = nullptr;
  //! Enregistreur suivant
  PropertySettingsRegisterer* m_next = nullptr;

 private:

  //! Nom de l'enregistreur
  const char* m_name = nullptr;
  //! Fonction de création
  CreateFunc m_create_func = nullptr;

 private:

  void _init();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Applique un visiteur à tous les 'IPropertySetting' enregistrés
 * via 'PropertySettingsRegisterer'.
 */
extern "C++" ARCCORE_COMMON_EXPORT void
visitAllRegisteredProperties(IPropertyVisitor* visitor);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::properties

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
