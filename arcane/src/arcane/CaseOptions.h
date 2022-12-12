// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CaseOptions.h                                               (C) 2000-2022 */
/*                                                                           */
/* Options du jeu de données.                                                */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CASEOPTIONS_H
#define ARCANE_CASEOPTIONS_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Array.h"
#include "arcane/utils/String.h"

#include "arccore/base/ReferenceCounter.h"

#include "arcane/XmlNode.h"
#include "arcane/ICaseOptions.h"
#include "arcane/ICaseOptionList.h"
#include "arcane/CaseOptionBase.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IServiceInfo;
class IModule;
class ICaseOptions;
class ICaseFunction;
class IStandardFunction;
class ICaseDocument;
class ICaseMng;
class IScriptImpl;
class StringDictionary;
class XmlNodeList;
class IPhysicalUnitConverter;

class CaseOptionBuildInfo;

#ifdef ARCANE_CHECK
#define ARCANE_CASEOPTION_CHECK_IS_INITIALIZED _checkIsInitialized()
#else
#define ARCANE_CASEOPTION_CHECK_IS_INITIALIZED
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename Type>
class CaseOptionTraitsT
{
 public:
  using ContainerType = Type;
  using ReferenceType = Type&;
  using ConstReferenceType = const Type&;
  using ArrayViewType = ArrayView<Type>;
  using ConstArrayViewType = ConstArrayView<Type>;
};

/*!
 * \brief Spécialisation pour les options 'Array'.
 *
 * Cela est nécessaire car on ne peut pas instancier la classe 'Array'. Pour
 * ce type d'options, il est interdit de modifier les valeurs de l'option donc
 * les vues sont forcément constantes.
 */
template<typename Type>
class CaseOptionTraitsT< Array<Type> >
{
 public:
  using ContainerType = UniqueArray<Type>;
  using ReferenceType = const Array<Type>&;
  using ConstReferenceType = const Array<Type>&;
  using ArrayViewType = ConstArrayView<ContainerType>;
  using ConstArrayViewType = ConstArrayView<ContainerType>;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe de base des options simples (uniquement une valeur).
 * \ingroup CaseOption
 */
class ARCANE_CORE_EXPORT CaseOptionSimple
: public CaseOptionBase
{
 public:

  explicit CaseOptionSimple(const CaseOptionBuildInfo& cob);
  CaseOptionSimple(const CaseOptionBuildInfo& cob,const String& physical_unit);
  ~CaseOptionSimple();

 public:

  //! Retourne \a true si l'option est présente
  bool isPresent() const { return !m_element.null(); }

  /*!
   * \brief Retourne l'élément de l'option.
   *
   * \deprecated L'implémentation interne ne doit pas être utilisée pour permettre
   * à terme d'utiliser un autre format que le XML.
   */
  ARCANE_DEPRECATED_REASON("Y2022: Do not access XML item from option")
  XmlNode element() const { return m_element; }

  /*!
   * \brief Fonction associée à cette option (0 si aucune).
   *
   * Si une fonction est associée à l'option, les valeurs de cette
   * dernière sont recalculées automatiquement à chaque itération.
   */
  ICaseFunction* function() const override { return m_function; }
  /*!
   * \brief Fonction standard associée à cette option (0 si aucune).
   *
   * Une fonction standard a un prototype spécifique et peut être appelée
   * directement. Contrairement à function(), la présence d'une fonction
   * standard ne change pas la valeur de l'option.
   */
  virtual IStandardFunction* standardFunction() const { return m_standard_function; }
  /*!
   * \brief Indique si la valeur a changée depuis la dernière itération.
   *
   * La valeur ne peut changée que si une fonction est associée à l'option.
   * La méthode retourne vrai si la valeur de l'option est différente de l'itération
   * précédente. Cette méthode fonctionne aussi en cas de retour arrière.
   */
  bool hasChangedSinceLastIteration() const;
  //! Nom complet au format donné par la norme XPath.
  String xpathFullName() const;
  /*!
   * \brief Unité physique par défaut de cette option (null si aucune unité),
   * spécifiée dans le fichier .axl.
   */
  String defaultPhysicalUnit() const;
  //! unité physique spécifiée dans le jeu de données (null si aucune unité)
  String physicalUnit() const;
  /*!
   * \brief Convertisseur d'unité physique.
   *
   * Ce convertisseur n'existe que pour les options de type 'Real'ou 'RealArray'.
   * Il est nul si l'option ne possède pas d'unité.
   */
  IPhysicalUnitConverter* physicalUnitConverter() const { return m_unit_converter; }
  /*!
   * \brief Indique si l'option est facultative.
   *
   * Si une option facultative n'est pas renseignée,
   * sa valeur est indéfinie et ne doit donc pas être utilisée.
   */
  bool isOptional() const { return m_is_optional; }

  /*!
   * \brief Indique si l'option a une valeur invalide.
   *
   * C'est apriori toujours le cas, sauf si l'option est facultative
   * (isOptional()==true) et non renseignée.
   */
  bool hasValidValue() const { return m_has_valid_value; }

  void visit(ICaseDocumentVisitor* visitor) const override;

 protected:

  void _search(bool is_phase1) override;
  virtual bool _allowPhysicalUnit() =0;
  void _setChangedSinceLastIteration(bool has_changed);
  void _searchFunction(XmlNode& velem);
  void _setPhysicalUnit(const String& value);
  void _setHasValidValue(bool v) { m_has_valid_value = v; }
  XmlNode _element() const { return m_element; }

 private:

  XmlNode m_element; //!< Element de l'option
  ICaseFunction* m_function = nullptr; //!< Fonction associée (ou nullptr)
  IStandardFunction* m_standard_function = nullptr; //!< Fonction standard associée (ou nullpt)
  //! Convertisseur d'unité (nullptr si pas besoin). Valide uniquement pour les options 'Real'
  IPhysicalUnitConverter* m_unit_converter = nullptr;
  bool m_changed_since_last_iteration;
  bool m_is_optional;
  bool m_has_valid_value;
  String m_default_physical_unit;
  String m_physical_unit;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup CaseOption
 * \brief Option du jeu de données de type simple (réel, entier, booléen, ...)
 *
 * La méthode la plus utilisée de cette classe est l'opérateur operator()()
 * qui permet de récupérer la valeur de l'option. Si une fonction (ICaseFunction)
 * est associée à l'option, il est possible de récupérer la valeur de l'option
 * au temps physique ou à l'itération passé en paramètre de la méthode
 * valueAtParameter().
 * \code
 * CaseOptionSimpleT<Real> real_option;
 * Real v = real_option(); // utilise operator()
 * Real v = real_option; // utilise opérateur de cast implicite
 * Real v = real_option.valueAtParameter(0.3); // valeur au temps physique 0.3
 * \endcode
 */
template<class T>
class CaseOptionSimpleT
: public CaseOptionSimple
{
 public:

  typedef CaseOptionSimpleT<T> ThatClass;
#ifndef SWIG
  typedef typename CaseOptionTraitsT<T>::ContainerType Type; //!< Type de l'option
#else
  typedef T Type; //!< Type de l'option
#endif
 public:

  ARCANE_CORE_EXPORT CaseOptionSimpleT(const CaseOptionBuildInfo& cob);
  ARCANE_CORE_EXPORT CaseOptionSimpleT(const CaseOptionBuildInfo& cob,const String& physical_unit);

 public:

  ARCANE_CORE_EXPORT virtual void print(const String& lang,std::ostream& o) const;

  //! Retourne la valeur de l'option
  const Type& value() const
  {
    ARCANE_CASEOPTION_CHECK_IS_INITIALIZED;
    return m_value;
  }

  //! Valeur de l'option
  operator const Type&() const { return value(); }

  //! Retourne la valeur de l'option pour le paramètre réel t.
  ARCANE_CORE_EXPORT Type valueAtParameter(Real t) const;

  //! Retourne la valeur de l'option pour le paramètre entier t.
  ARCANE_CORE_EXPORT Type valueAtParameter(Integer t) const;

  //! Retourne la valeur de l'option
  //const Type& operator()() const { return value(); }
  const Type& operator()() const { return value(); }

#ifdef ARCANE_DOTNET
  operator ThatClass*() { return this; }
  operator const ThatClass*() const { return this; }
  const ThatClass* operator->() const { return this; }
  static Real castTo__Arcane_Real(const ThatClass& v)
  {
    return (Real)(v);
  }
#endif

  //! Retourne la valeur de l'option pour le paramètre réel t.
  ARCANE_DEPRECATED Type operator()(Real t) const
  { return valueAtParameter(t); }

  //! Retourne la valeur de l'option pour le paramètre entier t.
  ARCANE_DEPRECATED Type operator()(Integer t) const
  { return valueAtParameter(t); }

  /*!
   * For internal use only
   * \internal
   */
  ARCANE_CORE_EXPORT virtual void updateFromFunction(Real current_time,Integer current_iteration);

  /*!
   * \brief Positionne la valeur par défaut de l'option.
   *
   * Si l'option n'est pas pas présente dans le jeu de données, alors sa valeur sera
   * celle spécifiée par l'argument \a def_value, sinon l'appel de cette méthode est sans effet.
   */
  ARCANE_CORE_EXPORT void setDefaultValue(const Type& def_value);

  //! Retourne la valeur de l'option si isPresent()==true ou sinon \a arg_value
  const Type& valueIfPresentOrArgument(const Type& arg_value)
  {
    ARCANE_CASEOPTION_CHECK_IS_INITIALIZED;
    return isPresent() ? m_value : arg_value;
  }

 protected:
	
  ARCANE_CORE_EXPORT virtual void _search(bool is_phase1);
  ARCANE_CORE_EXPORT virtual bool _allowPhysicalUnit();

 private:
	
  Type m_value; //!< Valeur de l'option

};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Classe de base d'une valeur d'une option complexe.
 *
 * Une option complexe est composé de plusieurs instances de cette classe.
 */
class ARCANE_CORE_EXPORT CaseOptionComplexValue
{
 public:

  CaseOptionComplexValue(ICaseOptionsMulti* opt,ICaseOptionList* clist,const XmlNode& parent_elem);
  virtual ~CaseOptionComplexValue();

 public:

  ARCANE_DEPRECATED_REASON("Y2022: Do not access XML item from option")
  XmlNode element() const { return m_element; }

  ARCANE_DEPRECATED_REASON("Y2022: This method is internal to Arcane. Do not use it")
  ICaseOptionList* configList() const { return m_config_list.get(); }

  //! Nom complet au format donné par la norme XPath.
  String xpathFullName() const { return m_element.xpathFullName(); }

 protected:

  // Les deux méthodes suivantes sont utilisés par le générateur 'axl2cc' et
  // ne doivent pas être modifiées.
  ICaseOptionList* _configList() { return m_config_list.get(); }
  XmlNode _element() { return m_element; }

 private:

  ReferenceCounter<ICaseOptionList> m_config_list;
  XmlNode m_element;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ARCANE_CORE_EXPORT CaseOptionMultiSimple
: public CaseOptionBase
{
 public:
  CaseOptionMultiSimple(const CaseOptionBuildInfo& cob)
  : CaseOptionBase(cob){}
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup CaseOption
 * \brief Option du jeu de données de type liste de types simples (réel, entier, booléen, ...).
 *
 * \warning L'utilisation de la classe de base `ArrayView<T>` est obsolète et
 * ne doit plus être utilisée. La méthode view() permet de récupérer une vue sur les
 * valeurs de l'option.
 */
#ifndef SWIG
template<class T>
class CaseOptionMultiSimpleT
: public CaseOptionMultiSimple
#ifdef ARCANE_HAS_PRIVATE_CASEOPTIONSMULTISIMPLE_BASE_CLASS
, private ArrayView<T>
#else
, public ArrayView<T>
#endif
{
 public:

  //! Type de la valeur de l'option
  using Type = typename CaseOptionTraitsT<T>::ContainerType;
  using ReferenceType = typename CaseOptionTraitsT<T>::ReferenceType;
  using ConstReferenceType = typename CaseOptionTraitsT<T>::ConstReferenceType;
  //! Type de la vue sur les valeurs de l'option
  using ArrayViewType = typename CaseOptionTraitsT<T>::ArrayViewType;
  //! Type de la vue constante sur les valeurs de l'option
  using ConstArrayViewType = typename CaseOptionTraitsT<T>::ConstArrayViewType;

 public:

  ARCANE_CORE_EXPORT CaseOptionMultiSimpleT(const CaseOptionBuildInfo& cob);
  ARCANE_CORE_EXPORT CaseOptionMultiSimpleT(const CaseOptionBuildInfo& cob,const String& physical_unit);
  ARCANE_CORE_EXPORT ~CaseOptionMultiSimpleT();

 public:

  ARCCORE_DEPRECATED_2021("Use view() instead")
  ArrayView<T> operator()()
  {
    return *this;
  }
  ARCCORE_DEPRECATED_2021("Use view() instead")
  const ArrayView<T> operator()() const
  {
    return *this;
  }

  //! Conversion vers la vue constante
  ARCCORE_DEPRECATED_2021("Use view() instead")
  operator ArrayView<T>() { ArrayView<T>* v = this; return *v; }

  //! Conversion vers la vue constante
  ARCCORE_DEPRECATED_2021("Use view() instead")
  operator ConstArrayView<T>() const { const ArrayView<T>* v = this; return *v; }

  //! Vue constante sur les éléments de l'option
  ConstArrayViewType view() const
  {
    return m_view;
  }
  //! Vue sur les éléments de l'option
  ArrayViewType view()
  {
    return m_view;
  }

  ConstReferenceType operator[](Integer i) const { return m_view[i]; }
  ReferenceType operator[](Integer i) { return m_view[i]; }

 public:

  ARCANE_CORE_EXPORT void print(const String& lang,std::ostream& o) const override;
  ICaseFunction* function() const override { return 0; }
  void updateFromFunction(Real,Integer) override {}

  ConstArrayView<T> values() const { const ArrayView<T>* v = this; return *v; }
  const T& value(Integer index) const { return this->operator[](index); }
  Integer size() const { return ArrayView<T>::size(); }
  ARCANE_CORE_EXPORT void visit(ICaseDocumentVisitor* visitor) const override;

 protected:
	
  void _search(bool is_phase1) override;
  virtual bool _allowPhysicalUnit();

 private:
  ArrayViewType m_view;
};
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

typedef CaseOptionSimpleT<Real> CaseOptionReal;
typedef CaseOptionSimpleT<Real2> CaseOptionReal2;
typedef CaseOptionSimpleT<Real3> CaseOptionReal3;
typedef CaseOptionSimpleT<Real2x2> CaseOptionReal2x2;
typedef CaseOptionSimpleT<Real3x3> CaseOptionReal3x3;
typedef CaseOptionSimpleT<bool> CaseOptionBool;
typedef CaseOptionSimpleT<Integer> CaseOptionInteger;
typedef CaseOptionSimpleT<Int32> CaseOptionInt32;
typedef CaseOptionSimpleT<Int64> CaseOptionInt64;
typedef CaseOptionSimpleT<String> CaseOptionString;

typedef CaseOptionSimpleT<RealArray> CaseOptionRealArray;
typedef CaseOptionSimpleT<Real2Array> CaseOptionReal2Array;
typedef CaseOptionSimpleT<Real3Array> CaseOptionReal3Array;
typedef CaseOptionSimpleT<Real2x2Array> CaseOptionReal2x2Array;
typedef CaseOptionSimpleT<Real3x3Array> CaseOptionReal3x3Array;
typedef CaseOptionSimpleT<BoolArray> CaseOptionBoolArray;
typedef CaseOptionSimpleT<IntegerArray> CaseOptionIntegerArray;
typedef CaseOptionSimpleT<Int32Array> CaseOptionInt32Array;
typedef CaseOptionSimpleT<Int64Array> CaseOptionInt64Array;
typedef CaseOptionSimpleT<StringArray> CaseOptionStringArray;

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
  void addAlternativeNodeName(const String& lang,const String& tname);
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
  CaseOptionEnumValue(const String& name,int value);
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
  void addEnumValue(CaseOptionEnumValue* value,bool do_clone);
  
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
  bool valueOfName(const String& name,const String& lang,int& value) const;

  //! Retourne le nom de correspondant à la valeur \a value pour le langage \a lang
  String nameOfValue(int value,const String& lang) const;

  /*!
   * \brief Remplit \a names avec les noms valides pour la langue \a lang.
   */
  void getValidNames(const String& lang,StringArray& names) const;

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

  CaseOptionEnum(const CaseOptionBuildInfo& cob,const String& type_name);
  ~CaseOptionEnum();

 public:

  virtual void print(const String& lang,std::ostream& o) const;
  virtual void updateFromFunction(Real current_time,Integer current_iteration)
  {
    _updateFromFunction(current_time,current_iteration);
  }

  void addEnumValue(CaseOptionEnumValue* value,bool do_clone)
  {
    m_enum_values->addEnumValue(value,do_clone);
  }
  CaseOptionEnumValues* enumValues() const { return m_enum_values; }

  virtual void visit(ICaseDocumentVisitor* visitor) const;

  int enumValueAsInt() const { return _optionValue(); }

 public:

 protected:
	
  virtual void _search(bool is_phase1);
  virtual bool _allowPhysicalUnit() { return false; }

  //! Positionne à \a v la valeur de l'option
  virtual void _setOptionValue(int v) =0;
  //! Retourne la valeur de l'option
  virtual int _optionValue() const =0;

 protected:

  void _setEnumDefaultValue(int def_value);

 private:

  String m_type_name; //!< Nom de l'énumération
  CaseOptionEnumValues* m_enum_values;
  void _updateFromFunction(Real current_time,Integer current_iteration);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup CaseOption
 * \brief Option du jeu de données de type énuméré.
 *
 * \a T est le type informatique de l'énumération.
 */
template<class EnumType>
class CaseOptionEnumT
: public CaseOptionEnum
{
 public:

  CaseOptionEnumT(const CaseOptionBuildInfo& cob,const String& type_name)
  : CaseOptionEnum(cob,type_name), m_value(EnumType()) {}

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
  { m_value = static_cast<EnumType>(i); }
  virtual int _optionValue() const
  { return static_cast<int>(m_value); }
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

  CaseOptionMultiEnum(const CaseOptionBuildInfo& cob,const String& type_name);
  ~CaseOptionMultiEnum();

 public:

  virtual void print(const String& lang,std::ostream& o) const;
  virtual ICaseFunction* function() const { return 0; }
  virtual void updateFromFunction(Real /*current_time*/,Integer /*current_iteration*/) {}

  void addEnumValue(CaseOptionEnumValue* value,bool do_clone)
  {
    m_enum_values->addEnumValue(value,do_clone);
  }

  CaseOptionEnumValues* enumValues() const { return m_enum_values; }

  virtual void visit(ICaseDocumentVisitor* visitor) const;

 protected:
	
  virtual void _search(bool is_phase1);
  virtual bool _allowPhysicalUnit() { return false; }

  //! Alloue un tableau pour \a size éléments
  virtual void _allocate(Integer size) =0;
  //! Retourne le nombre d'éléments du tableau.
  virtual Integer _nbElem() const =0;
  /*! Positionne la valeur de l'option à la valeur \a v.
   * \a v est directement convertie en la valeur de l'énumération.
   */
  virtual void _setOptionValue(Integer index,int v) =0;
  //! Retourne la valeur de l'énumération pour l'indice \a index.
  virtual int _optionValue(Integer index) const =0;

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
template<class T>
class CaseOptionMultiEnumT
: public CaseOptionMultiEnum
, public ArrayView<T>
{
 public:

  typedef T Type; //!< Type de l'option.

 public:

  CaseOptionMultiEnumT(const CaseOptionBuildInfo& cob,const String type_name)
  : CaseOptionMultiEnum(cob,type_name) {}

 public:

 protected:

  virtual void _allocate(Integer size)
  {
    m_values.resize(size);
    ArrayView<T>* view = this;
    *view = m_values.view();
  }
  virtual Integer _nbElem() const
    { return this->size(); }
  virtual void _setOptionValue(Integer index,int v)
    { (*this)[index] = static_cast<T>(v); }
  virtual int _optionValue(Integer index) const
    { return static_cast<int>((*this)[index]); }

 private:

  UniqueArray<T> m_values;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Option du jeu de données de type étendu.
 * \ingroup CaseOption
 */
class ARCANE_CORE_EXPORT CaseOptionExtended
: public CaseOptionSimple
{
 public:

  CaseOptionExtended(const CaseOptionBuildInfo& cob,const String& type_name)
  : CaseOptionSimple(cob), m_type_name(type_name) {}

 public:

  void print(const String& lang,std::ostream& o) const override;
  ICaseFunction* function() const override { return 0; }
  void updateFromFunction(Real /*current_time*/,Integer /*current_iteration*/) override {}
  void visit(ICaseDocumentVisitor* visitor) const override;

  /*!
   * \brief Positionne la valeur par défaut de l'option.
   *
   * Si l'option n'est pas pas présente dans le jeu de données, alors sa valeur sera
   * celle spécifiée par l'argument \a def_value, sinon l'appel de cette méthode est sans effet.
   */
  void setDefaultValue(const String& def_value);

 protected:

  virtual bool _tryToConvert(const String& s) =0;
  
  void _search(bool is_phase1) override;
  bool _allowPhysicalUnit() override { return false; }

  String _typeName() const { return m_type_name; }

 private:

  String m_type_name; //!< Nom du type de l'option
  String m_value; //!< Valeur de l'option sous forme de chaîne unicode
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Option du jeu de donnée de type étendu.
 *
 * \ingroup CaseOption
 * Cette classe se sert d'une fonction externe dont le prototype est:
 
 \code
 extern "C++" bool
 _caseOptionConvert(const CaseOption&,const String&,T& obj);
 \endcode
 
 pour retrouver à partir d'une chaine de caractère un objet du type \a T.
 Cette fonction retourne \a true si un tel objet n'est pas trouvé.
 Si l'objet est trouvé, il est stocké dans \a obj.
 */
#ifndef SWIG
template<class T>
class CaseOptionExtendedT
: public CaseOptionExtended
{
 public:

  CaseOptionExtendedT(const CaseOptionBuildInfo& cob,const String& type_name)
  : CaseOptionExtended(cob,type_name) {}

 public:

  //! Valeur de l'option
  operator const T&() const { return value(); }

  //! Valeur de l'option
  const T& value() const
  {
    ARCANE_CASEOPTION_CHECK_IS_INITIALIZED;
    return m_value;
  }

  //! Valeur de l'option
  const T& operator()() const { return value(); }

  //! Retourne la valeur de l'option si isPresent()==true ou sinon \a arg_value
  const T& valueIfPresentOrArgument(const T& arg_value)
  {
    ARCANE_CASEOPTION_CHECK_IS_INITIALIZED;
    return isPresent() ? m_value : arg_value;
  }

 protected:
	
  virtual bool _tryToConvert(const String& s)
  {
    // La fonction _caseOptionConvert() doit être déclarée avant
    // l'instantiation de cette template. Normalement le générateur automatique
    // de config effectue cette opération.
    return _caseOptionConvert(*this,s,m_value);
  }

 private:

  T m_value; //!< Valeur de l'option
};
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Option du jeu de donnée de type liste de types étendus.
 * \ingroup CaseOption
 */
class ARCANE_CORE_EXPORT CaseOptionMultiExtended
: public CaseOptionBase
{
 public:

  CaseOptionMultiExtended(const CaseOptionBuildInfo& cob,const String& type_name)
  : CaseOptionBase(cob), m_type_name(type_name) {}
  ~CaseOptionMultiExtended() {}

 public:

  void print(const String& lang,std::ostream& o) const override;
  ICaseFunction* function() const override { return 0; }
  void updateFromFunction(Real /*current_time*/,Integer /*current_iteration*/) override {}
  void visit(ICaseDocumentVisitor* visitor) const override;

 protected:
  
  virtual bool _tryToConvert(const String& s,Integer pos) =0;
  virtual void _allocate(Integer size) =0;
  virtual bool _allowPhysicalUnit() { return false; }
  virtual Integer _nbElem() const =0;
  String _typeName() const { return m_type_name; } 
  void _search(bool is_phase1) override;

 private:

  String m_type_name; //!< Nom du type de l'option
  UniqueArray<String> m_values; //!< Valeurs sous forme de chaînes unicodes.
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Option du jeu de donnée de type liste de types étendus.
 * \ingroup CaseOption
 * \warning Toutes les méthodes de cette classe doivent être visible dans la
 * déclaration (pour éviter des problèmes d'instanciation de templates).
 * \sa CaseOptionExtendedT
 */
#ifndef SWIG
template<class T>
class CaseOptionMultiExtendedT
: public CaseOptionMultiExtended
, public ArrayView<T>
{
 public:

  typedef T Type; //!< Type de l'option.

 public:

  CaseOptionMultiExtendedT(const CaseOptionBuildInfo& cob,const String& type_name)
  : CaseOptionMultiExtended(cob,type_name) {}
  virtual ~CaseOptionMultiExtendedT() {} // delete[] _ptr(); }

 public:

 protected:

  bool _tryToConvert(const String& s,Integer pos) override
  {
    // La fonction _caseOptionConvert() doit être déclarée avant
    // l'instantiation de cette template. Normalement le générateur automatique
    // d'options (axl2cc) effectue cette opération.
    T& value = this->operator[](pos);
    return _caseOptionConvert(*this,s,value);
  }
  void _allocate(Integer size) override
  {
    m_values.resize(size);
    ArrayView<T>* view = this;
    *view = m_values.view();
  }
  //virtual const void* _elemPtr(Integer i) const { return this->begin()+i; }
  virtual Integer _nbElem() const override { return m_values.size(); }

 private:

  UniqueArray<T> m_values;
};
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class CaseOptionsPrivate;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Classe de base d'une liste d'options du jeu de données.
 *
 * Les instances de cette classe doivent toutes être allouées par
 * l'opérateur new() et ne doivent pas être détruite, le gestionnaire
 * de cas (ICaseMng) s'en chargeant.
 */
class ARCANE_CORE_EXPORT CaseOptions
: public ICaseOptions
{
 public:
	
  //! Construit un jeu d'options.
  CaseOptions(ICaseMng* cm,const String& name);
  //! Construit un jeu d'options.
  CaseOptions(ICaseOptionList*,const String& name);
  //! Construit un jeu d'options.
  CaseOptions(ICaseMng* cm,const String& name,const XmlNode& parent);
  //! Construit un jeu d'options.
  CaseOptions(ICaseOptionList*,const String& name,const XmlNode& parent,bool is_optional=false,bool is_multi=false);
  //! Construit un jeu d'options.
 protected:
  CaseOptions(ICaseMng*,const String& name,ICaseOptionList* parent);
  //! Construit un jeu d'options.
  CaseOptions(ICaseOptionList*,const String& name,ICaseOptionList* parent);
 public:
  //! Libère les ressources
  ~CaseOptions() override;

 private:
 
  CaseOptions(const CaseOptions& rhs) = delete;
  CaseOptions& operator=(const CaseOptions& rhs) = delete;

 public:

  //! Retourne le vrai nom (non traduit) de l'élément. 
  String rootTagTrueName() const override;

  //! Retourne le nom de l'élément dans le langage du jeu de données.
  String rootTagName() const override;

  //! Nom dans la langue \a lang de l'option. Retourne \a rootTagTrueName() si pas de traduction.
  String translatedName(const String& lang) const override;

  //! Retourne le vrai nom (non traduit) de l'élément. 
  virtual String trueName() const { return rootTagTrueName(); }

  //! Retourne le nom de l'élément dans le langage du jeu de données.
  virtual String name() const { return rootTagName(); }

  void read(bool is_phase1) override
  {
    auto p = (is_phase1) ? eCaseOptionReadPhase::Phase1 : eCaseOptionReadPhase::Phase2;
    read(p);
  }

  void read(eCaseOptionReadPhase phase) override;

  void addInvalidChildren(XmlNodeList&) override;

  void printChildren(const String& lang,int indent) override;

  //! Retourne le service associé ou 0 s'il n'y en a pas.
  IServiceInfo* caseServiceInfo() const override;

  //! Retourne le module associé ou 0 s'il n'y en a pas.
  IModule* caseModule() const override;

  //! Associe le service \a m à ce jeu de données.
  void setCaseServiceInfo(IServiceInfo* m) override;

  //! Associe le module \a m à ce jeu de données.
  void setCaseModule(IModule* m) override;

  //! Ajoute à la liste \a col tous les options filles.
  void deepGetChildren(Array<CaseOptionBase*>& col) override;

  ICaseOptionList* configList() override;

  const ICaseOptionList* configList() const override;

  //! Fonction indiquant l'état d'activation de l'option
  ICaseFunction* activateFunction() override;

  /*!
    \brief Vrai si l'option est présente dans le fichier,
    faux s'il s'agit de la valeur par défaut.
  */
  bool isPresent() const override;
  
  /*!
    \brief Ajoute une traduction pour le nom de l'option.
    Ajoute le nom \a name de l'option correspondant au langage \a lang.
    Si une traduction existe déjà pour ce langage, elle est remplacée par
    celle-ci.
  */
  void addAlternativeNodeName(const String& lang,const String& name) override;

  ICaseMng* caseMng() const override;
  ITraceMng* traceMng() const override;
  ISubDomain* subDomain() const override;
  IMesh* mesh() const override;
  MeshHandle meshHandle() const override;
  ICaseDocument* caseDocument() const override;

  void detach() override;

  void visit(ICaseDocumentVisitor* visitor) const override;

  String xpathFullName() const override;

 public:

  void addReference() override;
  void removeReference() override;

 protected:

  void _setTranslatedName();
  bool _setMeshHandleAndCheckDisabled(const String& mesh_name);

 protected:

  CaseOptionsPrivate* m_p; //!< Implémentation

 private:

  void _setMeshHandle(const MeshHandle& handle);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
