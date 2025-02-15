// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* CaseOptionSimple.h                                          (C) 2000-2025 */
/*                                                                           */
/* Option simple du jeu de données.                                          */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CASEOPTIONSIMPLE_H
#define ARCANE_CASEOPTIONSIMPLE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/ICaseOptions.h"
#include "arcane/core/CaseOptionBase.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

class IStandardFunction;
class IPhysicalUnitConverter;
class StringDictionary;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

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

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
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
  ARCANE_DEPRECATED_LONG_TERM("Y2022: Do not access XML item from option")
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

  static String _convertFunctionRealToString(ICaseFunction* func,Real t);
  static String _convertFunctionIntegerToString(ICaseFunction* func,Integer t);

 private:

  XmlNode m_element; //!< Element de l'option
  ICaseFunction* m_function = nullptr; //!< Fonction associée (ou nullptr)
  IStandardFunction* m_standard_function = nullptr; //!< Fonction standard associée (ou nullpt)
  //! Convertisseur d'unité (nullptr si pas besoin). Valide uniquement pour les options 'Real'
  IPhysicalUnitConverter* m_unit_converter = nullptr;
  bool m_changed_since_last_iteration = false;
  bool m_is_optional = false;
  bool m_has_valid_value = false;
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
  bool isPresent() const { return !m_view.empty(); }

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

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
